from __future__ import annotations
import torch as th
import numpy as np
import time
from torch.distributions import Distribution
from typing import Tuple, Dict, List, Any, Optional
from omegaconf import DictConfig
from torch import nn

from .AgentBase import AgentBase, ActorBase, build_mlp, layer_init_with_orthogonal
import os

TEN = th.Tensor


class AgentReinforce(AgentBase):

    act: ActorDiscreteReinforce  # type: ignore[assignment]

    def __init__(self, state_dim: int, action_dim: int, gpu_id: int, args: DictConfig):
        super().__init__(state_dim, action_dim, gpu_id, args)

        # 1. 初始化 Actor (使用自定义的 Discrete Actor)
        self.act = ActorDiscreteReinforce(state_dim, action_dim, args.agent.actor).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), args.agent.actor_learning_rate)

        # 2. 算法模式配置
        self.mode = args.agent.mode  # "REINFORCE" 或 "GRPO"

        # 3. REINFORCE 专用：历史 Baseline 字典
        self.baseline_ema: Dict[int, float] = {}
        self.baseline_alpha = args.agent.baseline_alpha

        # 4. 训练超参
        self.ratio_clip = args.agent.ratio_clip
        self.lambda_entropy = args.agent.lambda_entropy
        self.repeat_times = args.agent.repeat_times

        self.valid_count:int = 0

    def _explore_one_env(self, env, horizon_len: int, task_config: Optional[Dict] = None) -> Tuple[TEN, ...]:
        """
        Fixed Horizon Collection.
        严格采集 horizon_len 步数据。
        """
        # 1. 解析任务配置
        if task_config is not None:
            ...
            temperature = task_config.get("temperature", 1.0)
        else:
            ...
            temperature = 1.0

        # 2. 预分配固定大小的 Tensor (不带 num_envs 维度)
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = (
            th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device)
            if not self.if_discrete
            else th.zeros(horizon_len, dtype=th.int32).to(self.device)
        )
        logprobs = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        # 额外数据：ID 和 Volume
        ids = th.zeros(horizon_len, dtype=th.long).to(self.device)
        target_position_vols = th.zeros(horizon_len, dtype=th.float32).to(self.device)

        #! 如果不想从终端的地方突然开始，就必须在worker里面 env.reset()之后把last_state给初始化好
        state = self.last_state  # Shape: (1, state_dim) 在worker里面会依据num_env初始化
        convert = self.act.convert_action_for_env

        # 4. 固定长度循环
        for t in range(horizon_len):
            # A. 采样
            action, logprob = self.explore_action(state, temperature=temperature)

            # 记录数据 (利用广播或索引自动处理 state 的 (1, D) 到 states[t] 的 (D,))
            states[t] = state
            actions[t] = action
            logprobs[t] = logprob

            # B. Step
            ary_action = convert(action).detach().cpu().numpy()
            next_state_ary, reward, terminal, truncate, info = env.step(ary_action)

            # 记录元数据
            assert (
                "absolute_id" in info and "target_position" in info
            ), "Env must return 'absolute_id' and 'target_position' in info dict."
            current_id = int(info["absolute_id"])
            current_target_position = float(info["target_position"])

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate
            ids[t] = current_id
            target_position_vols[t] = current_target_position

            # C. 处理 Done
            if terminal or truncate:
                next_state_ary, info = env.reset()

            state = th.as_tensor(next_state_ary, dtype=th.float32, device=self.device).unsqueeze(0)

        # 更新 last_state 供下次使用
        self.last_state = state

        # 5. 统一变换维度以匹配 (horizon_len, 1, ...)
        states = states.view((horizon_len, 1, self.state_dim))
        actions = (
            actions.view((horizon_len, 1, self.action_dim)) if not self.if_discrete else actions.view((horizon_len, 1))
        )
        logprobs = logprobs.view((horizon_len, 1))
        rewards = (rewards * self.reward_scale).view((horizon_len, 1))
        undones = th.logical_not(terminals).view((horizon_len, 1)).float()
        unmasks = th.logical_not(truncates).view((horizon_len, 1)).float()
        ids = ids.view((horizon_len, 1))
        target_position_vols = target_position_vols.view((horizon_len, 1))

        return states, actions, logprobs, rewards, undones, unmasks, ids, target_position_vols

    def explore_action(self, state: TEN, temperature: float = 1.0) -> Tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state=state, temperature=temperature)
        return actions, logprobs

    def update_net(self, buffer) -> Dict[str, float]:
        """
        1. update_net 传入 buffer (Horizon, Num_Seq, ...)
        2. 向量化计算 Adv 并拼接到 buffer
        3. 生成 valid_mask 截断末尾脏数据，摊平为 (Total_Valid, ...)
        4. update_objectives 随机采样更新
        """
        # buffer 里的 tensor 形状均为 (Horizon, Num_Seq, ...)
        states, actions, logprobs, rewards, undones, unmasks, ids, target_position_vols = buffer

        # ==========================================================
        # 1. & 2. 向量化计算 Advantage 和 Weights
        # ==========================================================
        # 这一步返回的 adv 和 weight 形状也是 (Horizon, Num_Seq)
        full_advantages, full_weights = self.get_advantages(rewards, undones, ids, target_position_vols)

        # ==========================================================
        # 3. 生成 Mask 并 截断+摊平 (Flatten with Mask)
        # ==========================================================
        # 我们需要找到每个 Sequence 最后一个 done (0.0) 的位置
        # 在这之后的数据是未完成的轨迹，需要丢弃

        horizon_len, num_seq = undones.shape

        # 技巧：找到每一列最后一个 0 的位置
        # 1. 翻转时间轴，找第一个 0
        flipped_undones = undones.flip(dims=[0])
        is_done = flipped_undones == 0.0

        # 2. 检查每一列是否有 done
        has_done = is_done.any(dim=0)  # (Num_Seq,)

        # 3. 找到翻转后第一个 0 的索引 (argmax 返回第一个 True 的索引)
        first_zero_idx_flipped = is_done.float().argmax(dim=0)

        # 4. 换算回原始索引：last_done_idx = (H - 1) - first_zero_idx
        last_done_idx = (horizon_len - 1) - first_zero_idx_flipped

        # 5. 构建 Mask (Horizon, Num_Seq)
        # indices: (Horizon, 1)
        indices = th.arange(horizon_len, device=self.device).unsqueeze(1)
        # 广播比较: index <= last_done_idx 且 该列确实包含 done
        valid_mask = (indices <= last_done_idx) & has_done.unsqueeze(0)

        # 检查是否全空
        self.valid_count = int(valid_mask.sum().item())
        if self.valid_count == 0:
            raise ValueError("All collected trajectories are invalid (no completed episodes).")

        # 6. 使用 Mask 提取并摊平所有合法数据
        # 结果形状: (Total_Valid_Samples, Dim)
        train_states = states[valid_mask]  # 自动展平
        train_actions = actions[valid_mask]
        train_logprobs = logprobs[valid_mask]
        train_advantages = full_advantages[valid_mask]
        train_weights = full_weights[valid_mask]

        # 标准化 Advantage (对所有合法数据做 Norm)
        #! 假设batch够大，相当于用std做了下scale
        train_advantages = (train_advantages) / (train_advantages.std() + 1e-8)

        # 打包干净的数据
        clean_buffer = (train_states, train_actions, train_logprobs, train_advantages, train_weights)

        # ==========================================================
        # 4. Random Batch 更新
        # ==========================================================
        obj_actors = []
        obj_entropies = []

        th.set_grad_enabled(True)
        update_times = int(max(1, self.valid_count * self.repeat_times / self.batch_size))
        for update_t in range(update_times):
            # 直接传入干净的 buffer，里面全是合法的，直接抽样即可
            obj_actor, obj_entropy = self.update_objectives(clean_buffer, update_t)
            obj_actors.append(obj_actor)
            obj_entropies.append(obj_entropy)
        th.set_grad_enabled(False)

        obj_entropy_avg = np.array(obj_entropies).mean() if len(obj_entropies) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return {
            "obj_actor_avg": obj_actor_avg,
            "obj_critic_avg": -1.0,
            "obj_entropy_avg": obj_entropy_avg,
            "valid_ratio": self.valid_count / (horizon_len * num_seq),
        }

    def update_objectives(self, clean_buffer: Tuple[TEN, ...], update_t: int) -> Tuple[float, ...]:
        """
        简单的随机 Batch 采样更新
        """
        states, actions, old_logprobs, advantages, weights = clean_buffer
        assert self.valid_count == states.shape[0]

        # 直接在 [0, total_valid) 范围内随机抽样
        indices = th.randint(self.valid_count, size=(self.batch_size,), device=self.device)

        mb_states = states[indices]
        mb_actions = actions[indices]
        mb_old_logprobs = old_logprobs[indices]
        mb_advantages = advantages[indices]
        mb_weights = weights[indices]

        # PPO 计算逻辑
        new_logprobs, entropy = self.act.get_logprob_entropy(mb_states, mb_actions)

        ratio = (new_logprobs - mb_old_logprobs).exp()
        surr1 = ratio * mb_advantages
        surr2 = ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip) * mb_advantages

        obj_entropy = entropy.mean()
        loss_actor = -(th.min(surr1, surr2) * mb_weights).mean()
        loss_actor = loss_actor - self.lambda_entropy * obj_entropy

        self.optimizer_backward(self.act_optimizer, loss_actor)

        return (loss_actor.item(), obj_entropy.item())

    def get_advantages(
        self,
        rewards: TEN,  # (H, N)
        undones: TEN,  # (H, N)
        ids: TEN,  # (H, N)
        target_position_vols: TEN,  # (H, N)
    ) -> Tuple[TEN, TEN]:
        """
        1. 计算每条轨迹的 Total Return (R)。
           注意：同一条轨迹内的所有 Step，其 G 应该都等于该轨迹的总 R。
        2. R - Baseline 计算 Advantage。
        3. 更新 Baseline。
        """
        horizon_len, num_seq = rewards.shape

        # ======================================================
        # 步骤 1: 计算 Return-to-Go (倒序)
        # ======================================================
        # G_rtg[t] 表示从 t 到 轨迹结束 的累积回报
        # 对于轨迹的起点 (Start)，G_rtg[Start] 就等于 Total Return
        G_rtg = th.zeros_like(rewards)
        curr_g = th.zeros(num_seq, device=self.device)

        for t in range(horizon_len - 1, -1, -1):
            mask = undones[t].float()
            # 遇到 done (mask=0) 则重置，开始累加新轨迹
            curr_g = rewards[t] + curr_g * mask
            G_rtg[t] = curr_g

        # ======================================================
        # 步骤 2: 广播 Total Return (正序)
        # ======================================================
        # 我们需要把每个轨迹起点的 G_rtg 值，铺满整条轨迹
        G_flat = th.zeros_like(rewards)

        # t=0 时，G_rtg[0] 必然是第一条轨迹目前的 Total Return (或者部分 Return，但会被截断逻辑处理)
        G_flat[0] = G_rtg[0]

        for t in range(1, horizon_len):
            # mask=1 表示 t-1 是 undone，说明 t 和 t-1 在同一条轨迹
            # mask=0 表示 t-1 是 done，说明 t 是新轨迹的起点
            mask = undones[t - 1].float()

            # 如果是同一条轨迹，继承上一步的 G_flat (即保持 Total R 不变)
            # 如果是新轨迹起点，使用当前的 G_rtg (它包含了新轨迹的完整 R)
            G_flat[t] = G_flat[t - 1] * mask + G_rtg[t] * (1 - mask)


        # 横向拼接并保存 rewards / undones / ids / G_rtg / G_flat 到 CSV（每列为一个特征）
        try:
            os.makedirs('/code/srwang', exist_ok=True)
            r_np = rewards.detach().cpu().numpy().reshape(-1, 1)
            u_np = undones.detach().cpu().numpy().reshape(-1, 1)
            ids_np = ids.detach().cpu().numpy().reshape(-1, 1).astype(float)
            grtg_np = G_rtg.detach().cpu().numpy().reshape(-1, 1)
            gflat_np = G_flat.detach().cpu().numpy().reshape(-1, 1)
            stacked = np.hstack([r_np, u_np, ids_np, grtg_np, gflat_np])
            np.savetxt('/code/srwang/adv_pre.csv', stacked, delimiter=',', fmt='%g')
        except Exception as e:
            print(f"Failed to save adv_pre: {e}")
        # ======================================================
        # 步骤 3: 提取 Baseline 并计算 Advantage
        # ======================================================
        Advantages = th.zeros_like(G_flat)
        Baselines = th.zeros_like(G_flat)

        # 提取 Batch 中所有出现的 Unique ID
        unique_ids = ids.unique()


        #! 现在没考虑上一个非法结束截断 会在下一次开始的时候更新 造成总是开始的时候莫名其妙的重复，要把开始/结尾不合法的给删去
        for uid in unique_ids:
            uid_val = int(uid.item())

            # 找到属于该 Case 的所有位置
            id_mask = ids == uid

            # 获取该 Case 对应的 Total R
            # 注意：由于 G_flat 在同一轨迹内是常数，这里取 mean 其实就是取那个常数
            # 如果 batch 里同一个 ID 跑了多次，这里取的是多次跑的平均 Total R，这作为 EMA 更新源更稳健
            current_r_tensor = G_flat[id_mask]
            #! 显然应该是先去重再做mean （因为一个episode全是同一个id）
            current_r_val = current_r_tensor.mean().item()

            # 1. 查表获取/初始化 Baseline
            if uid_val not in self.baseline_ema:
                self.baseline_ema[uid_val] = 0
                b_val = 0
            else:
                b_val = self.baseline_ema[uid_val]

            # 2. 填入 Baseline Tensor (用于计算 Adv)
            Baselines[id_mask] = b_val

            # 3. 更新 EMA (使用当前的 R)
            # 注意：是在计算完 Adv 之后更新，还是之前？
            # 你的要求：先算 R -> 减 Base -> 更新 Base。
            # 这里 b_val 是旧的 (或刚初始化的)，符合 "先减去历史 Base"
            self.baseline_ema[uid_val] = (1 - self.baseline_alpha) * b_val + self.baseline_alpha * current_r_val

        # 4. 计算 Advantage = Total_R - Baseline
        Advantages = G_flat - Baselines

        # ======================================================
        # 步骤 4: 计算 Weights
        # ======================================================
        avg_vol = target_position_vols.mean() + 1e-8
        # Weights = th.sqrt(target_position_vols / avg_vol).clamp(0.5, 3.0)
        Weights = th.ones_like(target_position_vols)

        print(Advantages.mean().item(), Advantages.std().item())
        try:
            adv_np = Advantages.detach().cpu().numpy().reshape(-1, 1)
            gflat_np = G_flat.detach().cpu().numpy().reshape(-1, 1)
            baselines_np = Baselines.detach().cpu().numpy().reshape(-1, 1)
            os.makedirs('/code/srwang', exist_ok=True)
            stacked = np.hstack([adv_np, gflat_np, baselines_np])
            np.savetxt('/code/srwang/adv.csv', stacked, delimiter=',', fmt='%g')
        except Exception as e:
            print(f"Failed to save adv: {e}")

        return Advantages, Weights


# ==============================================================================
# Customized Actor
# ==============================================================================


class ActorDiscreteReinforce(ActorBase):
    def __init__(self, state_dim: int, action_dim: int, cfg: DictConfig):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *cfg.mlp_args.net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.5)
        self.ActionDist: type[th.distributions.Categorical] = th.distributions.Categorical
        self.greedy_eps: float = cfg.greedy_eps
        self.temp_tau: float = cfg.temp_tau

    def _probs(self, state: TEN, temperature: Optional[float] = None) -> TEN:
        logits = self.net(state)
        #! 这里有配置文件和指定的temperature两种温度参数来源打架
        tau = temperature if temperature is not None else self.temp_tau
        tau = max(float(tau), 1e-6)
        a_prob = th.softmax(th.clamp(logits, -10.0, 10.0) / tau, dim=-1)
        if self.greedy_eps > 0.0:
            a_prob = (1.0 - self.greedy_eps) * a_prob + self.greedy_eps * (1.0 / self.action_dim)
        return a_prob

    def forward(self, state: TEN) -> TEN:
        return self._probs(state, temperature=1.0).argmax(dim=-1)

    def get_action(self, state: TEN, temperature: float = 1.0) -> Tuple[TEN, TEN]:
        a_prob = self._probs(state, temperature=temperature)
        dist = self.ActionDist(probs=a_prob)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> Tuple[TEN, TEN]:
        a_prob = self._probs(state, temperature=self.temp_tau)
        dist = self.ActionDist(probs=a_prob)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.long()
