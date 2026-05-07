from __future__ import annotations
import torch as th
import numpy as np
import time
import random
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
        wd = args.agent.weight_decay
        self.act_optimizer = th.optim.Adam(self.act.parameters(), args.agent.actor_learning_rate, weight_decay=wd)

        # 2. 算法模式配置
        self.mode = args.agent.mode  # "multicase"

        # 3. REINFORCE 专用：历史 Baseline 字典
        self.baseline_ema: Dict[int, float] = {}
        self.baseline_alpha = args.agent.baseline_alpha

        # 4. 训练超参
        self.ratio_clip_upper = args.agent.ratio_clip_upper
        self.ratio_clip_lower = args.agent.ratio_clip_lower

        self.lambda_entropy = args.agent.lambda_entropy

        self.valid_count: int = 0

    def _explore_one_env(self, env, horizon_len: int, task_config: Dict[str, Any]) -> Tuple[TEN, ...]:
        """
        Fixed Horizon Collection.
        严格采集 horizon_len 步数据。
        若最后一段 episode 未在 horizon 内完成，则把该段所有 step 的 id 标记为 -1（用于训练时丢弃）。
        """
        train_temp_tau = task_config["temperature"]
        train_greedy_eps = task_config["greedy_eps"]

        # 2) 预分配固定大小 Tensor（不带 num_envs 维度）
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32, device=self.device)

        if not self.if_discrete:
            actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32, device=self.device)
        else:
            actions = th.zeros((horizon_len,), dtype=th.int32, device=self.device)

        logprobs = th.zeros(horizon_len, dtype=th.float32, device=self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32, device=self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool, device=self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool, device=self.device)

        # 元数据：ID 和 target_position
        ids = th.zeros(horizon_len, dtype=th.long, device=self.device)
        target_position_vols = th.zeros(horizon_len, dtype=th.float32, device=self.device)

        # 3) reset（不从 last_state 续跑）
        state_ary, _ = env.reset(mode="train")
        assert isinstance(state_ary, np.ndarray)
        assert state_ary.shape == (self.state_dim,)
        state = th.as_tensor(state_ary, dtype=th.float32, device=self.device).unsqueeze(0)  # (1, D)

        convert = self.act.convert_action_for_env

        # 记录当前 horizon 内“最后一个 episode 的起点”，用于最终标记不完整尾段
        last_ep_start_idx = 0

        for t in range(horizon_len):
            # A) sample action
            action, logprob = self.explore_action(state, temperature=train_temp_tau,greedy_eps=train_greedy_eps)
    
            # 记录轨迹数据
            states[t] = state.squeeze(0)
            actions[t] = action.squeeze(0) if (self.if_discrete and action.ndim > 0) else action
            logprobs[t] = logprob.squeeze(0) if logprob.ndim > 0 else logprob

            # B) step env
            #ary_action = convert(action).detach().cpu().numpy() #! 这个会保留[]维度，适合在gpu上并行
            ary_action = convert(action).item() #! 标准的单核rollout只有一个action标量
            next_state_ary, reward, terminal, truncate, info = env.step(ary_action)

            # 记录元数据
            if "absolute_id" not in info or "target_position" not in info:
                raise KeyError("Env must return 'absolute_id' and 'target_position' in info dict.")
            current_id = int(info["absolute_id"])
            current_target_position = float(info["target_position"])

            rewards[t] = float(reward)
            terminals[t] = bool(terminal)
            truncates[t] = bool(truncate)
            ids[t] = current_id
            target_position_vols[t] = current_target_position

            # C) done/truncate -> reset，并更新下一条 episode 的起点 index
            if terminal or truncate:
                next_state_ary, _ = env.reset(mode="train")
                last_ep_start_idx = t + 1  # 下一步（若存在）是新 episode 的起点

            # 更新 state
            state = th.as_tensor(next_state_ary, dtype=th.float32, device=self.device).unsqueeze(0)

        # 5) 若最后一段 episode 不完整（最后一步不是 done/trunc），则标记其 id = -1
        #    last_ep_start_idx 可能等于 horizon_len（刚好在最后一步 reset 了），此时不需要标记
        if last_ep_start_idx < horizon_len:
            last_step_done = bool(terminals[horizon_len - 1].item() or truncates[horizon_len - 1].item())
            if not last_step_done:
                ids[last_ep_start_idx:] = -1

        # 更新 last_state（虽然你这里每次 reset 起步，但保留一致性）
        self.last_state = state

        # 6) 统一变换维度以匹配 (H, N, ...)
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

    def explore_action(self, state: TEN, temperature: float=1.0, greedy_eps: float=0.0) -> Tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state=state, temperature=temperature, greedy_eps=greedy_eps)
        return actions, logprobs

    def update_net(self, buffer) -> Dict[str, float]:
        self.act.train()  #! 这里确保 update_net 时 Actor 处于训练模式（启用 dropout 等）
        """
        buffer shapes（explore 末尾已统一）：
        states: (H, N, D)
        actions: (H, N)  或 (H, N, A)
        logprobs: (H, N)
        rewards: (H, N)
        undones: (H, N)
        unmasks: (H, N)
        ids: (H, N)  (id=-1 表示不完整尾段)
        target_position_vols: (H, N)
        """
        states, actions, logprobs, rewards, undones, unmasks, ids, target_position_vols = buffer

        # =========================
        # 0) Assert shapes only
        # =========================
        assert states.ndim == 3, f"states.ndim={states.ndim}, expect 3 (H,N,D)"
        H, N, D = states.shape
        assert D == self.state_dim, f"states last dim {D} != state_dim {self.state_dim}"

        assert ids.shape == (H, N), f"ids.shape={ids.shape}, expect (H,N)=({H},{N})"
        assert logprobs.shape == (H, N), f"logprobs.shape={logprobs.shape}, expect (H,N)=({H},{N})"
        assert rewards.shape == (H, N), f"rewards.shape={rewards.shape}, expect (H,N)=({H},{N})"
        assert undones.shape == (H, N), f"undones.shape={undones.shape}, expect (H,N)=({H},{N})"
        assert unmasks.shape == (H, N), f"unmasks.shape={unmasks.shape}, expect (H,N)=({H},{N})"
        assert target_position_vols.shape == (
            H,
            N,
        ), f"target_position_vols.shape={target_position_vols.shape}, expect (H,N)=({H},{N})"

        if self.if_discrete:
            assert actions.shape == (H, N), f"actions.shape={actions.shape}, expect (H,N)=({H},{N}) for discrete"
        else:
            assert actions.ndim == 3, f"actions.ndim={actions.ndim}, expect 3 (H,N,A) for continuous"
            assert actions.shape[:2] == (H, N), f"actions.shape={actions.shape}, expect (H,N,A)=({H},{N},A)"
            assert (
                actions.shape[2] == self.action_dim
            ), f"actions last dim {actions.shape[2]} != action_dim {self.action_dim}"
        # ==========================================================
        # 1) advantages / weights (H, N)
        # ==========================================================
        full_advantages, full_weights = self.get_advantages(rewards, undones, ids, target_position_vols)
        assert full_advantages.shape == (H, N), f"advantages.shape={full_advantages.shape}, expect (H,N)=({H},{N})"
        assert full_weights.shape == (H, N), f"weights.shape={full_weights.shape}, expect (H,N)=({H},{N})"
        # ==========================================================
        # 2) Filter: ids != -1
        # ==========================================================
        valid_mask = (ids != -1) & (th.abs(full_advantages) > 1) #! 这里从0.01改为1，过滤小adv样本，同时看valid_count可以知道目前的策略调整率？（但是1.2的逼近也很慢）
        self.valid_count = int(valid_mask.sum().item())
        if self.valid_count <= 1:
            print("[Warning] All collected samples are invalid or cold-start (Adv=0). Skipping actor update.")
            return {
                "obj_actor_avg": 0.0,
                "obj_critic_avg": -1.0,
                "obj_entropy_avg": 0.0,
                "valid_ratio": 0.0,
            }
        #print(full_advantages[:],full_weights[:],valid_mask[:])
        # ==========================================================
        # 3) Flatten by mask
        # ==========================================================
        train_states = states[valid_mask]  # (Total, D)
        train_actions = actions[valid_mask]  # (Total,) or (Total, A)
        train_logprobs = logprobs[valid_mask]  # (Total,)
        train_advantages = full_advantages[valid_mask]  # (Total,)
        train_weights = full_weights[valid_mask]  # (Total,)
        train_ids = ids[valid_mask]  # (Total,)
        # Advantage 标准化（只对合法样本）
        #train_advantages = train_advantages / (train_advantages.std() + 1e-8) #! 不能直接对所有验本标准化，应该是所有case的结果标准化，这里一个case有很多步adv都是一样的
        old_std = train_advantages.std().item()
        # =========================================================
        # ⚡ 极速分组求均值：彻底消灭 O(N) 的 Python 循环
        # =========================================================
        # 1. 获取唯一ID和反向索引（将离散的ID映射为连续的 0 ~ num_unique-1）
        unique_train_ids, inverse_indices = th.unique(train_ids, return_inverse=True)
        num_unique = unique_train_ids.numel()

        # 2. 瞬间累加相同 id 的 advantage
        sum_advs = th.zeros(num_unique, dtype=train_advantages.dtype, device=self.device)
        sum_advs.scatter_add_(0, inverse_indices, train_advantages)

        # 3. 瞬间统计每个 id 包含的样本数
        counts = th.bincount(inverse_indices).to(self.device).float()
        #print("Unique cases in batch:", num_unique, "Total valid samples:", self.valid_count)
        # 4. 向量化相除，得到每个 case 的均值
        unique_advs_tensor = sum_advs / counts
        # =========================================================

        if num_unique > 1:
            true_std = unique_advs_tensor.std() + 1e-8
            true_mean = unique_advs_tensor.mean()
        else:
            true_std = th.tensor(1.0, device=self.device)  # 仅有一个 Case 时防 NaN
            true_mean = unique_advs_tensor[0] if num_unique == 1 else th.tensor(0.0, device=self.device)
            
        true_mean_raw = train_advantages.mean().item()

        #train_advantages = (train_advantages) / (train_advantages.std() + 1e-8)  #! 方法0，直接标准化
        train_advantages = (train_advantages - true_mean) / true_std #! 方法1，标准化 
        #train_advantages = th.sign(train_advantages) * th.log1p(th.abs(train_advantages)) #! 方法2，任保留排序但压缩极值
        clean_buffer = (train_states, train_actions, train_logprobs, train_advantages, train_weights)
        print(f"old std: {old_std:.4f}, ",f"true std: {true_std.item():.4f}, " f"true mean: {true_mean.item():.4f}", f"valid samples: {self.valid_count}")
        #print(unique_advs_tensor)
        # ==========================================================
        # 4) Random Batch 更新
        # ==========================================================
        obj_actors: List[float] = []
        obj_entropies: List[float] = []

        th.set_grad_enabled(True)
        update_times = int(max(1, self.valid_count * self.repeat_times / self.batch_size))
        for update_t in range(update_times):
            obj_actor, obj_entropy = self.update_objectives(clean_buffer, update_t)
            obj_actors.append(obj_actor)
            obj_entropies.append(obj_entropy)
        th.set_grad_enabled(False)
        obj_entropy_avg = float(np.mean(obj_entropies)) if obj_entropies else 0.0
        obj_actor_avg = float(np.mean(obj_actors)) if obj_actors else 0.0
        return {
            "obj_actor_avg": obj_actor_avg,
            "obj_critic_avg": -1.0,
            "obj_entropy_avg": obj_entropy_avg,
            "valid_ratio": self.valid_count / (H * N),
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
        new_logprobs, entropy = self.act.get_logprob_entropy(mb_states, mb_actions) #! 向量化的完成
        ratio = (new_logprobs - mb_old_logprobs).exp()
        surr1 = ratio * mb_advantages
        surr2 = ratio.clamp(1 - self.ratio_clip_lower, 1 + self.ratio_clip_upper) * mb_advantages

        obj_entropy = entropy.mean()
        loss_actor = -(th.min(surr1, surr2) * mb_weights).mean()
        loss_actor = loss_actor - self.lambda_entropy * obj_entropy
        # print(loss_actor.item(), "and",obj_entropy.item())
        self.optimizer_backward(self.act_optimizer, loss_actor)
        return (loss_actor.item(), obj_entropy.item())

    def get_advantages(
        self,
        rewards: TEN,  # (H, N)
        undones: TEN,  # (H, N) float {0,1}
        ids: TEN,  # (H, N) long, -1 表示不完整尾段
        target_position_vols: TEN,  # (H, N)
    ) -> Tuple[TEN, TEN]:
        """
        1) RTG：G_rtg[t] = r[t] + undones[t] * G_rtg[t+1]（遇到 done 断开）
        2) Total Return 铺平：G_flat 在同一条 episode 内为常数（episode 起点的 total return）
        3) Advantage = G_flat - baseline_ema[id]
        - id == -1：baseline=0，且不更新 baseline_ema（后续 update_net 会过滤掉）
        4) 保留你的 debug 输出：adv_pre.csv 与 adv.csv
        """
        horizon_len, num_seq = rewards.shape

        # ======================================================
        # Step 1: Return-to-Go (倒序)
        # ======================================================
        G_rtg = th.zeros_like(rewards)
        curr_g = th.zeros((num_seq,), dtype=rewards.dtype, device=rewards.device)

        for t in range(horizon_len - 1, -1, -1):
            mask = undones[t]  # (N,)
            curr_g = rewards[t] + curr_g * mask
            G_rtg[t] = curr_g

        # ======================================================
        # Step 2: Broadcast Total Return (正序)
        # ======================================================
        G_flat = th.zeros_like(rewards)
        G_flat[0] = G_rtg[0]
        for t in range(1, horizon_len):
            mask = undones[t - 1].float()  # (N,)
            G_flat[t] = G_flat[t - 1] * mask + G_rtg[t] * (1.0 - mask)

        # ======================================================
        # Debug dump 1: adv_pre.csv (rewards, undones, ids, G_rtg, G_flat)
        # ======================================================
        """try:
            os.makedirs("/code/srwang", exist_ok=True)
            r_np = rewards.detach().cpu().numpy().reshape(-1, 1)
            u_np = undones.detach().cpu().numpy().reshape(-1, 1)
            ids_np = ids.detach().cpu().numpy().reshape(-1, 1).astype(float)
            grtg_np = G_rtg.detach().cpu().numpy().reshape(-1, 1)
            gflat_np = G_flat.detach().cpu().numpy().reshape(-1, 1)
            stacked = np.hstack([r_np, u_np, ids_np, grtg_np, gflat_np])
            np.savetxt("/code/srwang/adv_pre.csv", stacked, delimiter=",", fmt="%g")
        except Exception as e:
            print(f"Failed to save adv_pre: {e}")"""

        # ======================================================
        # Step 3: Baseline & Advantage
        # ======================================================
        ids_flat = ids.view(-1)
        G_flat_1d = G_flat.view(-1)

        unique_ids, inverse_indices = th.unique(ids_flat, return_inverse=True)
        num_unique = unique_ids.numel()

        sum_g = th.zeros(num_unique, dtype=G_flat.dtype, device=self.device)
        sum_g.scatter_add_(0, inverse_indices, G_flat_1d)
        counts = th.bincount(inverse_indices).float()
        mean_g = sum_g / counts

        b_vals = th.zeros(num_unique, dtype=G_flat.dtype, device=self.device)
        unique_ids_list = unique_ids.tolist()
        mean_g_list = mean_g.tolist()

        for i in range(num_unique):
            uid_val = unique_ids_list[i]

            # id=-1：baseline=0，不更新 EMA
            if uid_val == -1:
                continue

            current_r_val = mean_g_list[i]

            if uid_val not in self.baseline_ema:
                b_val = current_r_val
                self.baseline_ema[uid_val] = current_r_val
            else:
                b_val = float(self.baseline_ema[uid_val])

            #! b_val = float(self.baseline_ema.get(uid_val, 0.0))
            b_vals[i] = b_val

            # 先用旧 b_val 算 adv，再更新 baseline（符合你“先减历史 base”）
            self.baseline_ema[uid_val] = (1.0 - self.baseline_alpha) * b_val + self.baseline_alpha * current_r_val

        Baselines = b_vals[inverse_indices].view(horizon_len, num_seq)
        Advantages = G_flat - Baselines
        """valid_advs = Advantages
        if valid_advs.numel() > 0:
            lower_bound = th.quantile(valid_advs, 0.01)
            upper_bound = th.quantile(valid_advs, 0.99)
            Advantages = th.clamp(Advantages, min=lower_bound, max=upper_bound)"""
        #m2
        #Advantages = th.sign(Advantages) * th.sqrt(th.abs(Advantages))
        # ======================================================
        # Step 4: Weights
        # ======================================================
        # avg_vol = target_position_vols.mean() + 1e-8
        # Weights = th.sqrt(target_position_vols / avg_vol).clamp(0.5, 3.0)
        Weights = th.ones_like(target_position_vols)

        # ======================================================
        # Debug dump 2: print + adv.csv (Advantages, G_flat, Baselines)
        # ======================================================
        """print("Adv info:",Advantages.mean().item(), Advantages.std().item())
        try:
            adv_np = Advantages.detach().cpu().numpy().reshape(-1, 1)
            gflat_np = G_flat.detach().cpu().numpy().reshape(-1, 1)
            baselines_np = Baselines.detach().cpu().numpy().reshape(-1, 1)
            os.makedirs("/code/srwang", exist_ok=True)
            stacked = np.hstack([adv_np, gflat_np, baselines_np])
            np.savetxt("/code/srwang/adv.csv", stacked, delimiter=",", fmt="%g")
        except Exception as e:
            print(f"Failed to save adv: {e}")"""

        return Advantages, Weights


# ==============================================================================
# Customized Actor
# ==============================================================================
import numpy as np
import numba as nb

@nb.njit(fastmath=True, cache=True)
def _numba_get_action_and_logprob(logits: np.ndarray, temp: float, greedy_eps: float) -> Tuple[int, float]:
    dim = logits.shape[0]
    
    # ==========================================
    # 第一部分：带温度的采样 (Behavior)
    # ==========================================
    max_val_sample = -1e9
    scaled = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        v = logits[i]
        if v < -5.0: v = -5.0
        elif v > 5.0: v = 5.0 #! 对clamp 阻断了梯度传播？，但这里只会是采样所以没关系，就应该clamp
        v = v / temp  # ⚡ 施加温度
        scaled[i] = v
        if v > max_val_sample:
            max_val_sample = v
            
    sum_exp_sample = 0.0
    for i in range(dim):
        e = np.exp(scaled[i] - max_val_sample)
        scaled[i] = e
        sum_exp_sample += e
        
    greedy_mul = 1.0 - greedy_eps
    greedy_add = greedy_eps / dim
    
    rand_val = np.random.random()
    cdf = 0.0
    action = dim - 1  
    
    for i in range(dim):
        p = (scaled[i] / sum_exp_sample) * greedy_mul + greedy_add
        cdf += p
        if action == dim - 1 and rand_val <= cdf:
            action = i

    # ==========================================
    # 第二部分：纯净的概率计算 (Base/Target)
    # 彻底无视刚才的 temp 和 eps，直接用原始 logits 算 action 的概率！
    # ==========================================
    max_val_base = -1e9
    for i in range(dim):
        v = logits[i]
        if v < -5.0: v = -5.0
        elif v > 5.0: v = 5.0
        if v > max_val_base:
            max_val_base = v
            
    sum_exp_base = 0.0
    action_exp = 0.0
    for i in range(dim):
        v = logits[i]
        if v < -5.0: v = -5.0
        elif v > 5.0: v = 5.0
        e = np.exp(v - max_val_base)
        sum_exp_base += e
        if i == action:
            action_exp = e
            
    # 算出本体的纯净 logprob 返回给 Buffer
    base_prob = action_exp / sum_exp_base
    logprob = np.log(base_prob + 1e-8)
    
    return action, logprob

class ActorDiscreteReinforce(ActorBase):
    def __init__(self, state_dim: int, action_dim: int, cfg: DictConfig):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        dropout_p = cfg.dropout_p
        self.net = build_mlp(dims=[state_dim, *cfg.mlp_args.net_dims, action_dim], dropout_p=dropout_p)
        layer_init_with_orthogonal(self.net[-1], std=0.5)
        self.ActionDist: type[th.distributions.Categorical] = th.distributions.Categorical
        self.onnx_session = cfg.onnx_session
        #! self.K = cfg.需要K和padding0
   
    def _probs(self, state: TEN, temperature: float, greedy_eps: float ) -> TEN:
        logits = self.net(state)
        #a_prob = th.softmax(th.clamp(logits, -5.0, 5.0) / temperature, dim=-1)#! 这里clamp的含义是完全不管那些大的，不合理，不是原以为的更大的收束到5
        a_prob = th.softmax(logits / temperature, dim=-1)
        if greedy_eps > 0.0:
            a_prob = a_prob * (1.0 - greedy_eps) + greedy_eps / self.action_dim
        if random.random() < 0.000001:  #! debug
            print("temperature: ", temperature)
            print("[action_prob]", logits, "->", a_prob)
        return a_prob

    def forward(self, state: TEN) -> TEN: #! eval的时候用这个
        return self.net(state).argmax(dim=-1)
        #return self._probs(state, temperature=1.0).argmax(dim=-1)

    def get_action(self, state: TEN, temperature: float, greedy_eps: float) -> Tuple[TEN, TEN]: # 这个是用来rollout探索的
        if self.onnx_session:
            state_np = state.cpu().numpy()#.astype(np.float32)
            if state_np.ndim == 1:
                state_np = state_np.reshape(1, -1)
                
            logits_np = self.onnx_session.run(None, {'state': state_np})[0].reshape(-1) # 确保展平为 1D
            action_np, logprob_np = _numba_get_action_and_logprob(logits_np, temperature, greedy_eps)
            
            action = th.as_tensor(action_np, dtype=th.int64, device=state.device)
            logprob = th.as_tensor(logprob_np, dtype=th.float32, device=state.device)
            return action, logprob
        else:
            a_prob_sample = self._probs(state, temperature, greedy_eps)
            action = th.multinomial(a_prob_sample, 1).squeeze(-1)
            a_prob_base = self._probs(state, temperature=1.0, greedy_eps=0.0)
            selected_prob = a_prob_base.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            logprob = th.log(selected_prob + 1e-8)
           
            return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> Tuple[TEN, TEN]: #! 在训练learner时候调用，且是一个batch的计算，所以这里的state和action都是batch的
        #! 这个函数和rollout都调用_probs得到结果，一些随机化的操作的可以在这个函数里绑定，就不用额外记是怎么随机化了（针对important sampling的logprob计算，必须保证和探索的时候的temperature一致）
        a_prob = self._probs(state, temperature=1.0,greedy_eps=0.0) #! 在计算概率的时候始终按照原始概率，仅仅是采样的时候用temperature和greedy_eps来调整采样行为，这样保证了重要性采样的正确性（因为logprob计算必须基于原始概率分布）。如果在这里也应用temperature和greedy_eps，那么计算出来的logprob就不再是原始分布下的概率了，重要性采样的权重就会出问题。
        """dist = self.ActionDist(probs=a_prob)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()"""
        action_long = action.long()
        selected_prob = a_prob.gather(-1, action_long.unsqueeze(-1)).squeeze(-1)
        logprob = th.log(selected_prob + 1e-8)
        entropy = -(a_prob * th.log(a_prob + 1e-8)).sum(dim=-1)
        return logprob, entropy
    
    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.long()
