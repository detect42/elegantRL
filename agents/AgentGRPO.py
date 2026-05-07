from __future__ import annotations
import time
import torch as th
import numpy as np
import math
import random
from typing import Tuple, Dict, List, Any, Optional
from omegaconf import DictConfig
from .AgentBase import AgentBase, ActorBase, build_mlp, layer_init_with_orthogonal
import numba as nb

TEN = th.Tensor

# ==============================================================================
# Numba 极速探索：行为策略 (Behavior) 与 目标策略 (Target) 严格分离
# ==============================================================================
@nb.njit(fastmath=True, cache=True)
def _numba_get_action_and_logprob(logits: np.ndarray, temp: float, greedy_eps: float) -> Tuple[int, float]:
    dim = logits.shape[0]
    
    # --- 第一部分：带温度与噪音的采样 (Behavior) ---
    max_val_sample = -1e9
    scaled = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        v = logits[i]
        if v < -5.0: v = -5.0
        elif v > 5.0: v = 5.0
        v = v / temp  
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

    # --- 第二部分：纯净对数概率计算 (Target) ---
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
            
    base_prob = action_exp / sum_exp_base
    logprob = np.log(base_prob + 1e-8)
    
    return action, logprob

# ==============================================================================
# GRPO Actor (与 Reinforce 结构一致，但去掉了 Categorical 依赖)
# ==============================================================================
class ActorDiscreteGRPO(ActorBase):
    def __init__(self, state_dim: int, action_dim: int, cfg: DictConfig):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        dropout_p = cfg.dropout_p
        self.net = build_mlp(dims=[state_dim, *cfg.mlp_args.net_dims, action_dim], dropout_p=dropout_p)
        layer_init_with_orthogonal(self.net[-1], std=0.5)
        self.onnx_session = cfg.onnx_session
   
    def _probs(self, state: TEN, temperature: float, greedy_eps: float) -> TEN:
        logits = self.net(state)
        #a_prob = th.softmax(th.clamp(logits, -5.0, 5.0) / temperature, dim=-1)
        a_prob = th.softmax(logits / temperature, dim=-1)
        if greedy_eps > 0.0:
            a_prob = a_prob * (1.0 - greedy_eps) + greedy_eps / self.action_dim
        if random.random() < 0.000001:  #! debug
            print("Logits:", logits)
            print("Probs:", a_prob)
        return a_prob

    def forward(self, state: TEN) -> TEN: 
        return self.net(state).argmax(dim=-1)

    def get_action(self, state: TEN, temperature: float, greedy_eps: float) -> Tuple[TEN, TEN]: 
        if self.onnx_session:
            state_np = state.cpu().numpy()
            if state_np.ndim == 1:
                state_np = state_np.reshape(1, -1)
                
            logits_np = self.onnx_session.run(None, {'state': state_np})[0].reshape(-1) 
            action_np, logprob_np = _numba_get_action_and_logprob(logits_np, temperature, greedy_eps)
            
            action = th.as_tensor(action_np, dtype=th.int64, device=state.device)
            logprob = th.as_tensor(logprob_np, dtype=th.float32, device=state.device)
            return action, logprob
        else:
            a_prob_sample = self._probs(state, temperature, greedy_eps)
            action = th.multinomial(a_prob_sample, 1).squeeze(-1)
            # 纯净概率计算
            a_prob_base = self._probs(state, temperature=1.0, greedy_eps=0.0)
            selected_prob = a_prob_base.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            logprob = th.log(selected_prob + 1e-8)
            return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> Tuple[TEN, TEN]: 
        # 网络更新时，始终用纯净分布
        a_prob = self._probs(state, temperature=1.0, greedy_eps=0.0) 
        action_long = action.long()
        selected_prob = a_prob.gather(-1, action_long.unsqueeze(-1)).squeeze(-1)
        logprob = th.log(selected_prob + 1e-8)
        entropy = -(a_prob * th.log(a_prob + 1e-8)).sum(dim=-1)
        return logprob, entropy
    
    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.long()


# ==============================================================================
# Agent GRPO
# ==============================================================================
class AgentGRPO(AgentBase):
    act: ActorDiscreteGRPO 

    def __init__(self, state_dim: int, action_dim: int, gpu_id: int, args: DictConfig):
        super().__init__(state_dim, action_dim, gpu_id, args)

        self.act = ActorDiscreteGRPO(state_dim, action_dim, args.agent.actor).to(self.device)
        wd = args.agent.weight_decay
        self.act_optimizer = th.optim.Adam(self.act.parameters(), args.agent.actor_learning_rate, weight_decay=wd)
        self.group_size = args.agent.group_size
        self.mode = args.agent.mode  # "grpo"
        self.ratio_clip_upper = args.agent.ratio_clip_upper
        self.ratio_clip_lower = args.agent.ratio_clip_lower
        self.lambda_entropy = args.agent.lambda_entropy
        self.valid_count: int = 0

    def _explore_one_env(self, env, horizon_len: int, task_config: Dict[str, Any]) -> Tuple[TEN, ...]:
        # 1) 获取 GRPO 的组内同步 Seed 及探索参数
        group_seed = task_config["group_seed"]
        train_temp_tau = task_config["temperature"]
        train_greedy_eps = task_config["greedy_eps"]

        # ⚡ 建立专属摇号机，保证组内 N 人抽到的序列绝对同步！
        case_rng = random.Random(group_seed)
        pool_size = env.train_pool_size

        # 2) 预分配
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32, device=self.device)
        actions = th.zeros((horizon_len,), dtype=th.int32, device=self.device) if self.if_discrete else th.zeros((horizon_len, self.action_dim), dtype=th.float32, device=self.device)
        logprobs = th.zeros(horizon_len, dtype=th.float32, device=self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32, device=self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool, device=self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool, device=self.device)
        ids = th.zeros(horizon_len, dtype=th.long, device=self.device)
        target_position_vols = th.zeros(horizon_len, dtype=th.float32, device=self.device)

        # 3) 同步起步点
        initial_set_id = case_rng.randint(0, pool_size - 1)
        state_ary, _ = env.reset(mode="train", set_id=initial_set_id)
        state = th.as_tensor(state_ary, dtype=th.float32, device=self.device).unsqueeze(0)

        convert = self.act.convert_action_for_env
        last_ep_start_idx = 0

        import time
        #t_1 = 0.0
        #t_2 = 0.0
        #t_3 = 0.0
        #t_begin = time.time()
        for t in range(horizon_len):
            #t1 = time.time()
            action, logprob = self.explore_action(state, temperature=train_temp_tau, greedy_eps=train_greedy_eps)
            #t_1 += time.time() - t1
            states[t] = state.squeeze(0)
            actions[t] = action.squeeze(0) if (self.if_discrete and action.ndim > 0) else action
            logprobs[t] = logprob.squeeze(0) if logprob.ndim > 0 else logprob

            ary_action = convert(action).item()
            #t2 = time.time()
            next_state_ary, reward, terminal, truncate, info = env.step(ary_action)
            #t_2 += time.time() - t2
            current_id = int(info["absolute_id"])
            current_target_position = float(info["target_position"])

            rewards[t] = float(reward)
            terminals[t] = bool(terminal)
            truncates[t] = bool(truncate)
            ids[t] = current_id
            target_position_vols[t] = current_target_position

            if terminal or truncate:
                # ⚡ 结束时，使用同步摇号机抽取下一个 case
                next_set_id = case_rng.randint(0, pool_size - 1)
                #t3 = time.time()
                next_state_ary, _ = env.reset(mode="train", set_id=next_set_id)
                last_ep_start_idx = t + 1 
                #t_3 += time.time() - t3

            state = th.as_tensor(next_state_ary, dtype=th.float32, device=self.device).unsqueeze(0)
        #print(f"Exploration Time: action sampling {t_1:.3f}s, env stepping {t_2:.3f}s", f"resets {t_3:.3f}s.")
        #print(f"Total Exploration Time: {time.time() - t_begin:.3f}s for {horizon_len} steps.")
        if last_ep_start_idx < horizon_len:
            last_step_done = bool(terminals[horizon_len - 1].item() or truncates[horizon_len - 1].item())
            if not last_step_done:
                ids[last_ep_start_idx:] = -1

        self.last_state = state

        # 统一维度
        states = states.view((horizon_len, 1, self.state_dim))
        actions = actions.view((horizon_len, 1, self.action_dim)) if not self.if_discrete else actions.view((horizon_len, 1))
        logprobs = logprobs.view((horizon_len, 1))
        rewards = (rewards * self.reward_scale).view((horizon_len, 1))
        undones = th.logical_not(terminals).view((horizon_len, 1)).float()
        unmasks = th.logical_not(truncates).view((horizon_len, 1)).float()
        ids = ids.view((horizon_len, 1))
        target_position_vols = target_position_vols.view((horizon_len, 1))
        #print("explore_time", time.time() - t_begin)

        return states, actions, logprobs, rewards, undones, unmasks, ids, target_position_vols

    def explore_action(self, state: TEN, temperature: float=1.0, greedy_eps: float=0.0) -> Tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state=state, temperature=temperature, greedy_eps=greedy_eps)
        return actions, logprobs

    def update_net(self, buffer) -> Dict[str, float]:
        self.act.train() 
        states, actions, logprobs, rewards, undones, unmasks, ids, target_position_vols = buffer
        H, N, D = states.shape

        # ==========================================================
        # 1) 计算 GRPO Advantages 和 资金 Weights
        # ==========================================================
        full_advantages, full_weights = self.get_grpo_advantages(rewards, undones, ids, target_position_vols)
        
        # ==========================================================
        # 2) 过滤不完整轨迹 (ids != -1) 与近乎无梯度的平庸样本 (abs(A) > 0.05)
        # ==========================================================
        valid_mask = (ids != -1) & (th.abs(full_advantages) > 5)  #! 这里多筛一点
        self.valid_count = int(valid_mask.sum().item())
        
        if self.valid_count <= 1:
            print("[GRPO] Not enough valid differential samples. Skipping update.")
            return {
                "obj_actor_avg": 0.0, "obj_critic_avg": -1.0, 
                "obj_entropy_avg": 0.0, "valid_ratio": 0.0
            }

        train_states = states[valid_mask] 
        train_actions = actions[valid_mask] 
        train_logprobs = logprobs[valid_mask] 
        train_advantages = full_advantages[valid_mask] 
        train_weights = full_weights[valid_mask] 

        old_std = train_advantages.std().item()

        # ==================== 新增：计算并打印每 5% 的分位数 ====================
        # 生成 0.0 到 1.0 的 21 个分位点 (即 0%, 5%, ..., 100%)
        q = th.linspace(0, 1, steps=21, device=train_advantages.device)
        
        # 计算分位数 (转为float32以防半精度下 quantile 函数报错)
        quantiles = th.quantile(train_advantages.float(), q)
        
        # 格式化打印输出
        print("--- ADV Distribution Before Normalization (0% to 100% by 5%) ---")
        quantile_strs = [f"{int(p*100):>3}%: {v:>7.2f}" for p, v in zip(q.tolist(), quantiles.tolist())]
        
        # 为了避免刷屏，每 7 个换行打印 (可根据个人喜好调整)
        for i in range(0, len(quantile_strs), 7):
            print(" | ".join(quantile_strs[i:i+7]))
        # =========================================================================

        # ⚡ GRPO 全局缩放防爆炸 (注意：组内不除以 Std 才能保留资金规模的重要性！)
        print("std=", train_advantages.std().item(),"mean=", train_advantages.mean().item())
        train_advantages = train_advantages / (train_advantages.std() + 1e-8) 
        print("ADV\n", train_advantages.flatten()[:15].tolist())
        clean_buffer = (train_states, train_actions, train_logprobs, train_advantages, train_weights)

        # ==========================================================
        # 3) Random Batch PPO Update
        # ==========================================================
        obj_actors, obj_entropies = [], []
        th.set_grad_enabled(True)
        """update_times = int(max(1, self.valid_count * self.repeat_times / self.batch_size))
        for update_t in range(update_times):
            obj_actor, obj_entropy = self.update_objectives(clean_buffer,update_t)
            obj_actors.append(obj_actor)
            obj_entropies.append(obj_entropy)"""
        for epoch in range(self.repeat_times):
            shuffled_indices = th.randperm(self.valid_count, device=train_states.device)
            for start_idx in range(0, self.valid_count, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.valid_count)
                mb_indices = shuffled_indices[start_idx:end_idx]
                mb_buffer = (
                    train_states[mb_indices],
                    train_actions[mb_indices],
                    train_logprobs[mb_indices],
                    train_advantages[mb_indices],
                    train_weights[mb_indices]
                )
                #print(epoch, start_idx, end_idx)
                #print(train_states[mb_indices].shape)
                obj_actor, obj_entropy = self.update_objectives(mb_buffer,-1)
                obj_actors.append(obj_actor)
                obj_entropies.append(obj_entropy)

        th.set_grad_enabled(False)
        return {
            "obj_actor_avg": float(np.mean(obj_actors)) if obj_actors else 0.0,
            "obj_critic_avg": -1.0,
            "obj_entropy_avg": float(np.mean(obj_entropies)) if obj_entropies else 0.0,
            "valid_ratio": self.valid_count / (H * N),
        }

    def update_objectives(self, mb_buffer: Tuple[TEN, ...], update_t: int) -> Tuple[float, float]:
        mb_states, mb_actions, mb_old_logprobs, mb_advantages, mb_weights = mb_buffer
        new_logprobs, entropy = self.act.get_logprob_entropy(mb_states, mb_actions)
        ratio = (new_logprobs - mb_old_logprobs).exp()

        # =================================================================
        # 📊 打印 Ratio 分布监控 (The "Clip Fraction" Detector)
        # =================================================================
        if random.random() < 0.01:
            with th.no_grad():
                r_np = ratio.detach().cpu().numpy()
                # 生成 Bin 边界: [-inf, 0.50, 0.55, ..., 1.50, inf]
                edges = [-np.inf] + [round(0.5 + i * 0.05, 2) for i in range(21)] + [np.inf]
                # 统计每个区间的数量并转为百分比
                counts, _ = np.histogram(r_np, bins=edges)
                pcts = (counts / len(r_np)) * 100
                # 生成标签
                labels = ["<0.50"] + [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(1, 21)] + [">1.50"]
                # 拼接非零区间，打造极其干净的监控输出
                dist_str = " | ".join([f"{lbl}: {p:.1f}%" for lbl, p in zip(labels, pcts) if p > 0.05])
                print(f"[{update_t}] Ratio Dist: {dist_str}")
        # =================================================================

        surr1 = ratio * mb_advantages
        surr2 = ratio.clamp(1 - self.ratio_clip_lower, 1 + self.ratio_clip_upper) * mb_advantages

        obj_entropy = entropy.mean()
        
        # 结合资金量权重的 Actor Loss
        loss_actor = -(th.min(surr1, surr2) * mb_weights).mean()
        loss_actor = loss_actor - self.lambda_entropy * obj_entropy
        
        self.optimizer_backward(self.act_optimizer, loss_actor)
        return loss_actor.item(), obj_entropy.item()


    def get_grpo_advantages(self, rewards: TEN, undones: TEN, ids: TEN, target_position_vols: TEN) -> Tuple[TEN, TEN]:
            """
            ⚡ 真正的 GRPO 核心：横向比较，自我超越！
            [修复版] 严格保证 1条轨迹（无论走多少步）只拥有 1个权重，精确计算组内 Mean。
            """
            horizon_len, num_seq = rewards.shape

            # ==========================================================
            # 1) 精准标记 Episode 的终点 (is_end) 和起点 (is_start)
            # ==========================================================
            # undones 为 0 代表正常结束 (terminal), unmasks 为 0 代表超时截断 (truncate)
            is_end = (undones == 0.0)
            
            is_start = th.zeros_like(undones, dtype=th.bool)
            is_start[0, :] = True
            is_start[1:, :] = is_end[:-1, :] # 如果上一步结束了，这一步绝对是新起点

            # ==========================================================
            # 2) 算出准确的 Total Return 铺平 (G_rtg 和 G_flat)
            # ==========================================================
            G_rtg = th.zeros_like(rewards)
            curr_g = th.zeros((num_seq,), dtype=rewards.dtype, device=rewards.device)
            
            for t in range(horizon_len - 1, -1, -1):
                # 如果是终点，curr_g 直接重置为这一步的 reward
                curr_g = rewards[t] + curr_g * (~is_end[t]).float()
                G_rtg[t] = curr_g

            G_flat = th.zeros_like(rewards)
            G_flat[0] = G_rtg[0]
            for t in range(1, horizon_len):
                mask_start = is_start[t].float()
                # 如果是新起点，采用它的总收益；否则继承上一步的收益
                G_flat[t] = G_rtg[t] * mask_start + G_flat[t - 1] * (1.0 - mask_start)

            # ==========================================================
            # 3) 提取 Unique ID，并在【轨迹级别】而不是步级别上做聚合
            # ==========================================================
            print(ids.shape)
            ids_flat = ids.view(-1)
            unique_ids, step_inverse_indices = th.unique(ids_flat, return_inverse=True)
            num_unique = unique_ids.numel()

            # ⚡ 神级操作：用 is_start 掩码，全网只抽取每条轨迹的第 1 次总回报！
            ep_returns = G_flat[is_start] 
            # 同时提取这些轨迹对应的“班级索引”
            ep_inverse_indices = step_inverse_indices.view(horizon_len, num_seq)[is_start]

            # 现在，在“班级”里做累加。因为只抽取了起点，这里是严格的“1条轨迹加1次”！
            sum_g = th.zeros(num_unique, dtype=G_flat.dtype, device=self.device)
            sum_g.scatter_add_(0, ep_inverse_indices, ep_returns)
            
            # 统计班级人数（此时 counts 严格等于该 Case 的【总独立轨迹数】）
            counts = th.bincount(ep_inverse_indices, minlength=num_unique).float()
            
            # 算出平行宇宙同行的均值 (Baseline)
            mean_g = sum_g / counts.clamp_min(1.0)
            
            # 将均值广播回步级别的二维数组 (让轨迹里的每一步都减去相同的组均值)
            mean_g_expanded = mean_g[step_inverse_indices]
            th.set_printoptions(precision=2, sci_mode=False)
            print(num_unique, "unique groups in this batch.")
            print("Group counts (should be >= group_size):", counts.cpu().numpy())
            print(mean_g)
            # ==========================================================
            # 4) 计算 Advantage 及掩码清洗
            # ==========================================================
            G_flat_1d = G_flat.view(-1)
            A_1d = G_flat_1d - mean_g_expanded
            
            # 清洗残缺组：排除未完成段 (-1) 以及人数凑不齐 group_size 的被截断组
            valid_group_mask = (unique_ids != -1) & (counts >= self.group_size)
            valid_sample_mask = valid_group_mask[step_inverse_indices]
            
            # m1
            """valid_advs = A_1d[valid_sample_mask]
            if valid_advs.numel() > 0:
                lower_bound = th.quantile(valid_advs, 0.01)
                upper_bound = th.quantile(valid_advs, 0.99)
                A_1d = th.clamp(A_1d, min=lower_bound, max=upper_bound)"""

            # m2
            # A_1d = th.sign(A_1d) * th.sqrt(th.abs(A_1d))
            A_1d[~valid_sample_mask] = 0.0
            Advantages = A_1d.view(horizon_len, num_seq)

            # ==========================================================
            # 5) Debug 打印与返回
            # ==========================================================
            # th.set_printoptions(precision=2, sci_mode=False)
            # print(f"[{num_unique} unique cases] True Trajectory Counts per Case: {counts.cpu().numpy()}")
            
            Weights = th.ones_like(target_position_vols)
            return Advantages, Weights