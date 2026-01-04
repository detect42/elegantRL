from __future__ import annotations
import os
import numpy as np
import torch as th
from torch import nn
from torch.nn.utils import clip_grad_norm_
from typing import Union, Optional
import random
import torch.nn.functional as F
from ..train import ReplayBuffer
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
TEN = th.Tensor

"""agent"""


class AgentBase:
    """
    The basic agent of ElegantRL

    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    act: ActorBase
    act_target: ActorBase
    cri: CriticBase
    cri_target: CriticBase

    def __init__(self, state_dim: int, action_dim: int, gpu_id: int, args: DictConfig):
        self.if_discrete: bool = args.env.if_discrete

        self.state_dim = state_dim  # feature number of state
        self.action_dim = action_dim  # feature number of continuous action or number of discrete action

        self.gamma = args.train.gamma  # discount factor of future rewards
        self.max_step = args.env.max_step  # limits the maximum number of steps an agent can take in a trajectory.
        self.num_envs = args.env.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.train.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.train.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.train.reward_scale  # an approximate target reward usually be closed to 256
        self.if_off_policy = args.agent.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.train.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.train.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.train.state_value_tau  # the tau of normalize for value and state
        self.buffer_init_size = args.train.buffer_init_size  # train after samples over buffer_init_size for off-policy

        self.last_state: TEN  # last state of the trajectory. shape == (num_envs, state_dim)
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        """optimizer"""
        self.act_optimizer: th.optim.Optimizer
        self.cri_optimizer: th.optim.Optimizer

        self.criterion = instantiate(args.agent.criterion)
        self.if_vec_env = self.num_envs > 1  # use vectorized environment (vectorized simulator)
        self.if_use_per = args.train.if_use_per  # use PER (Prioritized Experience Replay)
        self.lambda_fit_cum_r = args.train.lambda_fit_cum_r  # critic fits cumulative returns

        """save and load"""
        self.save_attr_names = {"act", "act_target", "act_optimizer", "cri", "cri_target", "cri_optimizer"}

    def explore_env(self, env, horizon_len: int) -> tuple[TEN, ...]:
        if self.if_vec_env:
            return self._explore_vec_env(env=env, horizon_len=horizon_len)
        else:
            return self._explore_one_env(env=env, horizon_len=horizon_len)

    def explore_action(self, state: TEN) -> Union[TEN, tuple[TEN, TEN]]: #! action or (action,logprob)
        return self.act.get_action(state)  #! agentbase里没有定义get_action方法

    def _explore_one_env(self, env, horizon_len: int) -> tuple[TEN, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, rewards, undones, unmasks)` for off-policy
            `num_envs == 1`
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`

        #! for on-policy algorithm like PPO, add logprob as the 3rd return value
        """
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = (
            th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device)
            if not self.if_discrete
            else th.zeros(horizon_len, dtype=th.int32).to(self.device)
        )
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        # import time#!
        # t0 = time.time()#!
        # model_time = 0#!
        state = self.last_state
        for t in range(horizon_len):
            # t_start = time.time()#!
            action = self.explore_action(state)[0]

            # if_discrete == False  action.shape (1, action_dim) -> (action_dim, )
            # if_discrete == True   action.shape (1, ) -> ()

            states[t] = state
            actions[t] = action

            ary_action = action.detach().cpu().numpy()
            # t_end = time.time()#!
            # model_time += t_end - t_start #!
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device).unsqueeze(0)

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate
        # t1 = time.time()#!
        # print(f"| Explore Total Time: {t1 - t0:.3f}s, Model Time: {model_time:.3f}s", flush=True)#!

        self.last_state = state  # state.shape == (1, state_dim) for a single env.
        """add dim1=1 below for workers buffer_items concat"""
        states = states.view((horizon_len, 1, self.state_dim))
        actions = actions.view((horizon_len, 1, self.action_dim if not self.if_discrete else 1))
        actions = (
            actions.view((horizon_len, 1, self.action_dim)) if not self.if_discrete else actions.view((horizon_len, 1))
        )
        rewards = (rewards * self.reward_scale).view((horizon_len, 1))
        undones = th.logical_not(terminals).view((horizon_len, 1))
        unmasks = th.logical_not(truncates).view((horizon_len, 1))
        return states, actions, rewards, undones, unmasks

    def _explore_vec_env(self, env, horizon_len: int) -> tuple[TEN, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, rewards, undones, unmasks)` for off-policy
            `num_envs > 1`
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
        """
        states = th.zeros((horizon_len, self.num_envs, self.state_dim), dtype=th.float32).to(self.device)
        actions = (
            th.zeros((horizon_len, self.num_envs, self.action_dim), dtype=th.float32).to(self.device)
            if not self.if_discrete
            else th.zeros((horizon_len, self.num_envs), dtype=th.int32).to(self.device)
        )
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        state = self.last_state  # last_state.shape == (num_envs, state_dim)
        for t in range(horizon_len):
            action = self.explore_action(state)
            # if_discrete == False  action.shape (num_envs, action_dim)
            # if_discrete == True   action.shape (num_envs, )

            states[t] = state  # state.shape == (num_envs, state_dim)
            actions[t] = action

            state, reward, terminal, truncate, _ = env.step(action)

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state
        rewards *= self.reward_scale
        undones = th.logical_not(terminals)
        unmasks = th.logical_not(truncates)
        return states, actions, rewards, undones, unmasks

    def update_net(self, buffer: ReplayBuffer) -> dict[str, float]:  #! on-policy算法比如ppo 这里的buffer就是tuple，不是class，所以需要在对应子类agent redefine这个function
        objs_critic = []
        objs_actor = []

        if self.lambda_fit_cum_r != 0:
            buffer.update_cum_rewards(get_cumulative_rewards=self.get_cumulative_rewards)

        th.set_grad_enabled(True)
        update_times = int(buffer.cur_size * buffer.num_seqs * self.repeat_times / self.batch_size) #! add * num_seqs
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer=buffer, update_t=update_t)
            objs_critic.append(obj_critic)
            objs_actor.append(obj_actor) if isinstance(obj_actor, float) else None
        th.set_grad_enabled(False)

        obj_avg_critic = np.array(objs_critic).mean() if len(objs_critic) else 0.0
        obj_avg_actor = np.array(objs_actor).mean() if len(objs_actor) else 0.0
        return {"obj_critic_avg": obj_avg_critic, "obj_actor_avg": obj_avg_actor}

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> tuple[float, ...]:
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state, is_weight, is_index) = buffer.sample_for_per(
                    self.batch_size
                )
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            next_action = self.act(next_state)  # deterministic policy
            next_q = self.cri_target(next_state, next_action)

            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action) * unmask
        td_error = self.criterion(q_value, q_label) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            assert is_index is not None
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        if_update_act = bool(buffer.cur_size >= self.buffer_init_size)
        if if_update_act:
            action_pg = self.act(state)  # action to policy gradient
            obj_actor = self.cri(state, action_pg).mean()
            self.optimizer_backward(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        else:
            obj_actor = th.tensor(th.nan)
        return obj_critic.item(), obj_actor.item()

    def get_cumulative_rewards(self, rewards: TEN, undones: TEN) -> TEN:
        cum_rewards = th.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            cum_rewards[t] = next_value = rewards[t] + masks[t] * next_value
        return cum_rewards

    def optimizer_backward(self, optimizer: th.optim.Optimizer, objective: TEN):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = th.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_backward_amp(self, optimizer: th.optim.Optimizer, objective: TEN):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = th.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = th.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from th.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({"act", "act_optimizer"})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"

            if getattr(self, attr_name) is None:
                continue

            if if_save:
                th.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, th.load(file_path, map_location=self.device))


def get_optim_param(optimizer: th.optim.Optimizer) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, th.Tensor)])
    return params_list


"""network"""


class ActorBase(nn.Module, ABC):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net: nn.Module  # build_mlp(net_dims=[state_dim, *net_dims, action_dim])

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ActionDist = th.distributions.normal.Normal

    @abstractmethod
    def get_action(self, state: TEN) -> Union[TEN, tuple[TEN, TEN]]:
        pass

    def forward(self, state: TEN) -> TEN:
        action = self.net(state)
        return action.tanh()


class CriticBase(nn.Module, ABC):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net: nn.Module  # build_mlp(net_dims=[state_dim + action_dim, *net_dims, 1])


    """def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        values = self.net(th.cat((state, action), dim=1))
        return values  # Q values"""


"""utils"""


# ==== TCN components & builder (channels-last: B,K,C) ====

import torch as th
from torch import nn
import torch.nn.functional as F


class _CausalConv1d_BKC(nn.Module):
    """
    因果一维卷积（封装内部转置）：
    - 输入输出均为 (B, K, C) 形式
    - 内部仅在卷积时转换为 (B, C, K)
    - 因果左补，时间长度保持不变
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 建议用奇数（如3）"
        self.pad_left = dilation * (kernel_size - 1)
        self.pad = nn.ConstantPad1d((self.pad_left, 0), 0.0)  # 作用在 (B,C,K) 上
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0, bias=bias)

    def forward(self, x_bkc: th.Tensor) -> th.Tensor:
        # x_bkc: (B, K, C) -> (B, C, K)
        x_bck = x_bkc.transpose(1, 2).contiguous()
        y_bck = self.conv(self.pad(x_bck))  # (B, out_ch, K)
        y_bkc = y_bck.transpose(1, 2).contiguous()  # (B, K, out_ch)
        return y_bkc


class RMSNorm(nn.Module):
    def __init__(self, ch, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(th.ones(ch))
        self.eps = eps

    def forward(self, x):  # x: (B,K,C)
        rms = th.sqrt(th.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.g


class _TCNBlock_BKC(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1, dropout=0.00, norm="none", use_residual=True, init_scale=0.1):
        super().__init__()
        self.norm = norm
        self.use_residual = use_residual

        if norm == "ln":
            self.n1 = nn.LayerNorm(ch)
            self.n2 = nn.LayerNorm(ch)
        elif norm == "rms":
            self.n1 = RMSNorm(ch)
            self.n2 = RMSNorm(ch)

        self.conv1 = _CausalConv1d_BKC(ch, ch, kernel_size, dilation)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = _CausalConv1d_BKC(ch, ch, kernel_size, dilation)
        self.drop2 = nn.Dropout(dropout)

        # --- 稳定化关键点 ---
        # 1) 第二个卷积权重零初始化：残差分支初始≈0，整体≈恒等
        nn.init.zeros_(self.conv2.conv.weight)
        if self.conv2.conv.bias is not None:
            nn.init.zeros_(self.conv2.conv.bias)

        # 2) 残差缩放参数（LayerScale/ReZero 思路）
        self.res_scale = nn.Parameter(th.tensor(float(init_scale))) if use_residual else None

    def _norm(self, x, which):
        if self.norm == "ln":
            return self.n1(x) if which == 1 else self.n2(x)
        if self.norm == "rms":
            return self.n1(x) if which == 1 else self.n2(x)
        return x  # 'none'

    def forward(self, x):
        y = self._norm(x, which=1)
        y = F.silu(self.conv1(y))
        # y = self.drop1(y)

        y = self._norm(y, which=2)
        y = self.conv2(F.silu(y))
        # y = self.drop2(y)

        if self.use_residual:
            return x + self.res_scale * y
        else:
            return y


class _TemporalEncoderFlat(nn.Module):
    """
    输入支持：
      (B, K, S)  或  (B, K*S)  或  (B, K*S + action_dim)  —— 统一内部按 (B, K, S) 处理
    流程：左侧零补 → 逐步嵌入 S→C → TCN blocks → 取最后一帧 (B, C)
         若带 action，则返回 cat([z, action]) -> (B, C+action_dim)
    注：全程保持 channels-last (B, K, C)，无显式外部转置
    """

    def __init__(
        self,
        state_dim: int,
        K: int,
        emb_ch: int = 32,
        num_blocks: int = 2,
        kernel_size: int = 3,
        dilations=None,
        dropout: float = 0.00,
    ):
        super().__init__()
        self.state_dim = (state_dim) // K  # S
        assert self.state_dim * K == state_dim, "state_dim 必须能被 K 整除"
        self.K = int(K)
        self.emb_ch = emb_ch

        if dilations is None:
            dilations = [1, 2][:num_blocks]
        assert num_blocks == len(dilations), "num_blocks 必须与 dilations 数量一致"

        # 逐时间步特征混合（跨特征交互）
        self.step_fc1 = nn.Linear(self.state_dim, emb_ch)
        self.step_fc2 = nn.Linear(emb_ch, emb_ch)

        # TCN 段（channels-last 版本）
        self.blocks = nn.ModuleList(
            [
                _TCNBlock_BKC(
                    emb_ch,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    norm="ln",
                    use_residual=True,
                    init_scale=0.1,
                )
                for d in dilations
            ]
        )

    @staticmethod
    def _left_pad_to_K(x_bls: th.Tensor, K: int) -> th.Tensor:
        """x:(..., L, S) → 左侧零补/裁切到 (..., K, S)"""
        *B, L, S = x_bls.shape
        if L == K:
            return x_bls
        if L > K:
            return x_bls[..., -K:, :]
        pad = x_bls.new_zeros((*B, K - L, S))
        return th.cat([pad, x_bls], dim=-2)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        只支持两种输入：
        1) (B, 135)
        2) (B, ..., 135)   # 前面任意批量维，最后一维为 135
        其他形状一律报错。
        处理逻辑：
        - 将最后一维 135 视作 L*S（S=self.state_dim），还原为 (..., L, S）
        - 若 L < K：左侧零补到 K；若 L > K：裁到最后 K 帧
        - 逐步嵌入 S->C、TCN、取最后一帧，输出 (..., C)
        """
        S = self.state_dim
        K = self.K
        rd = random.random()
        flg = rd < 0.0000001
        # 仅接受最后一维存在，且是 S 的整数倍
        if x.dim() < 2:
            raise RuntimeError(f"不支持的输入形状: {tuple(x.shape)}，期望 (B,135) 或 (B,...,135)")
        assert x.size(-1) == (S) * K
        if flg:
            print(x.shape)
        # 记录前置批量维，并展平到 (Bf, L, S)
        *batch_shape, D = x.shape  # (..., 135)
        L = (D) // S
        flat_B = int(th.tensor(batch_shape).prod().item()) if len(batch_shape) else x.shape[0]
        if flg:
            print("flat_B= ", flat_B)
        x_bls = x.view(flat_B, L, S)  # (Bf, L, S)

        # print(L,D,S)

        # 左侧零补/裁切到 (Bf, K, S)
        if L == K:
            window = x_bls
        elif L > K:
            window = x_bls[:, -K:, :]
        else:
            pad = x_bls.new_zeros((flat_B, K - L, S))
            window = th.cat([pad, x_bls], dim=1)

        # 逐步嵌入： (Bf,K,S) -> (Bf,K,C)
        h = F.silu(self.step_fc1(window))
        h = F.silu(self.step_fc2(h))
        if flg:
            print("window:", window)
            print("h", h)

        # TCN blocks：保持 (Bf,K,C)
        for blk in self.blocks:
            h = blk(h)

        # 取最后一帧： (Bf, C)
        z = h[:, -1, :]

        # 还原前置批量维：(..., C)
        if len(batch_shape):
            z = z.view(*batch_shape, z.size(-1))
        if flg:
            print("Z", z.shape, z)
        return z


def build_tcn(
    state_dim: int,
    action_dim: int,
    K: int,
    net_dims: list[int] | None = None,
    *,
    emb_ch: int = 32,
    num_blocks: int = 2,
    kernel_size: int = 3,
    dilations: list[int] | None = None,
    dropout: float = 0.00,
    activation: type[nn.Module] = nn.SiLU,
    for_q: bool = False,
    return_feature_extractor: bool = False,  # ★ 新增：True 时仅返回特征提取器
) -> nn.Sequential:
    """
    构建：Encoder_BKC (输出 (B, C) / (B, C+A)) → [可选] MLP 头
    - 输入可为 (B,K,S) / (B,K*S) / (B,K*S+action_dim)
    - for_q=False: 输出 action_dim；for_q=True: 输出 1
    - 当 return_feature_extractor=True 时，仅返回特征提取器（编码器 + 可选 LayerNorm），由上层手动接末端 Linear/MLP
    - 默认行为保持不变（return_feature_extractor=False）：返回 Encoder + 头部 MLP
    """
    if net_dims is None:
        net_dims = [256, 128, 32]
    if dilations is None:
        dilations = [1, 2][:num_blocks]

    encoder = _TemporalEncoderFlat(
        state_dim=state_dim,
        K=K,
        emb_ch=emb_ch,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    )

    in_head = emb_ch

    if return_feature_extractor:
        # 仅返回特征提取器（和 normalize），方便外部像 MLP 一样手工接头部
        return nn.Sequential(
            encoder,
            # 可按需加 LN（与之前注释一致，这里保持默认不加；如需可解开）：
            # nn.LayerNorm(in_head),
        )

    # —— 默认行为：返回完整的 Encoder + 头部（保持旧代码兼容）——
    out_dim = 1 if for_q else action_dim
    layers: list[nn.Module] = [encoder]
    dims = [in_head, *net_dims, out_dim]
    # 如需 LayerNorm，可打开
    # layers.append(nn.LayerNorm(dims[0]))

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2:
            layers.append(activation())

    return nn.Sequential(*layers)


def build_mlp(
    dims: list[int], activation: Optional[type[nn.Module]] = None, if_raw_out: bool = True, use_ln: bool = False
) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    net_dims: the middle dimension, `net_dims[-1]` is the output dimension of this network
    activation: the activation function
    if_raw_out: if True, remove the last activation function (输出层保持 raw 值)
    use_ln: 是否在输入处加 LayerNorm，默认 False
    """
    if activation is None:
        activation = nn.GELU

    net_list: list[nn.Module] = []

    # 输入层可选 LayerNorm
    if use_ln:
        net_list.append(nn.LayerNorm(dims[0]))

    # 原始 MLP 逻辑
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # 删除最后一层激活

    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = th.cat((x1, self.dense1(x1)), dim=1)
        return th.cat((x2, self.dense2(x2)), dim=1)  # x3  # x2.shape==(-1, lay_dim*4)


class ConvNet(nn.Module):  # pixel-level state encoder
    def __init__(self, inp_dim, out_dim, image_size=224):
        super().__init__()
        if image_size == 224:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 224, 224)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=110
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(96, 128, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(128, 192, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(192, out_dim),  # size==(batch_size, out_dim)
            )
        elif image_size == 112:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 112, 112)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(96, 128, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(128, out_dim),  # size==(batch_size, out_dim)
            )
        else:
            assert image_size in {224, 112}

    def forward(self, x):
        # assert x.shape == (batch_size, inp_dim, image_size, image_size)
        x = x.permute(0, 3, 1, 2)
        x = x / 128.0 - 1.0
        return self.net(x)

    @staticmethod
    def check():
        inp_dim = 3
        out_dim = 32
        batch_size = 2
        image_size = [224, 112][1]
        # from elegantrl.net import Conv2dNet
        net = ConvNet(inp_dim, out_dim, image_size)

        image = th.ones((batch_size, image_size, image_size, inp_dim), dtype=th.uint8) * 255
        print(image.shape)
        output = net(image)
        print(output.shape)
