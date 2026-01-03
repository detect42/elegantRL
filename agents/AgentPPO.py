"""
PPO algorithm + GAE
"""

from __future__ import annotations
import random
from typing import List, Optional, Tuple
from omegaconf import DictConfig
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from .AgentBase import AgentBase, build_mlp, layer_init_with_orthogonal, build_tcn, ActorBase, CriticBase

TEN = th.Tensor


class AgentPPO(AgentBase):
    """PPO algorithm + GAE
    “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.
    “Generalized Advantage Estimation”. John Schulman. et al..
    """

    act: ActorPPO  # type: ignore[assignment]
    cri: CriticPPO  # type: ignore[assignment]

    def __init__(self, state_dim: int, action_dim: int, gpu_id: int, args: DictConfig):
        super().__init__(state_dim, action_dim, gpu_id, args)
        self.act = ActorPPO(state_dim=state_dim, action_dim=action_dim, cfg=args.agent.actor).to(self.device)
        self.cri = CriticPPO(state_dim=state_dim, action_dim=action_dim, cfg=args.agent.critic).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), args.agent.actor_learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), args.agent.critic_learning_rate)

        self.ratio_clip = args.agent.ratio_clip  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = args.agent.lambda_gae_adv  # could be 0.80~0.99
        self.lambda_entropy = args.agent.lambda_entropy  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.if_use_v_trace = args.agent.if_use_v_trace  # GAE or V-trace

    def _explore_one_env(self, env, horizon_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, logprobs, rewards, undones, unmasks)` for on-policy
            num_envs == 1
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `logprobs.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
        """
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

        state = self.last_state  # shape == (1, state_dim) for a single env.
        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, logprob = [t[0] for t in self.explore_action(state)]

            states[t] = state
            actions[t] = action
            logprobs[t] = logprob

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device).unsqueeze(0)

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state  # state.shape == (1, state_dim) for a single env.
        """add dim1=1 below for workers buffer_items concat"""
        states = states.view((horizon_len, 1, self.state_dim))
        actions = (
            actions.view((horizon_len, 1, self.action_dim)) if not self.if_discrete else actions.view((horizon_len, 1))
        )
        logprobs = logprobs.view((horizon_len, 1))
        rewards = (rewards * self.reward_scale).view((horizon_len, 1))
        undones = th.logical_not(terminals).view((horizon_len, 1))
        unmasks = th.logical_not(truncates).view((horizon_len, 1))
        return states, actions, logprobs, rewards, undones, unmasks

    def _explore_vec_env(self, env, horizon_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, logprobs, rewards, undones, unmasks)` for on-policy
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `logprobs.shape == (horizon_len, num_envs, action_dim)`
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
        logprobs = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        state = self.last_state  # shape == (num_envs, state_dim) for a vectorized env.

        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, logprob = self.explore_action(state)

            states[t] = state
            actions[t] = action
            logprobs[t] = logprob

            state, reward, terminal, truncate, _ = env.step(convert(action))  # next_state

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state
        rewards *= self.reward_scale
        undones = th.logical_not(terminals)
        unmasks = th.logical_not(truncates)
        return states, actions, logprobs, rewards, undones, unmasks

    def explore_action(self, state: TEN) -> tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state)
        return actions, logprobs

    def update_net(self, buffer) -> tuple[float, float, float]:
        buffer_size = buffer[0].shape[0]

        """get advantages reward_sums"""
        with th.no_grad():
            states, actions, logprobs, rewards, undones, unmasks = buffer
            bs = max(1, 2**14 // self.num_envs)  # set a smaller 'batch_size' to avoid CUDA OOM
            values_list = [self.cri(states[i : i + bs]) for i in range(0, buffer_size, bs)]  # Q
            values = th.cat(values_list, dim=0).squeeze(-1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # avoid CUDA OOM
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        """update network"""
        obj_entropies = []
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        # print(buffer[0].shape[0] , buffer[0].shape[1], self.repeat_times, self.batch_size)
        #!update_times = int(buffer_size * self.repeat_times / self.batch_size)
        update_times = int(buffer[0].shape[0] * buffer[0].shape[1] * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor, obj_entropy = self.update_objectives(buffer, update_t)
            obj_entropies.append(obj_entropy)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_entropy_avg = np.array(obj_entropies).mean() if len(obj_entropies) else 0.0
        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return obj_critic_avg, obj_actor_avg, obj_entropy_avg

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, ...]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        ids1 = th.div(ids, sample_len, rounding_mode="floor")  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        # print(value,"\n", reward_sum)#! debug
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        new_logprob, entropy = self.act.get_logprob_entropy(state, action)
        ratio = (new_logprob - logprob.detach()).exp()

        surrogate1 = advantage * ratio
        surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        surrogate = th.min(surrogate1, surrogate2)  # save as below
        # surrogate = advantage * ratio * th.where(advantage.gt(0), 1 - self.ratio_clip, 1 + self.ratio_clip)
        obj_surrogate = (surrogate * unmask).mean()  # major actor objective
        obj_entropy = (entropy * unmask).mean()  # minor actor objective
        # print(obj_surrogate, obj_entropy, flush=True)
        obj_actor_full = obj_surrogate - obj_entropy * self.lambda_entropy
        self.optimizer_backward(self.act_optimizer, -obj_actor_full)
        return obj_critic.item(), obj_surrogate.item(), obj_entropy.item()

    def get_advantages(self, states: TEN, rewards: TEN, undones: TEN, unmasks: TEN, values: TEN) -> TEN:
        advantages = th.empty_like(values)  # advantage value

        # update undones rewards when truncated
        truncated = th.logical_not(unmasks)
        if th.any(truncated):
            rewards[truncated] += self.cri(states[truncated]).squeeze(1).detach()
            undones[truncated] = False

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = self.last_state.clone()
        next_value = self.cri(next_state).detach().squeeze(-1)

        advantage = th.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        if self.if_use_v_trace:  # get advantage value in reverse time series (V-trace)
            for t in range(horizon_len - 1, -1, -1):
                next_value = rewards[t] + masks[t] * next_value
                advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
                next_value = values[t]
        else:  # get advantage value using the estimated value of critic network
            for t in range(horizon_len - 1, -1, -1):
                advantages[t] = rewards[t] - values[t] + masks[t] * advantage
                advantage = values[t] + self.lambda_gae_adv * advantages[t]
        return advantages

    def update_avg_std_for_normalization(self, states: TEN):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = (self.act.state_std * (1 - tau) + state_std * tau).clamp_min(1e-4)
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        """self.act_target.state_avg[:] = self.act.state_avg
        self.act_target.state_std[:] = self.act.state_std
        self.cri_target.state_avg[:] = self.cri.state_avg
        self.cri_target.state_std[:] = self.cri.state_std"""


class AgentA2C(AgentPPO):
    """A2C algorithm.
    “Asynchronous Methods for Deep Reinforcement Learning”. 2016.
    """

    def update_net(self, buffer) -> tuple[float, float, float]:
        buffer_size = buffer[0].shape[0]

        """get advantages reward_sums"""
        with th.no_grad():
            states, actions, logprobs, rewards, undones, unmasks = buffer
            bs = max(1, 2**10 // self.num_envs)  # set a smaller 'batch_size' to avoid CUDA OOM
            values_list = [self.cri(states[i : i + bs]) for i in range(0, buffer_size, bs)]
            values = th.cat(values_list, dim=0).squeeze(-1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages[::4, ::4].std() + 1e-5)  # avoid CUDA OOM
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        """update network"""
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer, update_t)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return obj_critic_avg, obj_actor_avg, 0

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        buffer_size = states.shape[0]
        indices = th.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
        state = states[indices]
        action = actions[indices]
        unmask = unmasks[indices]
        # logprob = logprobs[indices]
        advantage = advantages[indices]
        reward_sum = reward_sums[indices]

        value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
        obj_actor = (advantage * new_logprob).mean()  # obj_actor without policy gradient clip
        self.optimizer_backward(self.act_optimizer, -obj_actor)
        return obj_critic.item(), obj_actor.item()


class AgentDiscretePPO(AgentPPO):
    def __init__(self, state_dim: int, action_dim: int, gpu_id: int, args: DictConfig):
        super().__init__(state_dim, action_dim, gpu_id, args)

        self.act = ActorDiscretePPO(state_dim=state_dim, action_dim=action_dim, cfg=args.agent.actor).to(self.device)
        self.cri = CriticPPO(state_dim=state_dim, action_dim=action_dim, cfg=args.agent.critic).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), args.agent.actor_learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), args.agent.critic_learning_rate)


"""network"""


class ActorPPO(ActorBase):
    def __init__(self, state_dim: int, action_dim: int, cfg: DictConfig):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        if cfg.type == "mlp":
            self.net = build_mlp(dims=[state_dim, *cfg.mlp_args.net_dims, action_dim])
            layer_init_with_orthogonal(self.net[-1], std=0.5)

        self.action_std_log = nn.Parameter(th.zeros((1, action_dim)), requires_grad=True)  # trainable parameter
        self.ActionDist = th.distributions.normal.Normal

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:  #! 暂时不启用
        # return (state - self.state_avg) / (self.state_std + 1e-4)
        return state  # do not normalize state for now

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        action = self.net(state)
        # print(action)
        return self.convert_action_for_env(action)

    def get_action(self, state: TEN) -> tuple[TEN, TEN]:  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        # print(action_avg)
        action_std = self.action_std_log.exp()
        if random.random() < 0.00001:  #!debug
            # print("action_avg (action means):", action_avg.cpu().detach().numpy())
            print("std:", action_std.cpu().detach().numpy())
        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> tuple[TEN, TEN]:
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        # return 0.04*(action.tanh()+1)
        return action.tanh()


class ActorDiscretePPO(ActorPPO):
    ActionDist: type[th.distributions.Categorical]  # type: ignore[assignment]

    def __init__(self, state_dim: int, action_dim: int, cfg: DictConfig):
        """
        离散速率策略（K 桶）。仅做三件事：
        - 温度 τ：softmax(logits / τ) 拉平
        - ε-greedy：与均匀分布混合，防塌
        - 其余保持你原框架一致（Categorical）
        """
        super().__init__(state_dim=state_dim, action_dim=action_dim, cfg=cfg)

        # ★ 连续版里无用的高斯参数，删除以免被优化器更新
        if hasattr(self, "action_std_log"):
            del self.action_std_log

        self.ActionDist = th.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

        # ★ 这两个是止塌用的超参（可在训练中退火）
        self.greedy_eps: float = cfg.greedy_eps  # 与均匀分布混合的权重 ε
        self.temp_tau: float = cfg.temp_tau  # logits 温度 τ (>1 拉平)

    def _probs(self, state: TEN) -> TEN:
        """softmax(logits/τ)，再与均匀分布做 ε 混合，并可选设地板"""
        logits = self.net(self.state_norm(state))  # [B, K]
        tau = max(float(self.temp_tau), 1e-6)
        a_prob = th.softmax(th.clamp(logits, -5.0, 5.0) / tau, dim=-1)  # 温度缩放

        if self.greedy_eps > 0.0:
            assert a_prob.size(-1) == self.action_dim
            a_prob = (1.0 - self.greedy_eps) * a_prob + self.greedy_eps * (1.0 / self.action_dim)

        if random.random() < 0.00001:
            print(logits, "\n", a_prob)
        return a_prob

    def forward(self, state: TEN) -> TEN:
        """推理：返回离散动作索引（argmax）。env 侧 idx→rate。"""
        a_prob = self._probs(state)
        return a_prob.argmax(dim=-1)

    def get_action(self, state: TEN) -> tuple[TEN, TEN]:
        """训练采样：返回索引 + logprob"""
        a_prob = self._probs(state)

        dist = self.ActionDist(probs=a_prob)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> tuple[TEN, TEN]:
        a_prob = self._probs(state)
        dist = self.ActionDist(probs=a_prob)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.long()


class CriticPPO(CriticBase):
    def __init__(self, state_dim: int, action_dim: int, cfg: DictConfig):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        assert isinstance(action_dim, int)
        # self.state_single_len: int = state_dim // K
        self.net = build_mlp(dims=[state_dim, *cfg.mlp_args.net_dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        # state = state[..., -self.state_single_len:]
        value = self.net(state)
        return value  # advantage value

    def state_norm(self, state: TEN) -> TEN:
        return state  ###!QQQQQQQQQQQQQQQQQQQQQQQQQq
