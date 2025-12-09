import numpy as np
import torch as th
from torch import nn
import random
from ..train import Config
from .AgentBase import AgentBase, build_mlp, layer_init_with_orthogonal,build_tcn
import torch.nn.functional as F
TEN = th.Tensor


class AgentPPO(AgentBase):
    """PPO algorithm + GAE
    “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.
    “Generalized Advantage Estimation”. John Schulman. et al..
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,K=args.K).to(self.device)
        critic_net_dims = [128,64,16]
        self.cri = CriticPPO(net_dims=critic_net_dims, state_dim=state_dim, action_dim=action_dim,K=args.K).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.001)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)

    def _explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
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
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros(horizon_len, dtype=th.int32).to(self.device)
        logprobs = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        state = self.last_state  # shape == (1, state_dim) for a single env.
        #! 用上一次的last_state非常危险！在目前sample case的情况下
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
        '''add dim1=1 below for workers buffer_items concat'''
        states = states.view((horizon_len, 1, self.state_dim))
        actions = actions.view((horizon_len, 1, self.action_dim)) \
            if not self.if_discrete else actions.view((horizon_len, 1))
        logprobs = logprobs.view((horizon_len, 1))
        rewards = (rewards * self.reward_scale).view((horizon_len, 1))
        undones = th.logical_not(terminals).view((horizon_len, 1))
        unmasks = th.logical_not(truncates).view((horizon_len, 1))
        return states, actions, logprobs, rewards, undones, unmasks

    def _explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
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
        actions = th.zeros((horizon_len, self.num_envs, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, self.num_envs), dtype=th.int32).to(self.device)
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

        '''get advantages reward_sums'''
        with th.no_grad():
            states, actions, logprobs, rewards, undones, unmasks = buffer
            bs = max(1, 2 ** 10 // self.num_envs)  # set a smaller 'batch_size' to avoid CUDA OOM
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)] #Q
            values = th.cat(values, dim=0).squeeze(-1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # avoid CUDA OOM
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        '''update network'''
        obj_entropies = []
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        #print(buffer[0].shape[0] , buffer[0].shape[1], self.repeat_times, self.batch_size)
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

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        #print(value,"\n", reward_sum)#! debug
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        new_logprob, entropy = self.act.get_logprob_entropy(state, action)
        ratio = (new_logprob - logprob.detach()).exp()

        surrogate1 = advantage * ratio
        surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        surrogate = th.min(surrogate1, surrogate2)  # save as below
        #surrogate = advantage * ratio * th.where(advantage.gt(0), 1 - self.ratio_clip, 1 + self.ratio_clip)
        obj_surrogate = (surrogate * unmask).mean()  # major actor objective
        obj_entropy = (entropy * unmask).mean()  # minor actor objective
        #print(obj_surrogate, obj_entropy, flush=True)
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

        self.act_target.state_avg[:] = self.act.state_avg
        self.act_target.state_std[:] = self.act.state_std
        self.cri_target.state_avg[:] = self.cri.state_avg
        self.cri_target.state_std[:] = self.cri.state_std
class AgentBetaPPO(AgentPPO):
    """
    用 Bernoulli+Beta Actor 替换原来的高斯 Actor：
    - 动作维度固定为 2（[g, m]）
    - env 侧收到的是 convert_action_for_env 映射后的标量速率
    其它流程（buffer、GAE、PPO 更新）全部沿用 AgentPPO
    """
    def __init__(self,
                 net_dims: list[int],
                 state_dim: int,
                 action_dim: int = 2,        # 忽略外部传入，固定为 2
                 gpu_id: int = 0,
                 args: Config = Config()):
        # 先用 action_dim=2 调用父类，避免形状不一致
        super().__init__(net_dims=net_dims,
                         state_dim=state_dim,
                         action_dim=2,
                         gpu_id=gpu_id,
                         args=args)

        # ---- 覆盖 Actor 为 Bernoulli+Beta 版本 ----
        use_tcn_actor = getattr(args, "use_tcn_actor", True)
        hidden_dim    = getattr(args, "actor_hidden_dim", 32)

        # Gate+Beta 的数值/稳定性超参（可在 args 中设定）
        r_max        = getattr(args, "r_max", 0.08)
        gamma_shape  = getattr(args, "gamma_shape", 1.0)
        gate_tau     = getattr(args, "gate_tau", 1.5)     # 温度 (>1 更平滑)
        gate_eps     = getattr(args, "gate_eps", 0.10)    # 与 0.5 混合，防早塌
        min_conc     = getattr(args, "min_conc", 0.20)    # Beta 浓度下界
        max_conc     = getattr(args, "max_conc", 30.0)    # Beta 浓度上界

        # 用新的 Actor 替换
        self.act = ActorBernoulliBetaPPO(
            net_dims=net_dims,
            state_dim=state_dim,
            action_dim=2,               # ★ 固定 2（[g, m]）
            use_tcn=use_tcn_actor,
            K=args.K,
            hidden_dim=hidden_dim,
            r_max=r_max,
            gamma_shape=gamma_shape,
            gate_tau=gate_tau,
            gate_eps=gate_eps,
            min_conc=min_conc,
            max_conc=max_conc,
        ).to(self.device)

        # 重新创建 actor 优化器（因为参数集变了）
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        # —— 其余配置保持不变（继承自 AgentPPO.__init__ 已设置好）——
        # self.cri 保持原 CriticPPO
        # self.ratio_clip, self.lambda_gae_adv, self.lambda_entropy, self.if_use_v_trace 等保持

        # 若外部误传了 action_dim != 2，给个友好提醒（不抛错，避免训练中断）
        if action_dim != 2:
            print(f"[AgentBetaPPO] Warning: action_dim is forced to 2 (got {action_dim}, overridden).")

class AgentA2C(AgentPPO):
    """A2C algorithm.
    “Asynchronous Methods for Deep Reinforcement Learning”. 2016.
    """

    def update_net(self, buffer) -> tuple[float, float, float]:
        buffer_size = buffer[0].shape[0]

        '''get advantages reward_sums'''
        with th.no_grad():
            states, actions, logprobs, rewards, undones, unmasks = buffer
            bs = max(1, 2 ** 10 // self.num_envs)  # set a smaller 'batch_size' to avoid CUDA OOM
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = th.cat(values, dim=0).squeeze(-1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages[::4, ::4].std() + 1e-5)  # avoid CUDA OOM
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        '''update network'''
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

class AgentBetaPPO(AgentPPO):
    """
    用 Bernoulli+Beta Actor 替换原来的高斯 Actor：
    - 动作维度固定为 2（[g, m]）
    - env 侧收到的是 convert_action_for_env 映射后的标量速率
    其它流程（buffer、GAE、PPO 更新）全部沿用 AgentPPO
    """
    def __init__(self,
                 net_dims: list[int],
                 state_dim: int,
                 action_dim: int = 2,        # 忽略外部传入，固定为 2
                 gpu_id: int = 0,
                 args: Config = Config()):
        # 先用 action_dim=2 调用父类，避免形状不一致
        super().__init__(net_dims=net_dims,
                         state_dim=state_dim,
                         action_dim=2,
                         gpu_id=gpu_id,
                         args=args)

        # ---- 覆盖 Actor 为 Bernoulli+Beta 版本 ----
        use_tcn_actor = getattr(args, "use_tcn_actor", True)
        hidden_dim    = getattr(args, "actor_hidden_dim", 32)
        # Gate+Beta 的数值/稳定性超参（可在 args 中设定）
        r_max        = getattr(args, "r_max", 0.08)
        gamma_shape  = getattr(args, "gamma_shape", 1.0)
        gate_tau     = getattr(args, "gate_tau", 1.5)     # 温度 (>1 更平滑)
        gate_eps     = getattr(args, "gate_eps", 0.10)    # 与 0.5 混合，防早塌
        min_conc     = getattr(args, "min_conc", 0.20)    # Beta 浓度下界
        max_conc     = getattr(args, "max_conc", 30.0)    # Beta 浓度上界

        # 用新的 Actor 替换
        self.act = ActorBernoulliBetaPPO(
            net_dims=net_dims,
            state_dim=state_dim,
            action_dim=2,               # ★ 固定 2（[g, m]）
            use_tcn=use_tcn_actor,
            K=args.K,
            hidden_dim=hidden_dim,
            r_max=r_max,
            gamma_shape=gamma_shape,
            gate_tau=gate_tau,
            gate_eps=gate_eps,
            min_conc=min_conc,
            max_conc=max_conc,
        ).to(self.device)

        # 重新创建 actor 优化器（因为参数集变了）
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        # —— 其余配置保持不变（继承自 AgentPPO.__init__ 已设置好）——
        # self.cri 保持原 CriticPPO
        # self.ratio_clip, self.lambda_gae_adv, self.lambda_entropy, self.if_use_v_trace 等保持

        # 若外部误传了 action_dim != 2，给个友好提醒（不抛错，避免训练中断）
        if action_dim != 2:
            print(f"[AgentBetaPPO] Warning: action_dim is forced to 2 (got {action_dim}, overridden).")

class AgentDiscretePPO(AgentPPO):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentPPO.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorDiscretePPO(
            net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
            K=getattr(args, "K", 9),
            greedy_eps=getattr(args, "greedy_eps", 0.20),
            temp_tau=getattr(args, "temp_tau", 2.0),
        ).to(self.device)

        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,K=args.K).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.ratio_clip = getattr(args, "ratio_clip", 0.20)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)


class AgentDiscreteA2C(AgentDiscretePPO):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentDiscretePPO.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorDiscretePPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)


'''network'''


class ActorPPO(th.nn.Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int, K:int):
        super().__init__()
        #self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        self.net = build_tcn(state_dim=state_dim,
                     action_dim=action_dim,
                     K=K,
                     net_dims=net_dims,
                     emb_ch=32, num_blocks=2, kernel_size=3, dilations=[1,2],
                     dropout=0.05, activation=nn.SiLU, for_q=False)
        layer_init_with_orthogonal(self.net[-1], std=0.5)

        self.action_std_log = nn.Parameter(th.zeros((1, action_dim)), requires_grad=True)  # trainable parameter
        self.ActionDist = th.distributions.normal.Normal

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)
        self.K = K

    def state_norm(self, state: TEN) -> TEN:#! 暂时不启用
        #return (state - self.state_avg) / (self.state_std + 1e-4)
        return state  # do not normalize state for now

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        action = self.net(state)
        #print(action)
        return self.convert_action_for_env(action)

    def get_action(self, state: TEN) -> tuple[TEN, TEN]:  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        #print(action_avg)
        action_std = self.action_std_log.exp()
        import random
        if random.random() < 0.00001:  #!debug
            #print("action_avg (action means):", action_avg.cpu().detach().numpy())
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
        #return 0.04*(action.tanh()+1)
        return (action.tanh())


class ActorDiscretePPO(ActorPPO):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int,
                 K: int = 9, greedy_eps: float = 0.10, temp_tau: float = 2.0):
        """
        离散速率策略（K 桶）。仅做三件事：
        - 温度 τ：softmax(logits / τ) 拉平
        - ε-greedy：与均匀分布混合，防塌
        - 其余保持你原框架一致（Categorical）
        """
        # ★ 关键：把 K 传给父类（你的 ActorPPO 需要 K）
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, K=K)

        # ★ 连续版里无用的高斯参数，删除以免被优化器更新
        if hasattr(self, "action_std_log"):
            del self.action_std_log

        self.ActionDist = th.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

        # ★ 这两个是止塌用的超参（可在训练中退火）
        self.greedy_eps: float = greedy_eps   # 与均匀分布混合的权重 ε
        self.temp_tau: float = temp_tau       # logits 温度 τ (>1 拉平)

        # 可选：设置最小概率地板（不需要就留 0）
        self.prob_floor: float = 0.01

    def _probs(self, state: TEN) -> TEN:
        """softmax(logits/τ)，再与均匀分布做 ε 混合，并可选设地板"""
        logits = self.net(self.state_norm(state))  # [B, K]
        tau = max(float(self.temp_tau), 1e-6)
        a_prob = th.softmax(th.clamp(logits, -5.0, 5.0) / tau, dim=-1)  # 温度缩放

        if self.greedy_eps > 0.0:
            K = a_prob.size(-1)
            a_prob = (1.0 - self.greedy_eps) * a_prob + self.greedy_eps * (1.0 / K)

        if self.prob_floor > 0.0:
            a_prob = th.clamp(a_prob, self.prob_floor, 1.0)
            a_prob = a_prob / a_prob.sum(dim=-1, keepdim=True)
        if random.random() < 0.00001:
            print(logits,"\n",a_prob)
        return a_prob

    def forward(self, state: TEN) -> TEN:
        """推理：返回离散动作索引（argmax）。env 侧 idx→rate。"""
        a_prob = self._probs(state)
        return a_prob.argmax(dim=-1)

    def get_action(self, state: TEN) -> (TEN, TEN):
        """训练采样：返回索引 + logprob"""
        a_prob = self._probs(state)

        dist = self.ActionDist(probs=a_prob)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> (TEN, TEN):
        a_prob = self._probs(state)
        dist = self.ActionDist(probs=a_prob)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.long()

class ActorBernoulliBetaPPO(nn.Module):
    """
    两阶段动作：gate ~ Bernoulli(p)，magnitude ~ Beta(alpha, beta)
    - buffer存动作: [g, m]（2 维）
    - env动作: rate \in [0, r_max]，由 convert_action_for_env([g, m]) 给出
    - 概率：logpi = log Bernoulli(g|p) + g * log Beta(m|alpha,beta)
      熵：H ≈ H_bern(p) + p * H_beta(alpha,beta)
    """
    def __init__(self,
                 net_dims: list[int],
                 state_dim: int,
                 action_dim: int,            # 必须为 2（[g,m]）
                 *,
                 use_tcn: bool = False,      # 如需用你的 TCN，就设 True 并传 K
                 K: int = 1,
                 hidden_dim: int = 32,       # trunk输出维度
                 r_max: float = 0.08,
                 gamma_shape: float = 1.0,   # 形状偏好 r = r_max * g * m^gamma
                 # 门控稳定化
                 gate_tau: float = 1.5,      # 温度缩放（>1 更平）
                 gate_eps: float = 0.10,     # 与0.5混合，防早塌
                 # Beta 数值稳定
                 min_conc: float = 0.20,     # alpha/beta 浓度下界
                 max_conc: float = 30.0):    # 上界避免过尖
        super().__init__()
        assert action_dim == 2, "ActorBernoulliBetaPPO 需要 action_dim=2（[g,m]）"

        # —— 共享干（trunk）：MLP 或 TCN —— #
        if use_tcn:
            self.backbone = build_tcn(
                state_dim=state_dim, action_dim=hidden_dim, K=K,
                net_dims=net_dims, emb_ch=64, num_blocks=2,
                kernel_size=3, dilations=[1, 2],
                dropout=0.05, activation=nn.SiLU, for_q=False
            )
        else:
            self.backbone = build_mlp([state_dim, *net_dims, hidden_dim],
                                      activation=nn.SiLU, if_raw_out=True)

        # —— 三个头：p（伯努利），alpha/beta（Beta 浓度） —— #
        self.head_gate = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 2)
        layer_init_with_orthogonal(self.head_gate, std=0.1)
        layer_init_with_orthogonal(self.head_beta, std=0.1, bias_const=1.0)

        # 参数
        self.r_max = float(r_max)
        self.gamma_shape = float(gamma_shape)
        self.gate_tau = float(gate_tau)
        self.gate_eps = float(gate_eps)
        self.min_conc = float(min_conc)
        self.max_conc = float(max_conc)

        # 与你现有接口一致（占位，不启用也行）
        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std  = nn.Parameter(th.ones((state_dim,)),  requires_grad=False)

    # 可按需启用/替换
    def state_norm(self, s: TEN) -> TEN:
        return s

    # 生成 p, alpha, beta
    def _heads(self, state: TEN):
        h = self.backbone(self.state_norm(state))  # [B,H]
        # gate: 温度 + ε 混合，防早塌
        logit = self.head_gate(h) / max(self.gate_tau, 1e-6)      # [B,1]
        p = th.sigmoid(logit)# (0,1)
        p = th.clamp(p, 1, 1)  # 避免 log(0) 错误
        if self.gate_eps > 0:
            p = (1 - self.gate_eps) * p + self.gate_eps * 0.5

        # beta: softplus 保正，再裁剪范围
        ab_raw = self.head_beta(h)                                # [B,2]
        conc = F.softplus(ab_raw) + 1e-4
        conc = th.clamp(conc, self.min_conc, self.max_conc)
        alpha, beta = conc[..., 0:1], conc[..., 1:2]              # 各 [B,1]
        return p, alpha, beta

    # 评估/推理：返回“期望速率”（标量）
    def forward(self, state: TEN) -> TEN:
        p, alpha, beta = self._heads(state)
        # E[m^gamma] = B(alpha+gamma, beta) / B(alpha, beta)
        if self.gamma_shape == 1.0:
            m_moment = alpha / (alpha + beta)
        else:
            # 用 lgamma 计算 Beta 函数的比值，数值更稳
            lg = th.lgamma(alpha + self.gamma_shape) + th.lgamma(beta) \
                 - th.lgamma(alpha + beta + self.gamma_shape) \
                 - (th.lgamma(alpha) + th.lgamma(beta) - th.lgamma(alpha + beta))
            m_moment = th.exp(lg)
        rate = self.r_max * (p * m_moment)                       # [B,1]
        return rate.squeeze(-1)

    # 训练采样：返回潜在动作 [g,m] 以及 logprob（PPO 用）
    def get_action(self, state: TEN) -> tuple[TEN, TEN]:
        p, alpha, beta = self._heads(state)
        if random.random() < 0.0001:  #! debug
            print(p,alpha,beta)
        bern = th.distributions.Bernoulli(probs=p)
        beta_dist = th.distributions.Beta(alpha, beta)

        g = bern.sample()                               # [B,1], 0/1
        m = beta_dist.sample()                          # [B,1], (0,1)

        logprob = bern.log_prob(g).squeeze(-1) + (g.squeeze(-1) * beta_dist.log_prob(m.squeeze(-1)))
        #entropy  = bern.entropy().squeeze(-1) + (p.squeeze(-1) * beta_dist.entropy())

        action_latent = th.cat([g, m], dim=-1)         # [B,2] → buffer
        return action_latent, logprob

    # PPO 重算 logprob / 熵
    def get_logprob_entropy(self, state: TEN, action: TEN) -> tuple[TEN, TEN]:
        g = action[..., 0:1]
        m = action[..., 1:2].clamp(1e-6, 1 - 1e-6)

        p, alpha, beta = self._heads(state)
        bern = th.distributions.Bernoulli(probs=p)
        beta_dist = th.distributions.Beta(alpha, beta)

        logprob = bern.log_prob(g).squeeze(-1) + (g.squeeze(-1) * beta_dist.log_prob(m.squeeze(-1)))
        entropy  = bern.entropy().squeeze(-1) + (p.squeeze(-1) * beta_dist.entropy())
        return logprob, entropy

    # 映射到环境动作：把 [g,m] → 实际速率（标量）
    def convert_action_for_env(self, action: TEN) -> TEN:
        """
        输入: action [B,2] = [g, m]
        输出: 速率 [B,1]，范围 [0, r_max]
        说明:
          - 训练时 g 是 0/1；评估时若传入期望，也可把 g 当连续门（必要时可硬阈值）
        """
        g = action[..., 0:1]
        m = action[..., 1:2].clamp(0.0, 1.0)
        rate = self.r_max * (g * (m ** self.gamma_shape))
        return rate  # [B,1]


class CriticPPO(th.nn.Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int, K:int):
        super().__init__()
        assert isinstance(action_dim, int)
        self.state_single_len: int = state_dim // K
        self.net = build_mlp(dims=[self.state_single_len, *net_dims, 1])
        """self.net = build_tcn(state_dim=state_dim,
                     action_dim=1,
                     K=K,
                     net_dims=net_dims,
                     emb_ch=64, num_blocks=3, kernel_size=3, dilations=[1,2,4],
                     dropout=0.00, activation=nn.SiLU, for_q=False)"""
        layer_init_with_orthogonal(self.net[-1], std=0.5)

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        state = state[..., -self.state_single_len:]
        value = self.net(state)
        return value  # advantage value

    def state_norm(self, state: TEN) -> TEN:
        return state ###!QQQQQQQQQQQQQQQQQQQQQQQQQq
