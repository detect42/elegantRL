import math
import random
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch as th
from torch import nn

from ..train import Config, ReplayBuffer
from .AgentBase import ActorBase, AgentBase, CriticBase, build_mlp, build_tcn, layer_init_with_orthogonal

TEN = th.Tensor

try:
    from .AgentModSACDiscrete import ActorDiscreteSAC as _ActorDiscreteSAC  # 实际类所在处
    ActorDiscreteSAC = _ActorDiscreteSAC  # 导出一个同名符号，以兼容旧权重
except Exception as e:
    pass  # 若没有该文件，这里忽略；但加载旧模型会失败

class AgentSAC(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.num_ensembles = getattr(args, 'num_ensembles', 4)  # the number of critic networks
        use_tcn = getattr(args, 'use_tcn', False)
        K = getattr(args, 'K', 1)

        self.act = ActorSAC(net_dims, state_dim, action_dim, use_tcn=use_tcn, K=K).to(self.device)
        self.cri = CriticEnsemble(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles, use_tcn=use_tcn, K=K).to(self.device)
        # self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.alpha_log = th.tensor((-1,), dtype=th.float32, requires_grad=True, device=self.device)  # trainable var
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        #self.target_entropy = np.log(action_dim) #! 原来的
        self.target_entropy = -action_dim
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state)

    def _explore_one_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0))[0]

    def _explore_vec_action(self, state: TEN) -> TEN:
        return self.act.get_action(state)

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> Tuple[float, float]:
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            next_action, next_logprob = self.act.get_action_logprob(next_state)  # stochastic policy
            next_q = th.min(self.cri_target.get_q_values(next_state, next_action), dim=1)[0]
            alpha = self.alpha_log.exp()
            q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

        '''objective of critic (loss function of critic)'''
        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.view((-1, 1)).repeat(1, q_values.shape[1])
        td_error = self.criterion(q_values, q_labels).mean(dim=1) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        if self.lambda_fit_cum_r:
            cum_reward_mean = buffer.cum_rewards[buffer.ids0, buffer.ids1].detach_().mean().repeat(q_values.shape[1])
            obj_critic += self.criterion(cum_reward_mean, q_values.mean(dim=0)).mean() * self.lambda_fit_cum_r
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        '''objective of alpha (temperature parameter automatic adjustment)'''
        action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
        obj_alpha = (self.alpha_log * (self.target_entropy - logprob).detach()).mean()
        self.optimizer_backward(self.alpha_optim, obj_alpha)

        '''objective of actor'''
        alpha = self.alpha_log.exp().detach()
        with th.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-5, 1)

        q_value_pg = self.cri_target(state, action_pg).mean()
        obj_actor = (q_value_pg - logprob * alpha).mean()
        self.optimizer_backward(self.act_optimizer, -obj_actor)
        # self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), obj_actor.item()


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and Two Time-scale Update Rule
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.num_ensembles = getattr(args, 'num_ensembles', 8)  # the number of critic networks
        use_tcn = getattr(args, 'use_tcn', False)
        K = getattr(args, 'K', 1)

        self.act = ActorFixSAC(net_dims, state_dim, action_dim, use_tcn=use_tcn, K=K, emb_ch=64).to(self.device)
        self.cri = CriticEnsemble(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles, use_tcn=use_tcn, K=K, emb_ch=16).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.alpha_log = th.tensor((-1,), dtype=th.float32, requires_grad=True, device=self.device)  # trainable var
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        #self.target_entropy = getattr(args, 'target_entropy', -np.log(action_dim)) #! 原来的
        self.target_entropy = getattr(args, 'target_entropy', -action_dim)

        # for reliable_lambda
        self.critic_tau = getattr(args, 'critic_tau', 0.995)
        self.critic_value = 1.0  # for reliable_lambda
        self.update_a = 0  # the counter of update actor
        self.criterion = nn.SmoothL1Loss(reduction='none')  #! 对于大reward情况，不要让对loss冲击太大

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> Tuple[float, float]:
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            #next_action, next_logprob = self.act.get_action_logprob(next_state)  # stochastic policy #!  这是原来的
            next_action, next_logprob = self.act_target.get_action_logprob(next_state)
            next_q = th.min(self.cri_target.get_q_values(next_state, next_action), dim=1)[0]
            alpha = self.alpha_log.exp()
            q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

        '''objective of critic (loss function of critic)'''
        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.view((-1, 1)).repeat(1, q_values.shape[1])
        td_error = self.criterion(q_values, q_labels).mean(dim=1) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        if self.lambda_fit_cum_r != 0:
            cum_reward_mean = buffer.cum_rewards[buffer.ids0, buffer.ids1].detach_().mean().repeat(q_values.shape[1])
            obj_critic += self.criterion(cum_reward_mean, q_values.mean(dim=0)).mean() * self.lambda_fit_cum_r
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        '''objective of alpha (temperature parameter automatic adjustment)'''
        action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
        obj_alpha = (self.alpha_log * (self.target_entropy - logprob).detach()).mean()
        self.optimizer_backward(self.alpha_optim, obj_alpha)

        '''objective of actor'''
        alpha = self.alpha_log.exp().detach()
        with th.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-5, 1)

        #! 自己添加的，自适应根据critic的td误差调整actor跟新间隙
        with th.no_grad():
            td_rms = td_error.mean().sqrt().item()  # RMSE，数值随你的奖励/回报标度而定
            if random.random() < 0.01:
                print(f"td_rms={td_rms:.4f}, critic_value={self.critic_value:.4f}, update_a={self.update_a}, update_t={update_t}")
                print(q_values[:5],q_labels[:5])
            self.critic_value = self.critic_tau * self.critic_value + (1 - self.critic_tau) * td_rms
        # for reliable_lambda
        reliable_lambda = math.exp(-self.critic_value ** 2)
        self.update_a = 0 if update_t == 0 else self.update_a  # reset update_a to 0 when update_t is 0
        if (self.update_a / (update_t + 1)) < (1 / (2 - reliable_lambda)):  # auto Two-time update rule
            self.update_a += 1

            q_value_pg = self.cri_target(state, action_pg).mean()
            obj_actor = (q_value_pg - logprob * alpha).mean()
            self.optimizer_backward(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        else:
            obj_actor = th.tensor(th.nan)
        return obj_critic.item(), obj_actor.item()


'''network'''


class ActorSAC(ActorBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, *, use_tcn: bool = False, K: int = 1):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        if use_tcn and K != 1:
            # TCN encoder that outputs the latent dim net_dims[-1]
            self.net_s = build_tcn(state_dim=state_dim, action_dim=net_dims[-1], K=K,
                                   net_dims=net_dims, emb_ch=64, num_blocks=2, kernel_size=3,
                                   dilations=[1, 2], dropout=0.00, activation=nn.SiLU, for_q=False)
        else:
            self.net_s = build_mlp(dims=[state_dim, *net_dims], if_raw_out=False)  # MLP encoder
        self.net_a = build_mlp(dims=[net_dims[-1], action_dim * 2])  # the average and log_std of action
        layer_init_with_orthogonal(self.net_a[-1], std=0.1)

    def forward(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg = self.net_a(s_enc)[:, :self.action_dim]
        return a_avg.tanh()  # action

    def get_action(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-5, 1).exp()

        dist = self.ActionDist(a_avg, a_std)
        return dist.rsample().tanh()  # action (re-parameterize)

    def get_action_logprob(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-5, 1).exp()

        dist = self.ActionDist(a_avg, a_std)
        action = dist.rsample()

        action_tanh = action.tanh()
        #logprob = dist.log_prob(a_avg) #! 原版写错了哦
        logprob = dist.log_prob(action)
        logprob -= (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return action_tanh, logprob.sum(1)


class ActorFixSAC(ActorBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, *, use_tcn: bool = False, K: int = 1, emb_ch= 64):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        if use_tcn and K != 1:

            self.encoder_s = build_tcn(state_dim=state_dim, action_dim=net_dims[-1], K=K,
                                       net_dims=net_dims[:-1], emb_ch=emb_ch, num_blocks=2, kernel_size=3,
                                       dilations=[1, 2], dropout=0.00, activation=nn.SiLU, for_q=False,return_feature_extractor=False)
        else:
            self.encoder_s = build_mlp(dims=[state_dim, *net_dims])  # encoder of state
        self.decoder_a_avg = build_mlp(dims=[net_dims[-1], 32, action_dim])  # decoder of action mean
        self.decoder_a_std = build_mlp(dims=[net_dims[-1], 32, action_dim])  # decoder of action log_std
        self.soft_plus = nn.Softplus()

        layer_init_with_orthogonal(self.decoder_a_avg[-1], std=0.5)
        layer_init_with_orthogonal(self.decoder_a_std[-1], std=0.5)

    def forward(self, state: TEN) -> TEN:
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        return self.decoder_a_avg(state_tmp).tanh()  # action

    def get_action(self, state: TEN, **_kwargs) -> TEN:  # for exploration
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        action_avg = self.decoder_a_avg(state_tmp)
        action_std = self.decoder_a_std(state_tmp).clamp(-20, 2).exp()

        noise = th.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        return action.tanh()  # action (re-parameterize)

    def get_action_logprob(self, state: TEN) -> Tuple[TEN, TEN]:
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        action_log_std = self.decoder_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.decoder_a_avg(state_tmp)

        noise = th.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        logprob = -action_log_std - noise.pow(2) * 0.5 - np.log(np.sqrt(2 * np.pi))
        # dist = self.Normal(action_avg, action_std)
        # action = dist.sample()
        # logprob = dist.log_prob(action)

        '''fix logprob by adding the derivative of y=tanh(x)'''
        logprob -= (np.log(2.) - action - self.soft_plus(-2. * action)) * 2.  # better than below
        # logprob -= (1.000001 - action.tanh().pow(2)).log()
        return action.tanh(), logprob.sum(1)


class CriticEnsemble(CriticBase):
    def __init__(
        self,
        net_dims: List[int],
        state_dim: int,
        action_dim: int,
        num_ensembles: int = 4,
        *,
        use_tcn: bool = False,
        K: int = 1,
        emb_ch: int = 16,   #! 使用tcn的话 64对比action的一个维度太大了？？？
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.K = K
        self.use_tcn = bool(use_tcn and K != 1)
        self.num_ensembles = num_ensembles

        if not self.use_tcn:
            # —— MLP 路径（保持不变）——
            self.encoder_sa = build_mlp(dims=[state_dim // K + action_dim, net_dims[0]])
            self.decoder_qs = []
            for net_i in range(num_ensembles):
                decoder_q = build_mlp(dims=[*net_dims, 1])
                layer_init_with_orthogonal(decoder_q[-1], std=0.5)
                self.decoder_qs.append(decoder_q)
                setattr(self, f"decoder_q{net_i:02}", decoder_q)
        else:
            # —— TCN 路径：共享特征提取器 + 多头解码（与 MLP 结构对齐）——
            self.emb_ch = emb_ch
            self.add_action = True  # 给 TCN 编码器拼接 action
            # 共享的 TCN 特征抽取器（只返回特征）
            self.feat_extractor = build_tcn(
                state_dim=state_dim,
                action_dim=action_dim,
                K=K,
                net_dims=net_dims,       # 这里只用于保持接口一致；真正的解码头在下面建
                emb_ch=self.emb_ch,
                num_blocks=2,
                kernel_size=3,
                dilations=[1, 2],
                dropout=0.00,
                activation=nn.SiLU,
                for_q=False,             # 这里不直接出 Q
                return_feature_extractor=True,  # ★ 只要特征
            )
            # 每个 Q 头：从共享特征维度 -> net_dims -> 1
            in_head = self.emb_ch + (action_dim if self.add_action else 0)
            self.decoder_qs = []
            for net_i in range(num_ensembles):
                decoder_q = build_mlp(dims=[in_head, 128, 32, 1])
                layer_init_with_orthogonal(decoder_q[-1], std=0.5)
                self.decoder_qs.append(decoder_q)
                setattr(self, f"decoder_q{net_i:02}", decoder_q)

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        if hasattr(self, 'encoder_sa'):
            # —— MLP 路径 ——（保持不变）
            state_true_dim = state.shape[-1] // self.K
            state = state[..., -state_true_dim:]
            tensor_sa = self.encoder_sa(th.cat((state, action), dim=1))
            values = th.concat([decoder_q(tensor_sa) for decoder_q in self.decoder_qs], dim=-1)
            return values
        else:
            # —— TCN 路径：共享编码 + 多头解码 ——
            s_feat = self.feat_extractor(state)               # [B, emb_ch]
            #print(s_feat,"\n\n","action:",action)
            sa_feat = th.cat([s_feat, action], dim=1)         # [B, emb_ch + A]
            values = th.concat([decoder_q(sa_feat) for decoder_q in self.decoder_qs], dim=-1)  # [B, N]
            return values