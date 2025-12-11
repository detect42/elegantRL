import math
import random
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch as th
from torch import nn
from typing import List, Optional
from ..train import ReplayBuffer
from .AgentBase import ActorBase, AgentBase, CriticBase, build_mlp, build_tcn, layer_init_with_orthogonal
from omegaconf import DictConfig, OmegaConf

TEN = th.Tensor


class AgentModSACDiscrete(AgentBase):
    """
    离散版 ModSAC：
      - Critic 输出 Q(s,·)，多头取 min 做 TD/actor 目标
      - Actor 用期望目标（不用 reparam）
      - α 自动调节（采样式）
      - reliable_lambda + Two-time Update Rule
    """

    def __init__(
        self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Optional[DictConfig] = None
    ):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        use_tcn = args.model.use_tcn
        K = args.env.K
        self.num_ensembles = args.model.num_ensembles
        self.critic_dropout = args.model.critic_dropout
        self.L2_reg = args.train.L2_reg
        self.state_noise_std = args.model.state_noise_std
        self.emd_ch = args.model.emb_ch

        # 离散 Actor / Critic（注意：discrete critic 类）
        temp_tau = args.model.temp_tau
        greedy_eps = args.model.greedy_eps

        self.act = ActorDiscreteSAC(
            net_dims,
            state_dim,
            action_dim,
            use_tcn=use_tcn,
            K=K,
            temp_tau=temp_tau,
            greedy_eps=greedy_eps,
            emb_ch=self.emd_ch,
        ).to(self.device)

        self.cri = CriticEnsembleDiscrete(
            net_dims,
            state_dim,
            action_dim,
            num_ensembles=self.num_ensembles,
            use_tcn=use_tcn,
            K=K,
            emb_ch=self.emd_ch,
            dropout=self.critic_dropout,
            state_noise_std=self.state_noise_std,
        ).to(self.device)

        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.act_optimizer = th.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), lr=self.learning_rate, weight_decay=self.L2_reg)

        # α & 目标熵：离散建议 -log(K_actions)
        default_target_H = -float(math.log(max(action_dim, 1)))
        self.alpha_log = th.tensor((-1.0,), dtype=th.float32, requires_grad=True, device=self.device)
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        self.target_entropy = args.get("target_entropy", default_target_H)

        # critic 稳定：Huber
        self.criterion = nn.SmoothL1Loss(reduction="none")

        # reliable_lambda
        self.critic_tau = args.get("critic_tau", 0.995)  # critic EMA
        self.critic_value = 1.0
        self.update_a = 0

    def explore_action(self, state: TEN) -> TEN:
        # 环境交互：返回离散索引（Long）
        return self.act.get_action(state)

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> Tuple[float, float]:
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state, is_weight, is_index) = buffer.sample_for_per(
                    self.batch_size
                )
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            # 确保离散索引形状与类型
            if action.dim() > 1:
                action_idx = action.squeeze(-1).long()
            else:
                action_idx = action.long()

            # ---- 目标：V'(s') 的期望 ----
            alpha = self.alpha_log.exp()
            probs_next, log_probs_next = self.act_target.policy(next_state)  # [B,K]
            Q_all_next = self.cri_target.get_q_all(next_state)  # [B,N,K]
            Qmin_next = Q_all_next.min(dim=1)[0]  # [B,K]
            V_next = (probs_next * (Qmin_next - alpha * log_probs_next)).sum(dim=1)  # [B]
            q_label = reward + undone * self.gamma * V_next  # [B]

        # ---- critic：Huber over heads ----
        Q_heads = self.cri.get_q_values(state, action_idx)  # [B,N]
        q_labels = q_label.view((-1, 1)).repeat(1, Q_heads.shape[1])
        td_error = self.criterion(Q_heads, q_labels).mean(dim=1) * unmask  # [B]
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()

        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        # ---- alpha：采样式（与你现有实现一致）----
        _, logprob_sampled = self.act.get_action_logprob(state)  # [B]
        obj_alpha = (self.alpha_log * (self.target_entropy - logprob_sampled).detach()).mean()
        self.optimizer_backward(self.alpha_optim, obj_alpha)

        # ---- actor：期望目标 + reliable_lambda TTUR ----
        with th.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-5, 1)
            # critic EMA（越大表示越不稳）
            td_rms = td_error.mean().sqrt().item()
            self.critic_value = self.critic_tau * self.critic_value + (1.0 - self.critic_tau) * td_rms

        reliable_lambda = math.exp(-self.critic_value**2)
        self.update_a = 0 if update_t == 0 else self.update_a
        ratio = (self.update_a / (update_t + 1)) if (update_t + 1) > 0 else 0.0

        if ratio < (1.0 / (2.0 - reliable_lambda)):
            self.update_a += 1
            # 期望：sum_a π(a|s) [ α logπ(a|s) - Qmin(s,a) ]
            alpha_detach = self.alpha_log.exp().detach()
            probs, log_probs = self.act.policy(state)  # [B,K]
            Qmin = self.cri_target.get_q_all(state).min(dim=1)[0]  # [B,K]
            actor_loss = (probs * (alpha_detach * log_probs - Qmin)).sum(dim=1).mean()
            self.optimizer_backward(self.act_optimizer, actor_loss)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
            obj_actor = -actor_loss.detach().item()  # 为了与连续版返回“越大越好”的一致性
        else:
            obj_actor = float("nan")

        return obj_critic.item(), obj_actor


class ActorDiscreteSAC(ActorBase):
    """
    离散策略头：
      - 若 use_tcn 且 K!=1：用 TCN 特征抽取器（共享），再接 MLP 输出 logits[K]
      - 否则：MLP 直接输出 logits[K]
    """

    def __init__(
        self,
        net_dims: List[int],
        state_dim: int,
        action_dim: int,
        *,
        use_tcn: bool = False,
        K: int = 1,
        temp_tau: float = 1.0,
        greedy_eps: float = 0.0,
        emb_ch=96
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.K = K
        self.use_tcn = bool(use_tcn and K != 1)
        self.temp_tau = float(temp_tau)
        self.greedy_eps = float(greedy_eps)

        if self.use_tcn:
            # 只要特征：return_feature_extractor=True
            self.feat_extractor = build_tcn(
                state_dim=state_dim,
                action_dim=0,
                K=K,
                net_dims=net_dims,
                emb_ch=emb_ch,
                num_blocks=3,
                kernel_size=3,
                dilations=[1, 2, 3],
                dropout=0.00,
                activation=nn.SiLU,
                for_q=False,
                return_feature_extractor=True,
            )
            in_dim = emb_ch
            self.policy_head = build_mlp([in_dim, *net_dims, 32, action_dim])
        else:
            # 纯 MLP
            self.backbone = build_mlp([state_dim, *net_dims])
            self.policy_head = build_mlp([net_dims[-1], action_dim])

        layer_init_with_orthogonal(self.policy_head[-1], std=1)

    def _logits(self, state: TEN) -> TEN:
        if self.use_tcn:
            feat = self.feat_extractor(state)  # [B, C]
            logits = self.policy_head(feat)  # [B, K]
        else:
            h = self.backbone(state)
            logits = self.policy_head(h)
        return logits

    def policy(self, state: TEN) -> Tuple[TEN, TEN]:
        logits = self._logits(state)  # [B,K]
        if random.random() < 0.00001:
            print("logits:", logits)

        tau = max(self.temp_tau, 1e-6)
        probs = th.softmax(th.clamp(logits, -20.0, 20.0) / tau, dim=-1)
        if self.greedy_eps > 0.0:
            K = probs.size(-1)
            probs = (1.0 - self.greedy_eps) * probs + self.greedy_eps * (1.0 / K)
        log_probs = th.log(probs.clamp_min(1e-8))
        return probs, log_probs

    def forward(self, state: TEN) -> TEN:
        probs, _ = self.policy(state)
        return probs.argmax(dim=-1)  # greedy idx

    def get_action(self, state: TEN) -> TEN:
        probs, _ = self.policy(state)
        if random.random() < 0.00002:
            print(probs)
        dist = th.distributions.Categorical(probs=probs)
        action_idx = dist.sample()  # [B]
        return action_idx

    def get_action_logprob(self, state: TEN) -> Tuple[TEN, TEN]:
        probs, log_probs = self.policy(state)
        dist = th.distributions.Categorical(probs=probs)
        action_idx = dist.sample()
        logprob = dist.log_prob(action_idx)  # [B]
        return action_idx, logprob

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.long()


class CriticEnsembleDiscrete(nn.Module):
    """
    多头离散 Q(s,·)：
      - MLP：encoder_s(state) -> per-head decoder -> K 维
      - TCN：共享 TCN 特征抽取器 -> per-head decoder -> K 维
    """

    def __init__(
        self,
        net_dims: List[int],
        state_dim: int,
        action_dim: int,
        num_ensembles: int = 4,
        *,
        use_tcn: bool = False,
        K: int = 1,
        emb_ch: int = 64,
        dropout=0.05,
        state_noise_std=0.03
    ):
        super().__init__()
        self.K = K
        self.action_dim = action_dim  # K_actions（离散动作数）
        self.num_ensembles = num_ensembles
        self.use_tcn = bool(use_tcn and K != 1)
        self.state_noise_std = state_noise_std
        if not self.use_tcn:
            # MLP 共享编码（注意：如果你的状态是 K 个时间片拼接，这里只取最后一个片段）
            self.state_single_len = state_dim // K
            self.encoder_s = build_mlp([self.state_single_len, net_dims[0]])
            self.decoders = nn.ModuleList()
            for _ in range(num_ensembles):
                dec = build_mlp([*net_dims, 32, action_dim])  # 末层输出 K_actions
                layer_init_with_orthogonal(dec[-1], std=1)
                self.decoders.append(dec)
        else:
            # 共享 TCN 特征抽取
            self.feat_extractor = build_tcn(
                state_dim=state_dim,
                action_dim=0,
                K=K,
                net_dims=net_dims,
                emb_ch=emb_ch,
                num_blocks=3,
                kernel_size=3,
                dilations=[1, 2, 3],
                dropout=dropout,
                activation=nn.SiLU,
                for_q=False,
                return_feature_extractor=True,
            )
            in_dim = emb_ch
            self.decoders = nn.ModuleList()
            for _ in range(num_ensembles):
                dec = build_mlp([in_dim, *net_dims, 32, action_dim])
                layer_init_with_orthogonal(dec[-1], std=1)
                self.decoders.append(dec)

    def _features(self, state: TEN) -> TEN:
        if not self.use_tcn:
            # 若 state 拼了 K 段，就只取最后一段
            s_true = state[..., -self.state_single_len :]
            feat = self.encoder_s(s_true)
            return feat
        else:
            return self.feat_extractor(state)

    def get_q_all(self, state: TEN) -> TEN:
        """
        返回所有头、所有动作的 Q： [B, N, K_actions]
        """
        feat = self._features(state)
        heads = [dec(feat) for dec in self.decoders]  # list of [B,K_actions]
        Q_all = th.stack(heads, dim=1)  # [B, N, K_actions]
        return Q_all

    def get_q_values(self, state: TEN, action_idx: TEN) -> TEN:
        """
        根据离散动作索引提取每个头对应的 Q(s,a)： [B, N]
        """

        noise = th.randn_like(state).clamp(-2.0, 2.0)
        state_with_noise = state + self.state_noise_std * state * noise
        Q_all = self.get_q_all(state_with_noise)  # [B,N,K]
        a = action_idx.long().view(-1, 1, 1)  # [B,1,1]
        Q_sa = Q_all.gather(dim=-1, index=a.expand(-1, Q_all.size(1), 1)).squeeze(-1)  # [B,N]
        return Q_sa

    def forward(self, state: TEN, action_idx: TEN) -> TEN:
        """
        兼容性接口：返回均值 Q̄(s,a) ∈ [B,1]
        """
        Q_sa = self.get_q_values(state, action_idx)  # [B,N]
        return Q_sa.mean(dim=1, keepdim=True)
