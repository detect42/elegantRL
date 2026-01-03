from __future__ import annotations
import math
import random
from copy import deepcopy
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..train import ReplayBuffer
from .AgentBase import ActorBase, AgentBase, CriticBase, build_mlp, build_tcn, layer_init_with_orthogonal

TEN = th.Tensor


class AgentModSACDiscrete(AgentBase):
    """
    离散版 ModSAC：
      - Critic 输出 Q(s,·)，多头取 min 做 TD/actor 目标
      - Actor 用期望目标（不用 reparam）
      - α 自动调节（采样式）
      - reliable_lambda + Two-time Update Rule
    """

    act: ActorDiscreteSAC  # type: ignore[assignment]
    act_target: ActorDiscreteSAC  # type: ignore[assignment]
    cri: CriticEnsembleDiscrete  # type: ignore[assignment]
    cri_target: CriticEnsembleDiscrete  # type: ignore[assignment]

    def __init__(self, state_dim: int, action_dim: int, gpu_id: int, args: DictConfig):
        super().__init__(state_dim, action_dim, gpu_id, args)
        # ====================================================
        # 1. 初始化 Actor (直接传 args.agent.actor)
        # ====================================================
        self.act = ActorDiscreteSAC(
            state_dim=state_dim, action_dim=action_dim, actor_cfg=args.agent.actor  # <--- ★ 核心：把整块配置传进去
        ).to(self.device)

        # ====================================================
        # 2. 初始化 Critic (直接传 args.agent.critic)
        # ====================================================
        self.cri = CriticEnsembleDiscrete(
            state_dim=state_dim, action_dim=action_dim, critic_cfg=args.agent.critic  # <--- ★ 核心：把整块配置传进去
        ).to(self.device)

        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.act_optimizer = th.optim.Adam(self.act.parameters(), lr=args.agent.actor_learning_rate)
        self.cri_optimizer = th.optim.Adam(
            self.cri.parameters(), lr=args.agent.critic_learning_rate, weight_decay=args.train.L2_reg
        )

        # α & 目标熵：离散建议 -log(K_actions)
        default_target_H = -float(math.log(max(action_dim, 1)))
        self.alpha_log = th.tensor((-1.0,), dtype=th.float32, requires_grad=True, device=self.device)
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=args.agent.alpha_learning_rate)
        self.target_entropy = args.agent.target_entropy if args.agent.target_entropy is not None else default_target_H

        # reliable_lambda
        self.critic_tau = args.agent.critic_tau  # critic EMA
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
                # is_weight, is_index = None, None

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
        Q_heads = self.cri.get_q_values(state, action_idx)  # [B,N], N is num_ensembles

        q_labels = q_label.view((-1, 1)).repeat(1, Q_heads.shape[1])
        if random.random() < 0.001:
            print("Q_heads: ", Q_heads[:5].detach().cpu().numpy().round(2))
            print("q_labels: ", q_labels[:5].detach().cpu().numpy().round(2))
            print("reward: ", reward[:5].detach().cpu().numpy().round(2))
            print("action_idx: ", action_idx[:5].detach().cpu().numpy().round(2))

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

    def __init__(self, state_dim: int, action_dim: int, actor_cfg: DictConfig):
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        # 1. 解析通用参数
        self.K = actor_cfg.K
        self.temp_tau = actor_cfg.temp_tau
        self.greedy_eps = actor_cfg.greedy_eps

        # 2. 决定网络类型
        self.model_type = actor_cfg.type.lower()  # "tcn" or "mlp"

        if self.model_type == "tcn":
            tcn_params = cast(dict[str, Any], OmegaConf.to_container(actor_cfg.tcn_args, resolve=True))
            # 只要特征：return_feature_extractor=True
            self.feat_extractor = build_tcn(
                state_dim=state_dim,
                action_dim=action_dim,
                K=self.K,
                for_q=False,
                return_feature_extractor=True,
                activation=nn.SiLU,
                **tcn_params,
            )
            in_dim = tcn_params["emb_ch"]
            self.policy_head = build_mlp([in_dim, *tcn_params["net_dims"], 32, action_dim])
        elif self.model_type == "mlp":
            # 纯 MLP
            net_dim = actor_cfg.mlp_args.net_dims
            self.backbone = build_mlp([state_dim, *net_dim])
            self.policy_head = build_mlp([net_dim[-1], action_dim])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        layer_init_with_orthogonal(self.policy_head[-1], std=1)

    def _logits(self, state: TEN) -> TEN:
        if self.model_type == "tcn":
            feat = self.feat_extractor(state)  # [B, C]
            logits = self.policy_head(feat)  # [B, K]
        elif self.model_type == "mlp":
            h = self.backbone(state)
            logits = self.policy_head(h)
        else:  # should not reach here
            raise ValueError(f"Unknown model type: {self.model_type}")
        return logits

    def policy(self, state: TEN) -> Tuple[TEN, TEN]:
        logits = self._logits(state)  # [B,K]

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

        dist = th.distributions.Categorical(probs=probs)
        action_idx = dist.sample()  # [B]

        if random.random() < 0.000001:
            print(
                "logits:",
                self._logits(state).detach().cpu().numpy(),
                " probs:",
                probs.detach().cpu().numpy(),
                " action_idx:",
                action_idx.item(),
            )  # p
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

    def __init__(self, state_dim: int, action_dim: int, critic_cfg: DictConfig):
        super().__init__()
        self.K = critic_cfg.K
        self.action_dim = action_dim  # K_actions（离散动作数）
        self.num_ensembles = critic_cfg.num_ensembles
        self.state_noise_std = critic_cfg.state_noise_std
        # 2. 决定网络类型
        self.model_type = critic_cfg.type.lower()

        if self.model_type == "mlp":
            # MLP 共享编码（注意：如果你的状态是 K 个时间片拼接，这里只取最后一个片段）
            net_dims = critic_cfg.mlp_args.net_dims
            self.state_single_len = state_dim // self.K

            self.encoder_s = build_mlp([self.state_single_len, net_dims[0]])
            self.decoders = nn.ModuleList()
            for _ in range(self.num_ensembles):
                dec = build_mlp([*net_dims, 32, action_dim])  # 末层输出 K_actions
                layer_init_with_orthogonal(dec[-1], std=1)
                self.decoders.append(dec)
        if self.model_type == "tcn":
            # 共享 TCN 特征抽取
            tcn_params = cast(dict[str, Any], OmegaConf.to_container(critic_cfg.tcn_args, resolve=True))
            self.feat_extractor = build_tcn(
                state_dim=state_dim,
                action_dim=0,
                K=self.K,
                activation=nn.SiLU,
                for_q=False,
                return_feature_extractor=True,
                **tcn_params,
            )
            in_dim = tcn_params["emb_ch"]
            net_dims = tcn_params["net_dims"]
            self.decoders = nn.ModuleList()
            for _ in range(self.num_ensembles):
                dec = build_mlp([in_dim, *net_dims, 32, action_dim])
                layer_init_with_orthogonal(dec[-1], std=1)
                self.decoders.append(dec)

    def _features(self, state: TEN) -> TEN:
        if self.model_type == "mlp":
            # 若 state 拼了 K 段，就只取最后一段
            s_true = state[..., -self.state_single_len :]
            feat = self.encoder_s(s_true)
            return feat
        elif self.model_type == "tcn":
            return self.feat_extractor(state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

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
