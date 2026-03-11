import json
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import numpy.random as rd
import pandas as pd
import torch as th
from omegaconf import DictConfig, OmegaConf

from .auxiliary_data import error_idx

ARY = np.ndarray
np.random.seed(527)
# random.seed(527)


class FutureExecEnv:

    def __init__(
        self,
        cfg: DictConfig,
    ):
        # 定义严格的特征顺序（总共 31 个特征）
        self.STATE_FEATURES = [
            # Market
            "vwap_5s_over_60s",
            "vwap_30s_over_300s",
            "vwap_rank_12x5s",
            # "vwap_rank_12x30s",      # 删减：与 12x5s 严重共线，保留高频捕捉
            "cur_price/base_price",
            # "pv_corr_24x5s",         # 删减：相关性特征在非平稳序列中容易漂移
            # "shadow_vwap_dev",       # 删减：被 cur_price/base_price 覆盖
            # "fragility_24x5s_norm",  # 删减：因子解释度不如动量特征
            "KER_24x5s",
            # "Rejection_Bias_12x5s",  # 删减：盘口压力被 Flow_Toxicity 覆盖
            "Vol_Squeeze",
            "Flow_Toxicity_12x5s",
            # Factor
            "signal_5s",
            "signal_1m",
            "signal_1h",
            # Volatility
            # "vol_feat_300s",         # 删减：信息已被 Squeeze/Shock 吸收
            # "pressure_12x5s",        # 删减：属于盘口快照，毒性特征更稳健
            "smart_momentum_60x5s",
            # "Gini_300s",             # 删减：计算开销大且对 RL 信号增益比低
            "vol_shock",
            "log_position",
            # Process
            "pos_ratio",
            "participate_rate",
            "market_ratio",  # 保留：实盘部署的基准进度坐标
            "gap_to_market",
            # Slippage
            "exp_slippage_bp",
            "now_slippage_bp",
            # Event
            # "event_open_rush",       # 删减：按要求去除
            # "event_close_rush",      # 删减：按要求去除
            # "event_vol_breakout",    # 删减：按要求去除
            "event_sig_spike",
        ]
        self.num_envs = cfg.num_envs
        self.K: int = cfg.K
        self.state_dim: int = cfg.state_dim
        self.action_dim: int = cfg.action_dim
        self.if_discrete: bool = cfg.if_discrete
        self.padding0: bool = cfg.padding0
        self.max_step: int = cfg.max_step
        self.eval_num_workers: int = cfg.eval_num_workers
        self.noise_std_ratio: float = cfg.noise_std_ratio
        self.device = th.device("cpu") if cfg.gpu_id == -1 else th.device(f"cuda:{cfg.gpu_id}")
        self.tot_uncompleted = 0
        self.dataset = "1_sample"
        self.position_root = "/nas/srwang/DATA_preprocess/1_sample_feature_7"
        self.position: pd.DataFrame
        self.action_range = cfg.action_range if self.if_discrete else []
        self.rate_upper_bound: float = max(self.action_range) if self.if_discrete else 0.08
        # reset()
        self.samples: dict = {}
        self.absolute_id = 0
        self.id: str = ""
        self.uncompleted = False  # whether the current episode is uncompleted
        # environment information
        self.env_name: str = "FutureExecEnv_v8"
        self.mode: str = "sample"  # 默认 mode

        # self.max_step = 19260817
        self.freq = 5
        self.time_curr: int = 0
        self.realized_position: int = 0
        self.from_position: int = 0
        self.to_position: int = 0
        self.side: int = 0
        self.time_idx: int = 0
        self.contract_multiplier: int = 1
        self.total_time: int = 0

        self.total_cum_volume: float = 0.0
        self.total_cum_turnover: float = 0.0
        self.my_cum_turnover: float = 0.0
        self.my_cum_slippage: float = 0.0
        self.total_turnover: float = 0.0
        self.total_volume: float = 0.0
        self.base_price: float = 0.0
        self.history: list = []

        # 加载 json 数据
        with open(f"/code/srwang/Finrl/sample_pool/1_train_data.json", "r") as f:
            sample_data = json.load(f)
        print("sample data len= ", len(sample_data))
        self.error_idx = error_idx
        self.sample_pool = sample_data[:]
        self.sample_pool_size = len(self.sample_pool)
        with open(f"/code/srwang/Finrl/sample_pool/1_eval_data.json", "r") as f:
            eval_data = json.load(f)
        with open(f"/code/srwang/Finrl/sample_pool/1_test_data.json", "r") as f:
            test_data = json.load(f)
        self.eval_pool = eval_data[:]  #! 调整eval pool case数量大小
        self.test_pool = test_data[:]
        self.eval_pool_size = len(self.eval_pool)
        self.test_pool_size = len(self.test_pool)
        self.begin_time = 0
        self.end_time = 0
        self.cum_reward: float = 0.0

    def sample(self, sequential=False):
        if sequential:
            # 如果是第一次顺序采样，初始化索引
            if not hasattr(self, "_sequential_index"):
                self._sequential_index = 0

            # 顺序采样，可能需要跳过某些样本
            while True:
                sample = self.sample_pool[self._sequential_index]
                self._sequential_index = (self._sequential_index + 1) % len(self.sample_pool)

                id = sample["id"]
                if id in self.error_idx:
                    continue

                return sample
        else:
            # 原有的随机采样逻辑
            while True:
                sample = random.choice(self.sample_pool)
                id = sample["id"]
                if id in self.error_idx:
                    continue

                return sample

    def reset(self, set_id: int = -1, sequential: bool = False, mode: str = "sample") -> Tuple[ARY, dict]:
        if mode not in {"sample", "eval", "test"}:
            raise ValueError(f"Unknown reset mode: {mode}")

        if set_id == -1:
            sample = self.sample(sequential)
        else:
            if mode == "eval":
                sample = self.eval_pool[set_id]
            elif mode == "test":
                sample = self.test_pool[set_id]
            elif mode == "sample":
                sample = self.sample_pool[set_id]
            else:
                raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.samples = sample
        self.id = sample["id"]
        self.absolute_id = int(sample["Absolute_ID"])
        self.begin_time = sample["begin_time"]
        self.end_time = sample["end_time"]

        self.realized_position = 0
        self.from_position = 0
        self.to_position = sample["target_position"]
        self.total_time = self.end_time - self.begin_time

        self.uncompleted = False
        self.cum_reward = 0.0
        assert self.state_dim % self.K == 0, f"state_dim={self.state_dim} must be divisible by K={self.K}"
        self.history.clear()
        assert self.from_position != self.to_position, "from_position should not equal to to_position at reset."
        self.side = np.sign(self.to_position - self.from_position)

        self.position = pd.read_pickle(os.path.join(self.position_root, f"position_{self.id}.pkl"))
        self.position = self.position[
            (self.position["version_ts"] >= self.begin_time) & (self.position["version_ts"] <= self.end_time)
        ].copy()
        self.position.set_index("version_ts", inplace=True)
        self.process()
        self.time_idx = 0
        self.time_curr = self.position.index[self.time_idx]
        self.base_price = self.position.iloc[0]["mid_price"]

        self.total_cum_volume = 0.0
        self.total_cum_turnover = 0.0
        self.my_cum_turnover = 0.0
        self.my_cum_slippage = 0.0
        self.total_turnover = self.base_price * np.abs(
            self.to_position
        )  #! 之前没加abs,导致在negative情况下全是负的total_turnover
        self.total_volume = self.to_position

        new_state = self.get_state()
        self.history.append(new_state)
        if len(self.history) > self.K:
            self.history.pop(0)
        while len(self.history) < self.K:
            if self.padding0:
                self.history.insert(0, np.zeros_like(self.history[0]))
            else:
                self.history.insert(0, self.history[0])

            # self.history.insert(0, np.zeros_like(self.history[0]))
        assert len(self.history) == self.K
        state_stack = np.concatenate(self.history, axis=0)
        return state_stack, {
            "id": self.id,
            "begin_time": self.begin_time,
            "target_position": self.to_position,
            "end_time": self.end_time,
        }

    def process(self):

        valid_mask = (0.8 * self.position["mid_price"] <= self.position["vwap_5s"]) & (
            self.position["vwap_5s"] <= 1.2 * self.position["mid_price"]
        )
        self.position.loc[~valid_mask, "vwap_5s"] = self.position.loc[~valid_mask, "mid_price"]
        self.position["dh"] = np.nan
        self.position["realized_position"] = 0
        self.position["side"] = np.sign(self.to_position - self.from_position)
        self.position["position"] = self.to_position
        self.position["volume"] = self.position["volume_5s"] * 5

    def calc_expected_slippage(
        self, tick2: pd.Series, delta_position: int, base_price: float, side: float, calc_realized_vol: bool = False
    ) -> float:
        """
        Calculate the expected slippage based on the tick2 data and the delta_position.
        This function assumes that tick2 contains the necessary price and volume information.
        """
        if delta_position == 0:
            return 0.0
        sign = np.sign(delta_position)
        abs_delta = abs(delta_position)
        total_volume = 0.0
        total_slippage = 0.0

        for i in range(1, 7):  #! 对应改成7
            volume_col = f"{'ask' if sign == 1 else 'bid'}_volume{i}"
            price_col = f"{'ask' if sign == 1 else 'bid'}_price{i}"
            if i <= 5:  #! 严格来说应该要<=5
                volume = min(tick2[volume_col], abs_delta - total_volume)
                price = tick2[price_col]
            else:
                volume = abs_delta
                price = tick2[f"{'ask' if sign == 1 else 'bid'}_price5"] + sign * 1
            abs_delta -= volume
            if calc_realized_vol:
                self.realized_position += volume * sign
            total_slippage += volume * (price - base_price) * side
            if abs_delta == 0:
                break

        return total_slippage

    @staticmethod
    def trans_dict_norm(
        state_dict_market,
        state_dict_factor,
        state_dict_volatility,
        state_dict_process,
        state_dict_slippage,
        state_dict_event,
        side,
    ):
        # state_dict["realized_position_ratio"] = state_dict["realized_position_ratio"] / 8

        state_dict_market["vwap_5s_over_60s"] = (
            np.sign(state_dict_market["vwap_5s_over_60s"])
            * np.log(1 + np.abs(state_dict_market["vwap_5s_over_60s"]) / 3)
            * side
        )
        state_dict_market["vwap_30s_over_300s"] = (
            np.sign(state_dict_market["vwap_30s_over_300s"])
            * np.log(1 + np.abs(state_dict_market["vwap_30s_over_300s"]) / 3)
            * side
        )
        state_dict_market["vwap_rank_12x5s"] = state_dict_market["vwap_rank_12x5s"] * side
        state_dict_market["vwap_rank_12x30s"] = state_dict_market["vwap_rank_12x30s"] * side
        state_dict_market["cur_price/base_price"] = (
            np.sign(state_dict_market["cur_price/base_price"])
            * np.log(1 + np.abs(state_dict_market["cur_price/base_price"]) / 3)
            * side
        )
        state_dict_market["pv_corr_24x5s"] = state_dict_market["pv_corr_24x5s"] * side
        state_dict_market["shadow_vwap_dev"] = (
            np.sign(state_dict_market["shadow_vwap_dev"])
            * np.log(1 + np.abs(state_dict_market["shadow_vwap_dev"]) / 3)
            * side
        )
        state_dict_market["fragility_24x5s_norm"] = state_dict_market["fragility_24x5s_norm"]
        state_dict_market["KER_24x5s"] = state_dict_market["KER_24x5s"] * side
        state_dict_market["Rejection_Bias_12x5s"] = state_dict_market["Rejection_Bias_12x5s"] * side
        state_dict_market["Vol_Squeeze"] = state_dict_market["Vol_Squeeze"]
        state_dict_market["Flow_Toxicity_12x5s"] = state_dict_market["Flow_Toxicity_12x5s"] * side

        state_dict_factor["signal_5s"] = state_dict_factor["signal_5s"] * side
        state_dict_factor["signal_1m"] = state_dict_factor["signal_1m"] * side
        state_dict_factor["signal_1h"] = state_dict_factor["signal_1h"] * side

        state_dict_volatility["vol_feat_300s"] = state_dict_volatility["vol_feat_300s"]
        state_dict_volatility["pressure_12x5s"] = state_dict_volatility["pressure_12x5s"] * side
        state_dict_volatility["smart_momentum_60x5s"] = state_dict_volatility["smart_momentum_60x5s"] * side
        state_dict_volatility["Gini_300s"] = state_dict_volatility["Gini_300s"]
        state_dict_volatility["vol_shock"] = state_dict_volatility["vol_shock"]
        state_dict_volatility["log_position"] = state_dict_volatility["log_position"]

        state_dict_process["pos_ratio"] = state_dict_process["pos_ratio"]
        state_dict_process["participate_rate"] = state_dict_process["participate_rate"]
        state_dict_process["market_ratio"] = state_dict_process["market_ratio"]
        state_dict_process["gap_to_market"] = state_dict_process["gap_to_market"]

        state_dict_slippage["exp_slippage_bp"] = np.sign(state_dict_slippage["exp_slippage_bp"]) * np.log(
            1 + np.abs(state_dict_slippage["exp_slippage_bp"]) / 3
        )
        state_dict_slippage["now_slippage_bp"] = np.sign(state_dict_slippage["now_slippage_bp"]) * np.log(
            1 + np.abs(state_dict_slippage["now_slippage_bp"]) / 3
        )
        # state_dict.pop("signal_1m", None) #! debug test
        return {
            **state_dict_market,
            **state_dict_factor,
            **state_dict_volatility,
            **state_dict_process,
            **state_dict_slippage,
            **state_dict_event,
        }

    def get_state(self) -> ARY:
        if self.position is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        pos = self.position.iloc[self.time_idx]
        consumed_time = self.time_curr - self.begin_time
        remain_time = self.end_time - self.time_curr
        state_dict_market: dict[str, float] = {}
        state_dict_factor: dict[str, float] = {}
        state_dict_volatility: dict[str, float] = {}
        state_dict_process: dict[str, float] = {}
        state_dict_slippage: dict[str, float] = {}
        state_dict_event: dict[str, float] = {}

        state_dict_market["vwap_5s_over_60s"] = pos["vwap_5s_over_60s"].item()
        state_dict_market["vwap_30s_over_300s"] = pos["vwap_30s_over_300s"].item()
        state_dict_market["vwap_rank_12x5s"] = pos["vwap_rank_12x5s"].item()
        state_dict_market["vwap_rank_12x30s"] = pos["vwap_rank_12x30s"].item()
        state_dict_market["cur_price/base_price"] = (
            pos["last_price"].item() / self.base_price - 1.0
        ) * 10000  # 当前价格与基准价格的比率
        state_dict_market["pv_corr_24x5s"] = pos["pv_corr_24x5s"].item()
        state_dict_market["shadow_vwap_dev"] = (
            (self.total_cum_turnover / (self.total_cum_volume + 1e-4) - self.base_price) / self.base_price * 10000
        )
        state_dict_market["fragility_24x5s_norm"] = pos["fragility_24x5s_norm"].item()
        state_dict_market["KER_24x5s"] = pos["KER_24x5s"].item()
        state_dict_market["Rejection_Bias_12x5s"] = pos["Rejection_Bias_12x5s"].item()
        state_dict_market["Vol_Squeeze"] = pos["Vol_Squeeze"].item()
        state_dict_market["Flow_Toxicity_12x5s"] = pos["Flow_Toxicity_12x5s"].item()

        state_dict_factor["signal_5s"] = float(pos["signal_5s_norm"].item())
        state_dict_factor["signal_1m"] = float(pos["signal_1m_norm"].item())
        state_dict_factor["signal_1h"] = float(pos["signal_1h_norm"].item())

        state_dict_volatility["vol_feat_300s"] = pos["vol_feat_300s"].item()
        state_dict_volatility["pressure_12x5s"] = pos["pressure_12x5s"].item()
        state_dict_volatility["smart_momentum_60x5s"] = pos["smart_momentum_60x5s"].item()
        state_dict_volatility["Gini_300s"] = pos["Gini_300s"].item()
        state_dict_volatility["vol_shock"] = pos["vol_shock"].item()
        state_dict_volatility["log_position"] = np.log(1 + np.abs(self.to_position) / 50)
        assert self.to_position != 0, "to_position should not be zero "

        state_dict_process["pos_ratio"] = self.realized_position / self.to_position  # 已实现仓位占目标仓位的比例
        # state_dict_process["consumed_time_ratio"] = consumed_time / self.total_time  # 已消耗时间占总时间的比例
        if remain_time == 0:
            print(self.id, self.begin_time, self.end_time)
            raise ValueError("No remaining time in the episode.")
        state_dict_process["participate_rate"] = (
            np.abs(self.realized_position) / ((self.total_cum_volume) + 1) / self.rate_upper_bound
        )  # 已实现仓位占总成交量的比例，经过归一化
        state_dict_process["market_ratio"] = self.total_cum_volume / (np.abs(self.to_position) * 50 + 1e-4)
        state_dict_process["gap_to_market"] = state_dict_process["pos_ratio"] - state_dict_process["market_ratio"]

        state_dict_slippage["exp_slippage_bp"] = (
            (
                self.calc_expected_slippage(pos, self.to_position - self.realized_position, self.base_price, self.side)
                + self.my_cum_slippage
            )
            / self.total_turnover
            * 10000
        )  # 预期滑点，单位为基点
        state_dict_slippage["now_slippage_bp"] = (
            (self.my_cum_slippage) / (self.my_cum_turnover + 1) * 10000
        )  # 当前滑点，单位为基点

        state_dict_event["event_open_rush"] = pos["event_open_rush"].item()
        state_dict_event["event_close_rush"] = pos["event_close_rush"].item()
        state_dict_event["event_vol_breakout"] = pos["event_vol_breakout"].item()
        state_dict_event["event_sig_spike"] = pos["event_sig_spike"].item()

        """print(
            "state_dict_market",
            state_dict_market,
            "\n",
            "state_dict_factor",
            state_dict_factor,
            "\n",
            "state_dict_volatility",
            state_dict_volatility,
            "\n",
            "state_dict_process",
            state_dict_process,
            "\n",
            "state_dict_slippage",
            state_dict_slippage,
            "\n",
            "state_dict_event",
            state_dict_event,
        )"""

        state_dict_trans_norm = self.trans_dict_norm(
            state_dict_market,
            state_dict_factor,
            state_dict_volatility,
            state_dict_process,
            state_dict_slippage,
            state_dict_event,
            self.side,
        )

        try:
            state_values = [state_dict_trans_norm[k] for k in self.STATE_FEATURES]
        except KeyError as e:
            raise KeyError(f"Feature key {e} missing in full_dict. Please check your dictionaries.")
        # print(state_dict_trans)  # prif action_arr.size == 1 else int(action_arr.argmax())
        # 拼接到 State 后面
        # print(state_dict)
        State = np.array(state_values, dtype=np.float32)
        if self.mode == "sample" and self.noise_std_ratio > 0.0:
            # 生成均值为 1.0，标准差为 noise_std_ratio 的正态分布随机数
            noise_multiplier = np.random.normal(loc=1.0, scale=self.noise_std_ratio, size=State.shape).astype(
                np.float32
            )
            State = State * noise_multiplier
        # print(State)
        # time.sleep(1)
        if random.random() < 0.00000004:
            print("state", State.shape, [f"{x:.3f}" for x in State])

        # print(State.shape, "State shape")
        # print(State)
        # print("State:", State)
        if not np.isfinite(State).all():
            print("Warning: Non-finite values detected in state features, replacing with zeros.")
            print(State)
            print(state_dict_trans_norm)
            print(self.samples)
            State = np.nan_to_num(State, nan=0.0, posinf=0.0, neginf=0.0)
            raise ValueError("Non-finite values detected in state features.")
        return State

    def Calc_reward(self, delta_p, base_price, vwap) -> float:
        # print("delta_p:", delta_p, "mid_price:", mid_price, "vwap:", vwap)
        penalty = 0.0
        slippage = delta_p * (vwap - base_price)
        reward = -slippage
        reward -= penalty
        return reward

    @staticmethod
    def convert_action_for_env(action):
        # return action
        return 0.04 * (action + 1)

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
        if self.position is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        state = self.get_state()
        terminated = False
        truncated = False
        cur_rate = self.action_range[int(action)] if self.if_discrete else self.convert_action_for_env(action)
        # cur_rate = 0.08  #! debug
        # if random.random() < 0.00005:
        #    print("action:", cur_rate)

        # print(cur_rate, "cur_rate") #pr
        dh = self.to_position - self.realized_position
        side = np.sign(dh)
        dh = abs(dh)
        cap = cur_rate * self.position.iloc[self.time_idx + 1]["volume"]
        cap_int = int(cap)
        cap_xs = cap - cap_int
        num = np.random.random()
        if num < cap_xs:
            cap_int += 1
        cap = cap_int
        dh = min(dh, cap)

        self.realized_position = int(self.realized_position + side * dh)
        """print(
            self.from_position,
            "-->",
            self.to_position,
            "side:",
            side,
            "dh:",
            dh,
            "realized_position:",
            self.realized_position,
            "base_price:",
            self.base_price,
            "vwap:",
            self.position.iloc[self.time_idx + 1]["vwap_5s"],
        )"""
        self.time_idx += 1
        self.time_curr += self.freq
        self.my_cum_turnover += dh * self.position.iloc[self.time_idx]["vwap_5s"]
        self.my_cum_slippage += dh * side * (self.position.iloc[self.time_idx]["vwap_5s"] - self.base_price)
        self.total_cum_volume += self.position.iloc[self.time_idx]["volume"]
        self.total_cum_turnover += (
            self.position.iloc[self.time_idx]["volume"] * self.position.iloc[self.time_idx]["vwap_5s"]
        )
        reward = self.Calc_reward(  #! 这里必须要带方向的变化量
            dh * side, self.base_price, self.position.iloc[self.time_idx]["vwap_5s"]
        )
        self.cum_reward += reward
        if self.time_idx == len(self.position) - 1:
            terminated = True
            if self.realized_position != self.to_position:
                self.uncompleted = True

        penalty = 0
        reward -= penalty

        while self.realized_position == self.to_position and not terminated:
            self.time_idx += 1
            self.time_curr += self.freq
            if self.time_idx == len(self.position) - 1:
                terminated = True
                break
        # print(reward, "reward")
        if self.time_idx == len(self.position) - 1:
            # print("!")
            if self.uncompleted:
                self.tot_uncompleted += 1
                final_slippage = (
                    self.calc_expected_slippage(
                        self.position.iloc[self.time_idx],
                        self.to_position - self.realized_position,
                        self.base_price,
                        self.side,
                        calc_realized_vol=True,
                    )
                    * 1.0
                )  # pe
                self.cum_reward -= final_slippage
                reward -= final_slippage
            assert (
                self.realized_position == self.to_position
            ), f"Realized position does not match target position. {self.realized_position} != {self.to_position}, id={self.id}, begin_time={self.begin_time}, end_time={self.end_time}, absolute_id={self.absolute_id}"

        if terminated:
            new_state = state
        else:
            new_state = self.get_state()

        self.history.append(new_state)
        if len(self.history) > self.K:
            self.history.pop(0)
        while len(self.history) < self.K:
            if self.padding0:
                self.history.insert(0, np.zeros_like(self.history[0]))
            else:
                self.history.insert(0, self.history[0])

        assert len(self.history) == self.K
        # print(self.history)
        state_stack = np.concatenate(self.history, axis=0)
        return (
            state_stack,
            reward / 1,
            terminated,
            truncated,
            {"absolute_id": self.absolute_id, "target_position": self.to_position},
        )


def check_stock_trading_env():
    import numpy as np
    import torch

    random.seed(527)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path = "/code/srwang/Finrl/log/20260219-131849_Agent_reinforce entropy0.0005  base_alpha0.1 lr1e-4 net-256-256-64 gpu0/actor__003683647488.pt"
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    actor = torch.load(actor_path, map_location=device, weights_only=False)
    config_path = "/code/srwang/elegantrl/envs/config_env.yaml"
    full_cfg = OmegaConf.load(config_path)
    env_cfg = full_cfg.env
    env = FutureExecEnv(env_cfg)
    env.if_discrete = True
    """state, info = env.reset(
        set_id={
            "id": "500006676",
            "begin_time": 60.0,
            "target_position": -30.0,
            "end_time": 795.0,
            "weight": 1.0,
            "type": "short",
            "class": "plus5_sample",
        }
    )  # Example ID and index"""
    # target_id = "100003380"
    target_id = "100000792"
    set_id = -1
    for idx, sample in enumerate(env.sample_pool):
        if sample["id"] == target_id and sample["type"] == "short":
            set_id = idx
            break
    print("set_id:", set_id)
    state, info = env.reset(set_id=set_id, sequential=False, mode="sample")  # Example ID and index
    # state, info = env.reset(sequential=True)  # Example ID and index
    # print("Initial State:", state)
    slippage_bp = []
    action = 1
    rrr = [0.00, 0.02, 0.04, 0.06, 0.08]
    reward_cum = 0
    id = 0
    from tqdm import tqdm

    # for _ in tqdm(range(500000000), desc="Simulation Progress"):
    for _ in range(500000000):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print(_, "state=", state.cpu().numpy())
        # print(state.shape)
        # action = actor(state).detach().cpu().item()
        action = 2
        # print(action)
        # action = 0
        if _ <= 300:
            # print("state: ", state)
            # print("Action:", rrr[action])
            ...
        state, reward, terminated, truncated, info = env.step(action)
        # print("State:", state)
        """print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)"""
        # print("Info:", info)
        reward_cum += reward
        print("Reward: ", reward)
        if terminated or truncated:
            slippage_bp.append(-reward_cum)
            if random.random() < 1:
                print(-reward_cum, np.mean(slippage_bp))
            # print("-----------end-----------")
            id += 1
            if id == 10:
                break
            state, info = env.reset(sequential=True)
            # print(info)
            reward_cum = 0

    print("slippage value=", slippage_bp)
    print(np.mean(slippage_bp))
    print(env.tot_uncompleted)


if __name__ == "__main__":
    check_stock_trading_env()
