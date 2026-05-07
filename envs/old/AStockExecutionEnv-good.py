import json
import os
import random
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch as th
import numba as nb

np.random.seed(527)
random.seed(527)
ARY = np.ndarray

import os
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Dict, Any

from sample_pool import SamplePool


@nb.njit(cache=True)
def _numba_l2_execution(px_levels: np.ndarray, vol_levels: np.ndarray, exp_vol: float, max_level: int) -> Tuple[float, float]:
    if exp_vol == 0.0 or max_level <= 0:
        return 0.0, 0.0

    is_buy = exp_vol > 0.0
    sign = 1.0 if is_buy else -1.0
    rem_vol = abs(exp_vol)
    total_cost = 0.0
    actual_vol = 0.0

    upper = max_level if max_level < 10 else 10
    for i in range(upper):
        px = px_levels[i]
        v = vol_levels[i]
        if np.isnan(px) or px == 0.0 or np.isnan(v) or v == 0.0:
            continue

        if rem_vol <= v:
            total_cost += rem_vol * px
            actual_vol += rem_vol
            rem_vol = 0.0
            break
        total_cost += v * px
        actual_vol += v
        rem_vol -= v

    return total_cost, actual_vol * sign


@nb.njit(cache=True)
def _numba_forced_execution_cost(px_levels: np.ndarray, vol_levels: np.ndarray, mid_px: float, exp_vol: float) -> float:
    if exp_vol == 0.0:
        return 0.0

    is_buy = exp_vol > 0.0
    rem_vol = abs(exp_vol)
    total_cost = 0.0
    last_valid_px = mid_px

    for i in range(10):
        px = px_levels[i]
        v = vol_levels[i]
        if np.isnan(px) or px == 0.0 or np.isnan(v) or v == 0.0:
            continue

        last_valid_px = px
        if rem_vol <= v:
            total_cost += rem_vol * px
            rem_vol = 0.0
            break
        total_cost += v * px
        rem_vol -= v

    if rem_vol > 1e-8:
        penalty_px = (last_valid_px + 0.01) if is_buy else (last_valid_px - 0.01)
        penalty_px = max(0.01, penalty_px)
        total_cost += rem_vol * penalty_px

    return total_cost


@nb.njit(cache=True)
def _numba_fast_forward(time_idx: int,
                        max_steps: int,
                        sum_impact: float,
                        impact_decay: float,
                        vol_arr: np.ndarray,
                        amount_arr: np.ndarray) -> Tuple[int, float, float, float, bool]:
    add_vol = 0.0
    add_amt = 0.0
    terminated = False

    time_idx += 1
    sum_impact *= impact_decay
    add_vol += vol_arr[time_idx]
    add_amt += amount_arr[time_idx]

    if time_idx >= max_steps:
        terminated = True
        return time_idx, sum_impact, add_vol, add_amt, terminated

    while abs(sum_impact) >= 0.0002:
        time_idx += 1
        sum_impact *= impact_decay
        add_vol += vol_arr[time_idx]
        add_amt += amount_arr[time_idx]
        if time_idx >= max_steps:
            terminated = True
            break

    return time_idx, sum_impact, add_vol, add_amt, terminated


class AStockExecutionEnv:
    def __init__(self, cfg: DictConfig):
        # 1. 配置参数提取 (严格访问，缺失即报错)
        self.num_envs = cfg.num_envs
        self.K: int = cfg.K
        self.state_dim: int = cfg.state_dim
        assert self.state_dim % self.K == 0, f"state_dim={self.state_dim} must be divisible by K={self.K}"
        self.action_dim: int = cfg.action_dim
        self.lambda_decay: float = cfg.lambda_decay
        self.sample_method: str = cfg.sample_method
        self.if_discrete: bool = cfg.if_discrete
        self.padding0: bool = cfg.padding0
        self.max_step: int = cfg.max_step
        self.eval_num_workers: int = cfg.eval_num_workers
        self.noise_std_ratio: float = cfg.noise_std_ratio
        self.device = th.device("cpu") if cfg.gpu_id == -1 else th.device(f"cuda:{cfg.gpu_id}")
        self.mode:str = cfg.mode 
        self.env_name: str = "AStockExecutionEnv"
        print(f"Initialized AStockExecutionEnv with mode={self.mode}, num_envs={self.num_envs}, K={self.K}, state_dim={self.state_dim}, action_dim={self.action_dim}, if_discrete={self.if_discrete})")
        self.PREPROCESS_FEATURES = [
            "fast_signal_20260310_v1_norm", "buy_1m_ret_norm", "sell_1m_ret_norm",
            "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
            "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
        ]
        self.STATIC_STATE_FEATURES = [
            "fast_signal_20260310_v1_norm", "signal_1m_norm", 
            "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
            "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
        ]
        self.DYNAMIC_STATE_FEATURES = [
            "pos_ratio", "time_ratio", "gap", "level_1_baseprice_bp","shadow_vwap_dev_bp",
            "sum_impact_bp", "now_slippage_bp","exp_slippage_bp","L1_capacity",
            "log_position"
        ]
        self.STATE_FEATURES = self.STATIC_STATE_FEATURES + self.DYNAMIC_STATE_FEATURES
        
        # 动作空间定义 
        self.action_mapping:Dict[int, Tuple[float, int]] = {
            0: (0.0, 0),    
            1: (0.02, 1),   
            2: (0.05, 1),   
            3: (0.10, 1),   
            4: (0.10, 2),   
            5: (0.20, 1),  
            6: (0.20, 2),  
            7: (0.30, 2),  
        }
        
        self.train_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir,
            case_dir=cfg.case_dir,
            start_date=cfg.train_start,
            end_date=cfg.train_end,
            max_limit=int(1e15)
        )
        self.test_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir,
            case_dir=cfg.case_dir,
            start_date=cfg.test_start,
            end_date=cfg.test_end,
            max_limit=int(1e15)
        )
        self.train_pool_size:int= self.train_pool.sample_pool_size
        self.test_pool_size:int = self.test_pool.sample_pool_size
        # 环境运行时变量
        self.history : List[np.ndarray] = []

        self._l2_levels: int = 10
        self._impact_decay: float = np.exp(-self.lambda_decay)
        self._state_index: Dict[str, int] = {k: i for i, k in enumerate(self.STATE_FEATURES)}

    def _build_numpy_cache(self) -> None:
        self._mid_arr = self.df["mid"].to_numpy(dtype=np.float64, copy=False)
        self._vwap_arr = self.df["vwap"].to_numpy(dtype=np.float64, copy=False)
        self._vol_arr = self.df["vol"].to_numpy(dtype=np.float64, copy=False)
        self._amount_arr = self.df["amount"].to_numpy(dtype=np.float64, copy=False)
        self._cmf_arr = self.df["cmf_v2"].to_numpy(dtype=np.float64, copy=False)

        self._ask_px_arr = np.column_stack([
            self.df[f"sale{i}"].to_numpy(dtype=np.float64, copy=False) for i in range(1, 11)
        ])
        
        self._bid_px_arr = np.column_stack([
            self.df[f"buy{i}"].to_numpy(dtype=np.float64, copy=False) for i in range(1, 11)
        ])

        # 注意：如果是 int32，保持 int32
        self._ask_vol_arr = np.column_stack([
            self.df[f"sc{i}"].to_numpy(dtype=np.int64, copy=False) for i in range(1, 11)
        ])
        
        self._bid_vol_arr = np.column_stack([
            self.df[f"bc{i}"].to_numpy(dtype=np.int64, copy=False) for i in range(1, 11)
        ])

        self._feature_arr = np.column_stack([
            self.df[feat].to_numpy(dtype=np.float32, copy=False) for feat in self.PREPROCESS_FEATURES
        ])
        
    def reset(self, set_id: int = -1,mode: str = "train") -> Tuple[ARY, dict]:
        # 获取当前 Case 数据
        if mode not in ["train", "test"]:
            raise ValueError(f"Unknown mode: '{mode}'. Supported modes are 'train' and 'test'.")
        self.sample_pool: SamplePool = self.train_pool if mode == "train" else self.test_pool   
        self.case_info: Dict = self.sample_pool.get_sample(method=self.sample_method, set_id=set_id)
        
        # 加载 Tick 数据并截取当前 15min Window
        file_path = self.case_info["file_path"]
        raw_df = pd.read_feather(file_path)
        self.df = raw_df
        self._build_numpy_cache()
        
        # 目标与进度初始化
        self.target_position: float = self.case_info["target_change"]
        assert self.target_position != 0.0, f"Target position cannot be zero in sample {self.case_info['id']}"
        self.realized_position: float = 0.0
        self.side: int = np.sign(self.target_position)
        self.uncompleted_flag: bool = False
        
        # 金融状态初始化
        self.sum_impact: float = 0.0
        self.base_price: float = self.df.iloc[0]['mid']
        self.total_amount: float = abs(self.target_position * self.base_price)
        self.actual_cashflow: float = 0.0  
        self.ideal_cashflow: float = 0.0
        self.global_cum_volume: float = 0.0
        self.global_cum_amount: float = 0.0
        self.actual_amount: float = 0.0
        
        # 时间指针初始化
        self.time_idx: int = 0
        self.max_steps: int = len(self.df) - 1

        self.history.clear()
        initial_state = self._get_state()
        
        while len(self.history) < self.K:
            if self.padding0:
                self.history.insert(0, np.zeros_like(initial_state))
            else:
                self.history.insert(0, initial_state)
        assert len(self.history) == self.K
        state_stack = np.concatenate(self.history, axis=0)
        return state_stack, self.case_info

    def _compute_l2_execution(self, exp_vol: float, max_level: int) -> Tuple[float, float]:
        if exp_vol == 0:
            return 0.0, 0.0

        idx = self.time_idx
        if exp_vol > 0:
            px_levels = self._ask_px_arr[idx]
            vol_levels = self._ask_vol_arr[idx]
        else:
            px_levels = self._bid_px_arr[idx]
            vol_levels = self._bid_vol_arr[idx]
        total_cost, actual_vol = _numba_l2_execution(px_levels, vol_levels, float(exp_vol), int(max_level))

        return total_cost, actual_vol

    def _compute_forced_execution_cost(self, exp_vol: float) -> float:
        if exp_vol == 0:
            return 0.0

        idx = self.time_idx
        if exp_vol > 0:
            px_levels = self._ask_px_arr[idx]
            vol_levels = self._ask_vol_arr[idx]
        else:
            px_levels = self._bid_px_arr[idx]
            vol_levels = self._bid_vol_arr[idx]
        total_cost = _numba_forced_execution_cost(px_levels, vol_levels, self._mid_arr[idx], float(exp_vol))

        return total_cost
    def _trans_dict_norm(self, state_dict: Dict[str, float], side: int) -> Dict[str, float]:
        state_dict_norm = {}
        state_dict_norm["fast_signal_20260310_v1_norm"] = state_dict["fast_signal_20260310_v1_norm"] * side
        state_dict_norm["signal_1m_norm"] = (state_dict["buy_1m_ret_norm"] if side > 0 else state_dict["sell_1m_ret_norm"])
        state_dict_norm["pv_corr_40x3s"] = state_dict["pv_corr_40x3s"] * side
        state_dict_norm["KER_40x3s"] = state_dict["KER_40x3s"] * side
        state_dict_norm["Vol_Squeeze"] = state_dict["Vol_Squeeze"]
        state_dict_norm["vol_shock"] = state_dict["vol_shock"] 
        state_dict_norm["vol_feat_300s"] = state_dict["vol_feat_300s"]
        state_dict_norm["smart_momentum_300s"] = state_dict["smart_momentum_300s"] * side
        state_dict_norm["Gini_300s"] = state_dict["Gini_300s"]
        state_dict_norm["pressure_60s"] = state_dict["pressure_60s"] * side
        state_dict_norm["cmf_v2_norm"] = state_dict["cmf_v2_norm"]
        state_dict_norm["pos_ratio"] = state_dict["pos_ratio"]
        state_dict_norm["time_ratio"] = state_dict["time_ratio"]
        state_dict_norm["gap"] = state_dict["gap"]
        state_dict_norm["level_1_baseprice_bp"] = state_dict["level_1_baseprice_bp"] * side
        state_dict_norm["shadow_vwap_dev_bp"] = state_dict["shadow_vwap_dev_bp"] * side
        state_dict_norm["sum_impact_bp"] = state_dict["sum_impact_bp"] * side
        state_dict_norm["now_slippage_bp"] = state_dict["now_slippage_bp"]
        state_dict_norm["exp_slippage_bp"] = state_dict["exp_slippage_bp"]
        state_dict_norm["L1_capacity"] = state_dict["L1_capacity"]
        state_dict_norm["log_position"] = state_dict["log_position"]
        return state_dict_norm
    def _get_state(self) -> np.ndarray:
        idx = self.time_idx
        feat = self._feature_arr[idx]

        state_dict: Dict = {}
        for i, k in enumerate(self.PREPROCESS_FEATURES):
            state_dict[k] = feat[i]

        state_dict["pos_ratio"] = np.abs((self.realized_position) / (self.target_position))
        state_dict["time_ratio"] = self.time_idx / self.max_steps
        state_dict["gap"] = state_dict["pos_ratio"] - state_dict["time_ratio"]

        level_1_price = self._ask_px_arr[idx, 0] if self.side > 0 else self._bid_px_arr[idx, 0]
        if np.isnan(level_1_price):
            level_1_price = self._mid_arr[idx]
        state_dict["level_1_baseprice_bp"] = ((level_1_price - self.base_price) / self.base_price * 10000)
        state_dict["shadow_vwap_dev_bp"] = (self.global_cum_amount / (self.global_cum_volume + 1e-8) - self.base_price) / self.base_price * 10000 if self.global_cum_volume > 1e-8 else 0.0
        state_dict["sum_impact_bp"] = self.sum_impact * 10000
        state_dict["now_slippage_bp"] = (self.actual_cashflow - self.ideal_cashflow) / (self.actual_amount + 1e-8) * 10000 if self.actual_amount > 1e-8 else 0.0
        uncompleted_pos = self.target_position - self.realized_position
        if abs(uncompleted_pos) > 1e-6:
            exp_cost = self._compute_forced_execution_cost(uncompleted_pos)
            exp_cashflow = -np.sign(uncompleted_pos) * exp_cost * (1 + self.sum_impact)
            ideal_exp_cashflow = -uncompleted_pos * self.base_price
            exp_pnl = exp_cashflow - ideal_exp_cashflow
            exp_slippage_bp = ((exp_pnl + (self.actual_cashflow - self.ideal_cashflow)) / (self.total_amount)) * 10000
        else:
            exp_slippage_bp = 0.0
        state_dict["exp_slippage_bp"] = exp_slippage_bp

        state_dict["L1_capacity"] = (self._ask_vol_arr[idx, 0] if self.side > 0 else self._bid_vol_arr[idx, 0]) / (abs(self.target_position))
        state_dict["log_position"] = self.target_position

        #print(state_dict)
        state_dict_norm = self._trans_dict_norm(state_dict, self.side)
        
        state_arr = np.array([state_dict_norm[k] for k in self.STATE_FEATURES], dtype=np.float32)
        
        if not np.isfinite(state_arr).all():
            raise ValueError(f"{self.case_info}\nState contains non-finite values at time_idx {self.time_idx}: {state_dict}  --- {self.df.iloc[self.time_idx].to_dict()}")
            state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return state_arr

    def step(self, action: int) -> Tuple[ARY, float, bool, bool, dict]:
        #print(action)
        action = 4 #! debug
        target_pct, max_level = self.action_mapping[int(action)]
        
        uncompleted_delta = self.target_position - self.realized_position
        exp_vol = self.target_position * target_pct

        if self.side > 0:
            exp_vol = min(exp_vol, uncompleted_delta)
        else:
            exp_vol = max(exp_vol, uncompleted_delta)
            
        base_cost, actual_vol = self._compute_l2_execution(exp_vol, max_level)
        step_reward = 0.0
        if actual_vol != 0:
            cashflow_change = -np.sign(actual_vol) * base_cost * (1 + self.sum_impact)
            ideal_cashflow_change = -actual_vol * self.base_price
             
            step_reward = cashflow_change - ideal_cashflow_change
            
            self.actual_cashflow += cashflow_change
            self.ideal_cashflow += ideal_cashflow_change
            self.realized_position += actual_vol
            self.actual_amount += abs(base_cost) * (1 + self.sum_impact) #! 在单次调仓的假设里 和 actual_cashflow 完全一致 只不过是abs累加
            cmf = self._cmf_arr[self.time_idx]
            self.sum_impact += np.sign(actual_vol) * cmf * np.sqrt(abs(actual_vol))

        self.time_idx, self.sum_impact, add_vol, add_amt, terminated = _numba_fast_forward(
            self.time_idx,
            self.max_steps,
            self.sum_impact,
            self._impact_decay,
            self._vol_arr,
            self._amount_arr,
        )
        self.global_cum_volume += add_vol
        self.global_cum_amount += add_amt
        truncated = False
            
        if abs(self.realized_position - self.target_position) < 1e-4:
            terminated = True
            
        # 结算未完成部分11档强平
        if terminated:
            uncompleted_pos = self.target_position - self.realized_position
            if abs(uncompleted_pos) > 1e-4:
                self.uncompleted_flag = True
                
                forced_cost = self._compute_forced_execution_cost(uncompleted_pos)
                
                penalty_cashflow = -np.sign(uncompleted_pos) * forced_cost * (1 + self.sum_impact)
                ideal_cost_for_uncompleted = -uncompleted_pos * self.base_price
                
                step_reward += (penalty_cashflow - ideal_cost_for_uncompleted)
                self.actual_cashflow += penalty_cashflow
                self.realized_position += uncompleted_pos
                self.ideal_cashflow += ideal_cost_for_uncompleted
                self.actual_amount += abs(forced_cost) * (1 + self.sum_impact)
                assert self.ideal_cashflow - (-self.target_position * self.base_price) < 1e-4, f"Ideal cashflow should be exactly target_position * base_price, but got {self.ideal_cashflow} vs {-self.target_position * self.base_price}"
            assert self.target_position - self.realized_position < 1e-4, f"At termination, uncompleted position should be close to zero, but got {self.target_position - self.realized_position}"
                
        # 整理输出状态
        new_state = self._get_state() if not terminated else self.history[-1] 
        self.history.append(new_state)
        self.history.pop(0)
        state_stack = np.concatenate(self.history, axis=0)
        
        info = {
            "absolute_id": self.case_info["absolute_id"],
            "target_position": self.target_position,
        }
        
        return (state_stack, step_reward, terminated, truncated, info)

def check_stock_trading_env():
    import numpy as np
    import torch

    random.seed(527)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #actor_path = "/code/srwang/Finrl/log/20260219-131849_Agent_reinforce entropy0.0005  base_alpha0.1 lr1e-4 net-256-256-64 gpu0/actor__003683647488.pt"
    #device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    #actor = torch.load(actor_path, map_location=device, weights_only=False)
    config_path = "/home/songrui.wang/elegantRL/envs/config_env.yaml"
    full_cfg = OmegaConf.load(config_path)
    env_cfg = full_cfg.env
    env = AStockExecutionEnv(env_cfg)
    env.if_discrete = True
    state, info = env.reset(mode="train")  # Example ID and index
    print(info)
    # state, info = env.reset(sequential=True)  # Example ID and index
    # print("Initial State:", state)
    slippage_bp = []
    reward_cum = 0
    id = 0
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # for _ in tqdm(range(500000000), desc="Simulation Progress"):
    for _ in range(500000000):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # print(_, "state=", state.cpu().numpy())
        # print(state.shape)
        # action = actor(state).detach().cpu().item()
        action = 4
        # print(action)
        # action = 0
        if _ > 20000000000:

            break
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
        # print("Reward: ", reward)
        if terminated or truncated:
            slippage_bp.append(-reward_cum)

            if random.random() < 1:
                print(f"slippage={-reward_cum:.6f}, mean={np.mean(slippage_bp):.6f}")
            # print("-----------end-----------")
            id += 1
            if id == 10000:
                break
            state, info = env.reset(mode="train")  # Example ID and index
            #print(info)
            reward_cum = 0

    print("slippage value=", slippage_bp)
    print(np.mean(slippage_bp))
   # print(env.tot_uncompleted)


if __name__ == "__main__":
    check_stock_trading_env()