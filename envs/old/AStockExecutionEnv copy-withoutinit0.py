import json
import os
import random
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch as th
import numba as nb
from multiprocessing import shared_memory
ARY = np.ndarray

import os
import random
import numpy as np
import pandas as pd
import atexit
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import List, Tuple, Dict, Any

from elegantRL.envs.sample_pool import SamplePool


@nb.njit
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


@nb.njit
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


@nb.njit
def _numba_fast_forward(time_idx: int,
                        max_steps: int,
                        sum_impact: float,
                        impact_decay: float,
                        vol_arr: np.ndarray,
                        amount_arr: np.ndarray,
                        signal_1m_row: np.ndarray,
                        signal_1m_norm: np.ndarray,
                        side: int,
                        impact_threshold: float,
                        signal_norm_threshold: float,
                        signal_impact_k: float,
                        signal_1m_row_offset: float,
                        ) -> Tuple[int, float, float, float, bool]:
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

    #while signal_1m_norm[time_idx] <= 0.65 or signal_1m_row[time_idx] < side * sum_impact * 8: 
    #while 1==-1:
    # while sum_impact * side > impact_threshold:
    #while signal_1m_norm[time_idx] <= signal_norm_threshold or signal_1m_row[time_idx] < side * sum_impact * 0: 
    while signal_1m_row[time_idx] < side * sum_impact * signal_impact_k + signal_1m_row_offset:
        time_idx += 1
        sum_impact *= impact_decay
        add_vol += vol_arr[time_idx]
        add_amt += amount_arr[time_idx]
        if time_idx >= max_steps:
            terminated = True
            break
    return time_idx, sum_impact, add_vol, add_amt, terminated


@nb.njit
def _numba_norm_bp(value_bp: float, side: int, apply_side: bool) -> float:
    scaled = np.sign(value_bp) * np.log1p(np.abs(value_bp) / 3.0)
    return scaled * side if apply_side else scaled


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
        #self.device = th.device("cpu") if cfg.gpu_id == -1 else th.device(f"cuda:{cfg.gpu_id}")
        self.mode:str = cfg.mode 
        self.impact_threshold: float = cfg.impact_threshold
        self.signal_norm_threshold: float = cfg.signal_norm_threshold
        self.signal_impact_k: float = cfg.signal_impact_k
        self.signal_1m_row_offset: float = cfg.signal_1m_row_offset

        self.env_name: str = "AStockExecutionEnv"
        print(f"Initialized AStockExecutionEnv with mode={self.mode}, num_envs={self.num_envs}, K={self.K}, state_dim={self.state_dim}, action_dim={self.action_dim}, if_discrete={self.if_discrete})")
        self.PREPROCESS_FEATURES = [
            "fast_signal_20260310_v1_norm", "buy_1m_ret_norm", "sell_1m_ret_norm", 
            "vwap_6s_over_60s", "vwap_rank_60s", "vwap_30s_over_300s", "vwap_rank_360s",
            "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
            "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
        ]
        self.STATIC_STATE_FEATURES = [
            "fast_signal_20260310_v1_norm", "signal_1m_norm", 
            "vwap_6s_over_60s", "vwap_rank_60s", "vwap_30s_over_300s", "vwap_rank_360s",
            "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
            "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
        ]
        self.DYNAMIC_STATE_FEATURES = [
            "pos_ratio", "time_ratio", "gap", "level_1_baseprice_bp","shadow_vwap_dev_bp",
            "sum_impact_bp", "now_slippage_bp","exp_slippage_bp","L1_capacity",
            "log_position"
        ]
        self.STATE_FEATURES = self.STATIC_STATE_FEATURES + self.DYNAMIC_STATE_FEATURES
        raw_mapping = OmegaConf.to_container(cfg.action_mapping, resolve=True)
        if isinstance(raw_mapping, dict):
            self.action_mapping: Dict[int, Tuple[float, int]] = {
                int(str(k)): tuple(v) for k, v in raw_mapping.items()
            }
        else:
            raise ValueError(f"action_mapping must be a dict, got {type(raw_mapping)}: {raw_mapping}")
        assert self.action_dim == len(self.action_mapping), \
            f"YAML 中的 action_dim ({self.action_dim}) 与 action_mapping 的长度 ({len(self.action_mapping)}) 不一致！"

        train_date_list = []
        with open(cfg.train_date_list_path, 'r') as f:
                train_date_list = [line.strip() for line in f if line.strip()]
        test_date_list = []
        with open(cfg.test_date_list_path, 'r') as f:
                test_date_list = [line.strip() for line in f if line.strip()]
        self.train_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir,
            case_dir=cfg.case_dir,
            date_list=train_date_list,
            max_limit=int(1e15)
        )
        self.test_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir,
            case_dir=cfg.case_dir,
            date_list=test_date_list,
            max_limit=int(1e15)
        )
        self.train_pool_size:int= self.train_pool.sample_pool_size
        self.test_pool_size:int = self.test_pool.sample_pool_size
        # 环境运行时变量
        self.history : List[np.ndarray] = []

        self._l2_levels: int = 10
        self._impact_decay: float = np.exp(-self.lambda_decay)
        self._state_index: Dict[str, int] = {k: i for i, k in enumerate(self.STATE_FEATURES)}

        index_name = OmegaConf.select(cfg, "shm_index_name", default=None)
        index_size = OmegaConf.select(cfg, "shm_index_size", default=0)
        if index_name:
            # 从共享内存中恢复出完整的元数据字典索引
            try:
                idx_shm = shared_memory.SharedMemory(name=index_name)
                # 使用 pickle 瞬间恢复
                import pickle
                self.shm_metadata = pickle.loads(idx_shm.buf[:index_size])
                idx_shm.close() # 映射完即关闭句柄，不影响内存存在
            except Exception as e:
                print(f"读取索引内存失败: {e}")
                self.shm_metadata = {}
        else:
            self.shm_metadata = {}

        self._active_shms: list = []
        assert len(self.shm_metadata) > 0, "Shared memory metadata is empty. Preloading may have failed or not been performed."
        # 如果pool里有没有的就raise error
        for sample in self.train_pool.sample_pool + self.test_pool.sample_pool:
            file_path = sample["file_path"]
            if file_path not in self.shm_metadata:
                raise ValueError(f"File path {file_path} from sample pool not found in shared memory metadata. Check if preload_data_to_shm was successful and covers all files.")
       
    def _close_shm(self):
        """释放上一个 case 的内存指针，防内存泄漏"""
        for shm in self._active_shms:
            shm.close()
        self._active_shms.clear()

    def _map_shm(self, meta_info: dict) -> np.ndarray:
        try:
            shm = shared_memory.SharedMemory(name=meta_info["name"])
            arr = np.ndarray(meta_info["shape"], dtype=meta_info["dtype"], buffer=shm.buf)
            self._active_shms.append(shm)
            return arr
        except FileNotFoundError:
            raise RuntimeError(f"共享内存块 {meta_info['name']} 不存在！请检查主进程是否意外触发了清理。")
    def reset(self, set_id: int = -1,mode: str = "train") -> Tuple[ARY, dict]:
        # 获取当前 Case 数据
        if mode not in ["train", "test"]:
            raise ValueError(f"Unknown mode: '{mode}'. Supported modes are 'train' and 'test'.")
        self.sample_pool: SamplePool = self.train_pool if mode == "train" else self.test_pool   
        self.case_info: Dict = self.sample_pool.get_sample(method=self.sample_method, set_id=set_id)
        # 加载 Tick 数据并截取当前 15min Window
        file_path = self.case_info["file_path"]

        #if file_path in self.shm_metadata:
        self._close_shm() # 关闭旧内存
        meta = self.shm_metadata[file_path]
        #print(meta)
        self._mid_arr = self._map_shm(meta["mid"])
        self._vwap_arr = self._map_shm(meta["vwap"])
        self._vol_arr = self._map_shm(meta["vol"])
        self._amount_arr = self._map_shm(meta["amount"])
        self._cmf_arr = self._map_shm(meta["cmf"])
        self._ask_px_arr = self._map_shm(meta["ask_px"])
        self._bid_px_arr = self._map_shm(meta["bid_px"])
        self._ask_vol_arr = self._map_shm(meta["ask_vol"])
        self._bid_vol_arr = self._map_shm(meta["bid_vol"])
        self._buy_1m_ret_arr = self._map_shm(meta["buy_1m_ret"])
        self._sell_1m_ret_arr = self._map_shm(meta["sell_1m_ret"])
        self._buy_1m_ret_norm_arr = self._map_shm(meta["buy_1m_ret_norm"])
        self._sell_1m_ret_norm_arr = self._map_shm(meta["sell_1m_ret_norm"])
        self._feature_arr = self._map_shm(meta["features"])
        
        # 目标与进度初始化
        self.target_position: float = self.case_info["target_change"]
        assert self.target_position != 0.0, f"Target position cannot be zero in sample {self.case_info['id']}"
        self.realized_position: float = 0.0
        self.side: int = np.sign(self.target_position)
        self.uncompleted_flag: bool = False
        
        # 金融状态初始化
        self.sum_impact: float = 0.0
        self.base_price: float = self._mid_arr[0]
        self.total_amount: float = abs(self.target_position * self.base_price)
        self.actual_cashflow: float = 0.0  
        self.ideal_cashflow: float = 0.0
        self.global_cum_volume: float = 0.0
        self.global_cum_amount: float = 0.0
        self.actual_amount: float = 0.0
        self.begin_flag: bool = True 
        # 时间指针初始化
        self.time_idx: int = 0
        self.max_steps: int = self._mid_arr.shape[0] - 1

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
        state_dict_norm["vwap_6s_over_60s"] = _numba_norm_bp(state_dict["vwap_6s_over_60s"], side, True)
        state_dict_norm["vwap_rank_60s"] = state_dict["vwap_rank_60s"] * side
        state_dict_norm["vwap_30s_over_300s"] = _numba_norm_bp(state_dict["vwap_30s_over_300s"], side, True)
        state_dict_norm["vwap_rank_360s"] = state_dict["vwap_rank_360s"] * side
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
        state_dict_norm["level_1_baseprice_bp"] = _numba_norm_bp(state_dict["level_1_baseprice_bp"], side, True)
        state_dict_norm["shadow_vwap_dev_bp"] = _numba_norm_bp(state_dict["shadow_vwap_dev_bp"], side, True)
        state_dict_norm["sum_impact_bp"] = _numba_norm_bp(state_dict["sum_impact_bp"], side, True)
        state_dict_norm["now_slippage_bp"] = _numba_norm_bp(state_dict["now_slippage_bp"], side, False)
        state_dict_norm["exp_slippage_bp"] = _numba_norm_bp(state_dict["exp_slippage_bp"], side, False)
        state_dict_norm["L1_capacity"] = np.clip(state_dict["L1_capacity"], 0.0, 2.0)
        state_dict_norm["log_position"] = np.log1p(np.abs(state_dict["position"])/2000)
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
        state_dict["position"] = self.target_position

        #print(state_dict)
        state_dict_norm = self._trans_dict_norm(state_dict, self.side)
        #if random.random() < 0.0000002:  #! debug
        #    print(f"Raw state dict at time_idx {self.time_idx}:", state_dict_norm)
        
        state_arr = np.array([state_dict_norm[k] for k in self.STATE_FEATURES], dtype=np.float32)
        
        if not np.isfinite(state_arr).all():
            raise ValueError(f"{self.case_info}\nState contains non-finite values at time_idx {self.time_idx}: {state_dict}")
            state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return state_arr

    def step(self, action: int) -> Tuple[ARY, float, bool, bool, dict]:
        #print(action)
        #action = 3 #! debug
        """if self.begin_flag:
            self.begin_flag = False
            action = 0"""
        target_pct, max_level = self.action_mapping[int(action)]
        uncompleted_delta = self.target_position - self.realized_position
        exp_vol = self.target_position * target_pct
        
        if self.side > 0:
            exp_vol = min(exp_vol, uncompleted_delta)
        else:
            exp_vol = max(exp_vol, uncompleted_delta)
            
        base_cost, actual_vol = self._compute_l2_execution(exp_vol, max_level)
        if(action==-1):
            print(action)
            print("time_idx:", self.time_idx, "signal_1m_norm:", self._buy_1m_ret_norm_arr[self.time_idx] if self.side > 0 else self._sell_1m_ret_norm_arr[self.time_idx],"impact:", self.sum_impact)
            print("time_idx:", self.time_idx, "target_position:", self.target_position, "realized_position:", self.realized_position, "target_pct:", target_pct, "exp_vol:", exp_vol,"actual_vol:", actual_vol)
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
            self._buy_1m_ret_arr if self.side > 0 else self._sell_1m_ret_arr,
            self._buy_1m_ret_norm_arr if self.side > 0 else self._sell_1m_ret_norm_arr,
            self.side,
            self.impact_threshold,
            self.signal_norm_threshold,
            self.signal_impact_k,
            self.signal_1m_row_offset
        )
        #signal_1m_row = self._buy_1m_ret_arr if self.side > 0 else self._sell_1m_ret_arr
        #signal_1m_norm = self._buy_1m_ret_norm_arr if self.side > 0 else self._sell_1m_ret_norm_arr
        #print(f"terminated({terminated}) Fast-forwarded to time_idx {self.time_idx} with signal_1m_norm={signal_1m_norm[self.time_idx]:.4f}, signal_1m={signal_1m_row[self.time_idx]:.4f}, sum_impact={self.sum_impact:.6f}")
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
                #self.actual_amount += abs(forced_cost) * (1 + self.sum_impact)
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

        if terminated:
            info["ideal_total_amount"] = self.total_amount
            info["actual_total_amount"] = self.actual_amount
        
        return (state_stack, step_reward, terminated, truncated, info)

def check_stock_trading_env():
    import numpy as np
    import torch
    import atexit
    from omegaconf import OmegaConf, open_dict
    
    # 【新增】引入你写好的预加载和清理工具
    from elegantRL.envs.shm_utils import preload_data_to_shm, cleanup_all_shm

    random.seed(527)
    
    config_path = "/home/songrui.wang/elegantRL/envs/config_env.yaml"
    full_cfg = OmegaConf.load(config_path)
    env_cfg = full_cfg.env
    
    # =========================================================
    # 💥 【核心模拟区】：在实例化 Env 之前，强行拉取数据进共享内存
    # =========================================================
    # 1. 临时创建池子获取文件列表 (与 main.py 逻辑一致)
    train_date_list = []
    with open(env_cfg.train_date_list_path, 'r') as f:
            train_date_list = [line.strip() for line in f if line.strip()] 
    test_date_list = []
    with open(env_cfg.test_date_list_path, 'r') as f:
            test_date_list = [line.strip() for line in f if line.strip()]
    temp_train_pool = SamplePool(
        meta_dir=env_cfg.meta_dir, case_dir=env_cfg.case_dir,
        date_list=train_date_list, max_limit=int(1e15)
    )
    temp_test_pool = SamplePool(
        meta_dir=env_cfg.meta_dir, case_dir=env_cfg.case_dir,
        date_list=test_date_list, max_limit=int(1e15)
    )
    
    # 2. 收集文件并去重
    all_files = set()
    for s in temp_train_pool.sample_pool: all_files.add(s["file_path"])
    for s in temp_test_pool.sample_pool: all_files.add(s["file_path"])
    all_files = list(all_files)
    all_files.sort() #! 可选：排序后拉取，方便调试时定位文件
    # 💡 调试技巧：本地单测只拉取 100 个文件，瞬间完成，防止 OOM
    test_files = all_files[:]
    print(f"\n[Local Test] 准备拉取 {len(test_files)} 个 case 进行独立测试...")
    len_test = len(test_files)
    # 3. 启动并行拉取 (测试环境给 8 个 worker 足够了)
    shm_info = preload_data_to_shm(test_files, max_workers=16)
    
    # 4. 把生成的索引块凭证强行塞进 env_cfg 中
    with open_dict(env_cfg):
        env_cfg.shm_index_name = shm_info["index_shm_name"]
        env_cfg.shm_index_size = shm_info["index_shm_size"]
        
    # 5. 注册打扫钩子，防止测试中断导致内存泄漏
    atexit.register(cleanup_all_shm, shm_info)
    # =========================================================

    # 现在 env_cfg 里已经有取件码了，再初始化 Env 就不会报错了！
    env = AStockExecutionEnv(env_cfg)
    env.if_discrete = True
    
    state, info = env.reset(mode="train")  
    print(f"Initial Info: {info}")
    
    slippage_pnl = []
    reward_cum = 0
    id = 0
    device = "cpu"

    #actor_path = "/home/songrui.wang/RL_training/log/20260406-201001_modsac_discrete size200000 rep2/actor__009868148736_-0058.459.pt"
    actor_path = "/home/songrui.wang/RL_training/log/20260406-200608_reinforce size200000 rep2/actor__013542359040_-0009.683.pt"
    actor = torch.load(actor_path, map_location=device, weights_only=False)
    actor.eval()
    #torch.set_num_threads(1)

    import onnxruntime as ort
    onnx_path = "actor_static.onnx"
    dummy_state = torch.zeros((1, env.state_dim), dtype=torch.float32)
    torch.onnx.export(actor, (dummy_state,), onnx_path, 
                      input_names=['state'], output_names=['action'])
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    import time
    # 4.97 4.34 3.9 3.33 1.72
    actor_time = 0
    begin_time = time.time()
    total_amount_cum = 0.0
    actual_amount_cum = 0.0
    with torch.no_grad():
        for _ in range(500000000):
            #state_ts = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            state_np = state.astype(np.float32).reshape(1, -1)
            #action = 3 #! Debug action: 始终执行固定动作
            begin_actor_time = time.time()
            #action = actor(state_ts).item()
            action_array = ort_session.run(None, {'state': state_np})[0]
            action = int(action_array.item())
            actor_time += time.time() - begin_actor_time
            if _ > 200000000:
                break
            state, reward, terminated, truncated, info = env.step(action)
            reward_cum += reward
            
            if terminated or truncated:
                slippage_pnl.append(-reward_cum)
                total_amount_cum += info["ideal_total_amount"]
                actual_amount_cum += info["actual_total_amount"]
                if random.random() < 1.0: # 100% 打印每次结束的 slip
                    print(f"[Case {id:04d}] [Step {_}] slippage={-reward_cum:.6f}, mean={np.mean(slippage_pnl):.6f}",info["actual_total_amount"], info["ideal_total_amount"])
                
                id += 1
                if id >= len_test: # 跑完我们拉取的所有 test_files 就结束
                    print("所有测试 case 运行完毕！")
                    break
                
                state, info = env.reset(mode="train")
                reward_cum = 0
    print(f"Total testing time: {time.time() - begin_time:.2f} seconds", f"(Actor inference time: {actor_time:.2f} seconds)")
    print("最终平均 Slippage value (pnl) =", np.mean(slippage_pnl))
    print("total_amount_cum:", total_amount_cum, "actual_amount_cum:", actual_amount_cum, "ratio:", (actual_amount_cum) / (total_amount_cum + 1e-8))

if __name__ == "__main__":
    check_stock_trading_env()
