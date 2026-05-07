import itertools
import json
import os
import random
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import math

class SamplePool:
    def __init__(self, meta_dir: str, case_dir: str,
                 date_list: List[str], max_limit: int = int(1e15)):
        self.meta_dir = meta_dir
        self.case_dir = case_dir
        
        self.sample_pool: List[Dict] = []
        
        cnt_id:int = 0
        for date_str in date_list:
            parquet_path = os.path.join(self.meta_dir, f"{date_str}.parquet")
            
            if not os.path.exists(parquet_path):
                continue
                
            df_meta = pd.read_parquet(parquet_path)
            
            # 遍历当天的 metadata
            for row in df_meta.itertuples():
                raw_id = str(row.id) 
                id_str = raw_id.zfill(8) if raw_id.isdigit() else raw_id
                
                file_path = os.path.join(self.case_dir, date_str, f"{id_str}.feather")
                
                self.sample_pool.append({
                    "date": date_str,
                    "id": id_str,
                    "absolute_id": cnt_id,  
                    "code": str(row.code),
                    "weight": math.log1p(float(row.weight)), # type: ignore
                    "target_change": float(row.target_change), # type: ignore
                    "file_path": file_path
                })
                cnt_id += 1
                
                if len(self.sample_pool) >= max_limit:
                    print(f"Reached max_limit of {max_limit} samples. Stopping further loading.")
                    break
                
        if not self.sample_pool:
            raise ValueError(f"empty sample pool. Check your data paths.")
            
        # 计算采样权重
        raw_weights = [case["weight"] for case in self.sample_pool]
        total_weight = sum(raw_weights)
        
        if total_weight <= 0:
            raise ValueError("Total weight of samples is 0 or negative. Check your parquet data.")
            
        self.cum_weights = list(itertools.accumulate(raw_weights))


        self.sample_pool_size = len(self.sample_pool) 
        self._sequential_index = 0
        
        print(f"Successfully loaded {self.sample_pool_size} cases ")

    def get_sample(self, method: str = "random", set_id: int = -1) -> Dict: #! 这里可以方便拓展为一次sample若干个
        if set_id != -1:
            return self.sample_pool[set_id]
            
        if method == "random":
            return random.choice(self.sample_pool)
        elif method == "seq":
            sample = self.sample_pool[self._sequential_index]
            self._sequential_index = (self._sequential_index + 1) % len(self.sample_pool)
            return sample
        elif method == "weighted":
            # random.choices 返回的是一个 list，取 [0]
            return random.choices(self.sample_pool, cum_weights=self.cum_weights, k=1)[0]
        else:
            raise ValueError(
                f"Unknown sampling method: '{method}'. "
                f"Must be one of ['random', 'sequential', 'weighted']."
            )

    def __len__(self) -> int:
        return self.sample_pool_size