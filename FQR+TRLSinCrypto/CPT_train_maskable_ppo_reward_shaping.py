# -*- coding: utf-8 -*-
# train_maskable_ppo_reward_shaping_original.py

import os
import json
import signal
import pickle
from typing import Tuple
import numpy as np
import pandas as pd
import torch

from sb3cr_contrib.ppo_reward_shaping.ppo_reward_shaping import Maskable_ppo_reward_shaping
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *           # 表达式算子
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet

from crypto_data_calculator import CryptoDataCalculator

from alphagen.rl.policy import LSTMSharedNet as _LSTMSharedNet
import torch as _torch
from numbers import Number
# --- features extractor 包装：只负责把参数按正确位置传给 LSTMSharedNet ---
from alphagen.rl.policy import LSTMSharedNet as _LSTMSharedNet
import torch as _torch

class LSTMExtractor(_LSTMSharedNet):
    def __init__(self,
                 observation_space,
                 n_layers: int = 2,
                 d_model: int = 128,
                 dropout: float = 0.1,
                 device: _torch.device = None,
                 **kwargs):
        if device is None:
            device = _torch.device('cuda:0') if _torch.cuda.is_available() else _torch.device('cpu')
        # 严格按作者签名：(obs, n_layers, d_model, dropout, device)
        super().__init__(observation_space, int(n_layers), int(d_model), float(dropout), device)


def _to_int(x, default):
    try:
        if isinstance(x, Number):
            return int(x)
        return int(float(x))
    except Exception:
        return int(default)

def _to_float(x, default):
    try:
        if isinstance(x, Number):
            return float(x)
        return float(x)
    except Exception:
        return float(default)

class WrappedLSTMPolicy(_LSTMSharedNet):
    """
    适配算法签名 (observation_space, action_space, **kwargs)
    并对 n_layers/d_model/dropout/device 做清洗与兜底，防止被上层 kwargs 覆盖为非法对象。
    """
    def __init__(self,
                 observation_space,
                 action_space,                 # 忽略，不上传
                 n_layers=2,
                 d_model=128,
                 dropout=0.1,
                 device=None,
                 **kwargs):

        # --- 清洗：如果被上层 kwargs 覆盖为奇怪类型（比如函数），强制回落为默认 ---
        n_layers = _to_int(n_layers, 2) if not callable(n_layers) else 2
        d_model  = _to_int(d_model, 128) if not callable(d_model) else 128
        dropout  = _to_float(dropout, 0.1) if not callable(dropout) else 0.1

        if device is None or callable(device):
            device = _torch.device('cuda:0') if _torch.cuda.is_available() else _torch.device('cpu')

        # 避免同名脏键再通过 kwargs 传给父类
        for k in ["n_layers", "d_model", "dropout", "device"]:
            if k in kwargs:
                kwargs.pop(k, None)

        # 关键：严格用“位置参数”按作者顺序上传，彻底杜绝错位
        super().__init__(observation_space, n_layers, d_model, dropout, device)

# ===== 回调：每个 rollout 记录并打印关键指标、并落 CSV =====
from stable_baselines3.common.callbacks import BaseCallback
import time

class CustomCallback(BaseCallback):
    def __init__(
        self,
        pool,                     # AlphaPool 实例（训练用的那一个）
        test_calculator,          # 用于评估的 test calculator
        save_path,                # 保存 CSV 的目录
        name_prefix="crypto",
        show_freq_rollouts=1,     # 每多少个 rollout 打印一次；=1 就是"每个"
        verbose=0,
        train_calculator=None,    # 用于因子验证的训练计算器（可选）
        valid_calculator=None,    # 用于因子验证的验证计算器（可选）
        save_factors_freq=50,     # 每多少个 rollout 保存一次因子（0=不保存，仅在结束时保存）
    ):
        super().__init__(verbose)
        self.pool = pool
        self.test_calculator = test_calculator
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.show_freq_rollouts = max(1, int(show_freq_rollouts))
        self.train_calculator = train_calculator
        self.valid_calculator = valid_calculator
        self.save_factors_freq = int(save_factors_freq) if save_factors_freq > 0 else 0

        self._rollout_counter = 0
        self._csv_dir = None
        self._csv_file = None
        self._ts_tag = time.strftime("%Y%m%d_%H%M%S")

    def _on_training_start(self) -> None:
        # 日志目录
        os.makedirs(self.save_path, exist_ok=True)
        self._csv_dir = os.path.join(self.save_path, f"{self.name_prefix}_{self._ts_tag}")
        os.makedirs(self._csv_dir, exist_ok=True)
        self._csv_file = os.path.join(self._csv_dir, "rollout_log.csv")
        # 写 CSV 头
        with open(self._csv_file, "w") as f:
            f.write("num_timesteps,rollout_idx,pool_size,best_ic_ret,test_ic,test_long_short\n")

    def _on_rollout_end(self) -> None:
        # 评估（测试集）
        try:
            test_ic, test_ls = self.pool.test_ensemble(self.test_calculator)
        except Exception:
            # 某些早期阶段可能评估失败，给个 NaN 填位
            test_ic, test_ls = float("nan"), float("nan")

        # TensorBoard 记录
        assert self.logger is not None
        self.logger.record("pool/size", self.pool.size)
        # 有的实现里 best_ic_ret 可能不存在；做个稳妥的 getattr
        best_ic_ret = float(getattr(self.pool, "best_ic_ret", float("nan")))
        self.logger.record("pool/best_ic_ret", best_ic_ret)
        self.logger.record("test/ic", test_ic)
        self.logger.record("test/long_short_return", test_ls)

        # 累计 rollout 次数
        self._rollout_counter += 1

        # 控制台打印（按频率）
        if self._rollout_counter % self.show_freq_rollouts == 0:
            print(
                f"[rollout_end] steps={self.num_timesteps} | "
                f"pool.size={self.pool.size} | "
                f"best_ic_ret={best_ic_ret:.6f} | "
                f"test.ic={test_ic:.6f} | "
                f"test.long_short={test_ls:.6f}"
            )

        # 追加到 CSV
        try:
            with open(self._csv_file, "a") as f:
                f.write(
                    f"{self.num_timesteps},{self._rollout_counter},"
                    f"{self.pool.size},{best_ic_ret:.6f},{test_ic:.6f},{test_ls:.6f}\n"
                )
        except Exception as e:
            print("[warn] 写入 rollout_log.csv 失败：", e)
        
        # === 定期保存因子（如果配置了）===
        if (self.save_factors_freq > 0 and 
            self._rollout_counter % self.save_factors_freq == 0 and 
            self.pool.size > 0 and
            self.train_calculator is not None and 
            self.valid_calculator is not None):
            try:
                print(f"\n💾 [自动保存] 在第 {self._rollout_counter} 个 rollout 时保存因子 (池大小: {self.pool.size})...")
                # 创建临时保存目录
                temp_save_dir = os.path.join(self.save_path, f"{self.name_prefix}_{self._ts_tag}_checkpoints")
                os.makedirs(temp_save_dir, exist_ok=True)
                # 保存因子（不进行完整验证，只保存基本信息）
                self._quick_save_factors(temp_save_dir, self._rollout_counter)
                print(f"✅ 因子已保存到: {temp_save_dir}\n")
            except Exception as e:
                print(f"[warn] 定期保存因子失败：{e}")
        
        # === 早期播种（自动探测可用的加入方法名）===
        try:
            if getattr(self.pool, "size", 0) == 0:
                class _SeedExpr:
                    def evaluate(self, data, period):
                        close = data.get("close", period)
                        if close.shape[0] < 4:
                            return torch.zeros_like(close)
                        mom = close[3:] / (close[:-3] + 1e-12) - 1.0
                        pad = torch.zeros((3, close.shape[1]), dtype=close.dtype, device=close.device)
                        return torch.cat([pad, mom], dim=0)

                ic_seed = self.pool.calculator.calc_single_IC_ret(_SeedExpr())
                if np.isfinite(ic_seed):
                    # 1) 优先尝试“看起来像加入”的方法名
                    method_candidates = []
                    for name in dir(self.pool):
                        lname = name.lower()
                        if any(k in lname for k in ["add", "accept", "insert", "push", "submit"]):
                            if callable(getattr(self.pool, name)):
                                method_candidates.append(name)
                    method_candidates.extend(["add_expr", "add", "add_candidate", "add_last_candidate", "accept"])

                    used = False
                    for m in dict.fromkeys(method_candidates):  # 去重保序
                        if hasattr(self.pool, m):
                            try:
                                fn = getattr(self.pool, m)
                                # 常见签名：fn(expr) / fn(expr, **kwargs)
                                res = fn(_SeedExpr())
                                print(f"[early-seed] tried {m}, result={res}")
                                used = True
                                break
                            except TypeError:
                                # 有的签名可能是 (expr, stats)；给最小兜底
                                try:
                                    res = fn(_SeedExpr(), None)
                                    print(f"[early-seed] tried {m}(expr, None), result={res}")
                                    used = True
                                    break
                                except Exception as e2:
                                    print(f"[early-seed] {m} signature mismatch:", e2)
                            except Exception as e:
                                print(f"[early-seed] {m} failed:", e)

                    if not used:
                        # 2) 如果没有任何“加入”方法，就尝试找“候选队列”属性名，直接 append
                        for attr in ["candidates", "candidate_queue", "queue", "buffer"]:
                            if hasattr(self.pool, attr):
                                try:
                                    q = getattr(self.pool, attr)
                                    if hasattr(q, "append"):
                                        q.append(_SeedExpr())
                                        print(f"[early-seed] appended seed to pool.{attr}")
                                        used = True
                                        break
                                except Exception as e:
                                    print(f"[early-seed] append to {attr} failed:", e)

                    if not used:
                        print("[early-seed] still no usable hook on pool; will rely on relaxed thresholds only.")
        except Exception as e:
            print("[early-seed] seeding error:", e)

    def _quick_save_factors(self, save_dir, rollout_idx):
        """快速保存因子（不进行完整验证，只保存基本信息）"""
        import pickle
        from datetime import datetime
        
        factors_info = []
        for i in range(self.pool.size):
            try:
                expr = self.pool.exprs[i] if hasattr(self.pool, 'exprs') else None
                weight = self.pool.weights[i] if hasattr(self.pool, 'weights') else 0.0
                
                if expr is None:
                    continue
                
                # 只计算测试集的 IC（快速）
                try:
                    ic, ls = self.test_calculator.calc_single_IC_ret_with_ls(expr)
                except:
                    ic, ls = float('nan'), float('nan')
                
                factor_info = {
                    'index': i,
                    'weight': float(weight),
                    'expression_str': str(expr),
                    'test_ic': float(ic) if np.isfinite(ic) else np.nan,
                    'test_long_short': float(ls) if np.isfinite(ls) else np.nan,
                    'rollout_idx': rollout_idx,
                    'timestamp': datetime.now().isoformat()
                }
                factors_info.append(factor_info)
            except Exception as e:
                pass
        
        # 保存到文件
        checkpoint_file = os.path.join(save_dir, f"factors_checkpoint_rollout_{rollout_idx}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(factors_info, f)
        
        # 也保存一个 CSV 摘要
        if factors_info:
            import pandas as pd
            df_data = []
            for info in factors_info:
                df_data.append({
                    'index': info['index'],
                    'weight': info['weight'],
                    'test_ic': info['test_ic'],
                    'test_long_short': info['test_long_short'],
                    'expression': info['expression_str']
                })
            df = pd.DataFrame(df_data)
            csv_file = os.path.join(save_dir, f"factors_checkpoint_rollout_{rollout_idx}.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    def _on_step(self) -> bool:
        # 不需要逐步逻辑的话，返回 True 让训练继续
        return True


    # （可选）想看池里每条表达式的权重/表现，可以补一个方法：
    def show_pool_state(self, topk=10):
        try:
            # 你的 AlphaPool 若有导出字串的方法可直接用；否则这里只示意
            print(f"[pool] size={self.pool.size}")
            # 例如：print(self.pool.dumps(topk=topk))
        except Exception:
            pass


# ====== 1) 专家示范路径（改成你的绝对路径）======
EXPERT_DEMO_PATH = r"C:\Users\江尚霖\Desktop\QuantFactor\TRLSinCrypto\sb3_contrib\ppo_reward_shaping\expert_demo_crypto_15m.pkl"

# ====== 2) 目标函数（等价于 Ref(close,-6)/Ref(close,-1)-1）======
def target_fn(fields):
    c = fields["close"]
    return c.shift(-6) / c - 1

# ====== 3) 从 util_for_expert_demo 构造 panel（严格沿用 util 的输入结构）======
def build_panel_from_futures() -> dict:
    """直接从 futures 目录读取六字段面板数据"""
    import os
    import glob
    import pandas as pd
    import numpy as np
    
    DATA_DIR = "/Users/mac/Downloads/QFR/futures"
    TF = "15m"
    
    # 读取所有 feather 文件
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"*{TF}*.feather")))
    if not files:
        raise FileNotFoundError(f"未找到 {DATA_DIR} 下含 '{TF}' 的 feather 文件")
    
    print(f"找到 {len(files)} 个文件")
    
    # 读取并合并数据
    dfs = []
    for f in files:
        df = pd.read_feather(f)
        cols = {c.lower(): c for c in df.columns}
        
        # 容错处理列名
        ts = cols.get('timestamp', cols.get('date'))
        sym = cols.get('symbol', cols.get('instrument', cols.get('code')))
        
        # 如果没有找到时间戳列，跳过
        if ts is None:
            print(f"警告: {f} 缺少时间戳列")
            continue
        
        # 如果没有找到符号列，从文件名提取
        if sym is None:
            # 从文件名提取符号 (例如: "1INCH_USDT_USDT-15m-futures.feather" -> "1INCH")
            basename = os.path.basename(f)
            symbol_name = basename.split('_')[0].split('-')[0]
            df['symbol_from_file'] = symbol_name
            sym = 'symbol_from_file'
        
        # 检查必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = []
        for col in required_cols:
            if col not in cols:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"警告: {f} 缺少列: {missing_cols}")
            continue
            
        # 选择需要的列
        needed_cols = [ts, sym] + [cols[col] for col in required_cols]
        df = df[needed_cols].copy()
        df.columns = ['timestamp', 'symbol'] + required_cols
        
        # 处理时间戳
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("没有找到有效的数据文件")
    
    # 合并所有数据
    big = pd.concat(dfs, ignore_index=True).dropna(subset=['timestamp', 'symbol'])
    big = big.sort_values(['timestamp', 'symbol'])
    
    # 创建六字段面板
    panel = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        panel[field] = big.pivot(index='timestamp', columns='symbol', values=field)
        panel[field] = panel[field].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        panel[field] = panel[field].dropna(axis=1, how='all')
    
    # 计算 VWAP (Volume Weighted Average Price)
    if 'volume' in panel and 'close' in panel:
        panel['vwap'] = panel['close']  # 简化版，实际应该是 (high+low+close)/3
    
    # 确保所有面板有相同的索引和列
    base_idx = panel['close'].index.sort_values()
    base_cols = panel['close'].columns
    
    for field in panel:
        panel[field] = panel[field].reindex(index=base_idx, columns=base_cols)
    
    print(f"面板数据: {len(base_idx)} 个时间点, {len(base_cols)} 个合约")
    print(f"合约: {list(base_cols)}")
    
    return panel

def split_time_by_days(idx: pd.DatetimeIndex, train_days=60, valid_days=15, test_days=15):
    idx = idx.sort_values()
    # 提取唯一日期（去除时区信息，只保留日期部分）
    days = idx.normalize().unique()
    if len(days) < train_days + valid_days + test_days:
        raise ValueError("可用天数不足以按 60/15/15 切分")
    def day_end(d): return pd.Timestamp(d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    train_start, train_end = pd.Timestamp(days[0]), day_end(days[train_days-1])
    valid_start, valid_end = pd.Timestamp(days[train_days]), day_end(days[train_days+valid_days-1])
    test_start,  test_end  = pd.Timestamp(days[train_days+valid_days]), day_end(days[train_days+valid_days+test_days-1])
    
    # 确保返回的时间戳与原索引时区匹配（如果需要的话）
    return (train_start, train_end), (valid_start, valid_end), (test_start, test_end)

def make_calculators_from_panel(panel: dict):
    close_idx = panel["close"].index
    print(f"面板数据总日期数: {len(close_idx)}, 唯一日期数: {len(close_idx.normalize().unique())}")
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = split_time_by_days(close_idx, 60, 15, 15)
    print(f"训练集: {tr_s} 到 {tr_e}")
    print(f"验证集: {va_s} 到 {va_e}")
    print(f"测试集: {te_s} 到 {te_e}")
    calc_train = CryptoDataCalculator(panel, target_fn=target_fn, start=tr_s, end=tr_e)
    print(f"训练集创建成功，样本数: {len(calc_train._stock_data.index)}")
    calc_valid = CryptoDataCalculator(panel, target_fn=target_fn, start=va_s, end=va_e)
    print(f"验证集创建成功，样本数: {len(calc_valid._stock_data.index)}")
    calc_test  = CryptoDataCalculator(panel, target_fn=target_fn, start=te_s, end=te_e)
    print(f"测试集创建成功，样本数: {len(calc_test._stock_data.index)}")
    return calc_train, calc_valid, calc_test

def main(seed: int = 5, pool: int = 20, steps: int = 600_000):
    np.random.seed(seed)
    torch.manual_seed(seed)

    panel = build_panel_from_futures()
    calc_train, calc_valid, calc_test = make_calculators_from_panel(panel)
        # === probes: 触发带统计版 IC 计算 + 检查表达式输出维度 ===
    class _NaiveCloseMomExpr:
        def evaluate(self, data, period):
            close = data.get("close", period)   # 期望 [T, N]
            if close.shape[0] < 4:
                return torch.zeros_like(close)
            mom = close[3:] / (close[:-3] + 1e-12) - 1.0
            pad = torch.zeros((3, close.shape[1]), dtype=close.dtype, device=close.device)
            return torch.cat([pad, mom], dim=0)

    ic_probe, ls_probe = calc_train.calc_single_IC_ret_with_ls(_NaiveCloseMomExpr())
    print(f"[probe] naive expr => IC={ic_probe}, LS={ls_probe}")

    try:
        df_probe = calc_train._eval_expr(_NaiveCloseMomExpr())
        print("[expr-shape] naive expr matrix shape =", df_probe.shape)  # 期望 (T, N)
    except Exception as e:
        print("[expr-shape] evaluate failed:", e)


        # === 数据健康度自检：目标覆盖率 + 横截面样本量 ===
    try:
        y_tr = calc_train.get_target()  # DataFrame: index=时间, columns=合约
        notna_per_day = y_tr.notna().sum(axis=1)
        print("[data-check] train days =", len(y_tr.index),
              "symbols =", y_tr.shape[1],
              "median xsec count =", int(notna_per_day.median()),
              "p10 =", int(notna_per_day.quantile(0.1)),
              "p90 =", int(notna_per_day.quantile(0.9)),
              "overall_nan_ratio =", float(y_tr.isna().mean().mean()))
    except Exception as e:
        print("[data-check][ERROR] 读取训练目标失败：", e)

    # === 基线因子自检：用 very-simple baseline 因子算一次 IC，看看是否能得到非 NaN ===
    try:
        base_res = calc_train.debug_baselines()  # 下面补丁 B 会在 calculator 里提供
        print("[baseline-IC] 1) past_1bar_ret IC =", f"{base_res.get('ic_past1', 'NA')}")
        print("[baseline-IC] 2) zscore(close)   IC =", f"{base_res.get('ic_zclose', 'NA')}")
        print("[baseline-IC] 3) zscore(vwap)    IC =", f"{base_res.get('ic_zvwap', 'NA')}")
    except Exception as e:
        print("[baseline-IC][WARN] 无法计算基线 IC（看起来计算器字段/结构异常）：", e)


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    alpha_pool = AlphaPool(
        capacity=pool,
        calculator=calc_train,
        device=device,
    )

        # === 暂时放宽 AlphaPool 的接收阈值（有则改、无则跳）===
    relax_kv = [
        ("min_eval_points", 10),          # 单时点最少样本数
        ("min_n", 3),                     # 逐时点 min_n
        ("min_train_ic", -1.0),           # 训练 IC 下限
        ("min_train_rankic", -1.0),       # 训练 RankIC 下限
        ("min_valid_ic", -1.0),           # 验证 IC 下限
        ("min_coverage", 1),              # 覆盖的最少时点数
        ("require_positive_ls", False),   # 是否必须多空>0
        ("require_finite_valid", False),  # 是否要求验证集全 finite
        ("accept_negative_ic", True),     # 允许负IC（先收进来，后续再淘汰）
    ]
    for k, v in relax_kv:
        if hasattr(alpha_pool, k):
            try:
                setattr(alpha_pool, k, v)
                print(f"[relax] set {k} = {v}")
            except Exception as e:
                print(f"[relax] fail {k}: {e}")

    from sb3cr_contrib.common.maskable.policies import MaskableActorCriticPolicy
    env = AlphaEnv(alpha_pool)
    policy = MaskableActorCriticPolicy

    policy_kwargs = {
        "features_extractor_class": LSTMExtractor,
        "features_extractor_kwargs": {
            "n_layers": 2,
            "d_model": 128,
            "dropout": 0.1,
            "device": device,
        },
    }

    # === 新增：TensorBoard 根目录（可按你的项目路径改）===
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tb_root = os.path.join(current_dir, "tensorboard", "ppo_reward_shaping_crypto")
    os.makedirs(tb_root, exist_ok=True)

    # === 这里明确把 device 传给算法体，并打开 tensorboard_log ===
    algo = Maskable_ppo_reward_shaping(
        policy,
        env,
        gamma=1.0,
        policy_kwargs=policy_kwargs,
        device=device,
        tensorboard_log=tb_root,
        verbose=1,
    )

    # === 新增：立刻检查并打印“到底有没有用上 GPU”===
    print("[device-check] torch.cuda.is_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[device-check] cuda name =", torch.cuda.get_device_name(0))
    print("[device-check] policy.device =", algo.policy.device)
    # 兼容：有 CUDA 用 CUDA；否则尝试 MPS；都不可用时使用 CPU
    if torch.cuda.is_available():
        if not str(algo.policy.device).startswith("cuda"):
            try:
                algo.policy.to("cuda")
                print(f"[OK] 已迁移到 {algo.policy.device}")
            except Exception as e:
                print(f"[WARN] CUDA 迁移失败，将继续使用 {algo.policy.device}: {e}")
    else:
        # 为稳定性禁用 MPS 回退，直接使用 CPU（避免 MPS 在 nn.Embedding 等算子上的已知问题）
        if str(algo.policy.device).startswith("mps"):
            try:
                algo.policy.to("cpu")
                print("[INFO] 已从 MPS 回退到 CPU 以避免不兼容算子问题")
            except Exception as e:
                print(f"[WARN] 回退 CPU 失败，仍在 {algo.policy.device}: {e}")
        else:
            print("[INFO] 无 CUDA，使用 CPU 训练")

    # === 新增：挂上我们的回调（每个 rollout 都打印；想降频就把 show_freq_rollouts 改成 >1）===
    save_dir = os.path.join(current_dir, "checkpoints", "ppo_reward_shaping_crypto")
    callback = CustomCallback(
        pool=alpha_pool,
        test_calculator=calc_test,
        save_path=save_dir,
        name_prefix=f"pool{pool}_seed{seed}",
        show_freq_rollouts=1,    # 每个 rollout 打印；改成 5 表示每 5 个 rollout 打印一次
        train_calculator=calc_train,  # 用于定期保存因子（可选）
        valid_calculator=calc_valid,  # 用于定期保存因子（可选）
        save_factors_freq=0,    # 定期保存频率（设为 0 禁用，>0 表示每 N 个 rollout 保存一次）
    )

    # === 添加中断时保存因子池的功能 ===
    def save_pool_on_interrupt(signum, frame):
        """中断时保存因子池"""
        print(f"\n⚠️  收到中断信号，正在保存因子池...")
        pool_save_file = os.path.join(save_dir, f"alpha_pool_seed{seed}_interrupted.pkl")
        try:
            with open(pool_save_file, 'wb') as f:
                pickle.dump({
                    'alpha_pool': alpha_pool,
                    'calc_train': calc_train,
                    'calc_valid': calc_valid,
                    'calc_test': calc_test,
                    'save_dir': save_dir,
                    'seed': seed,
                    'pool_size': pool,
                    'interrupted': True
                }, f)
            print(f"💾 因子池已保存到: {pool_save_file}")
            print(f"   可以使用此文件手动提取因子: python extract_factors_manual.py --pool_file {pool_save_file}")
        except Exception as e:
            print(f"❌ 保存因子池失败: {e}")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, save_pool_on_interrupt)
    signal.signal(signal.SIGTERM, save_pool_on_interrupt)

    # === 学习：tb_log_name 会作为 TensorBoard 下的子目录名字 ===
    try:
        algo.learn(total_timesteps=steps, callback=callback, tb_log_name=f"crypto_pool{pool}_seed{seed}")
    except KeyboardInterrupt:
        print("\n⚠️  训练被中断")
        raise

    # （保留你原来的末尾评估打印）
    try:
        ic_val, ret_val = alpha_pool.test_ensemble(calc_valid)
        ic_test, ret_test = alpha_pool.test_ensemble(calc_test)
        print(f"[EVAL] valid: IC={ic_val:.4f}, ret={ret_val:.4f} | test: IC={ic_test:.4f}, ret={ret_test:.4f}")
    except Exception as e:
        print("[WARN] 评估阶段出错（可忽略训练已完成）:", e)
    
    # === 训练完成，自动提取因子 ===
    print(f"\n✅ 训练完成！因子池大小: {alpha_pool.size}")
    
    # 保存 alpha_pool 对象，方便后续手动提取
    pool_save_file = os.path.join(save_dir, f"alpha_pool_seed{seed}_final.pkl")
    try:
        with open(pool_save_file, 'wb') as f:
            pickle.dump({
                'alpha_pool': alpha_pool,
                'calc_train': calc_train,
                'calc_valid': calc_valid,
                'calc_test': calc_test,
                'save_dir': save_dir,
                'seed': seed,
                'pool_size': pool
            }, f)
        print(f"💾 因子池对象已保存到: {pool_save_file}")
    except Exception as e:
        print(f"[WARN] 保存因子池对象失败: {e}")
    
    # 自动提取因子
    if alpha_pool.size > 0:
        print(f"\n{'='*80}")
        print(f"开始自动提取因子...")
        print(f"{'='*80}\n")
        try:
            extract_factors(alpha_pool, calc_train, calc_valid, calc_test, save_dir)
            print(f"\n✅ 因子提取完成！")
            print(f"📁 因子文件保存在: {save_dir}/extracted_factors/")
        except Exception as e:
            print(f"\n⚠️  自动提取因子失败: {e}")
            print(f"💡 可以稍后手动运行: python extract_factors_manual.py --pool_file {pool_save_file}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️  因子池为空，跳过因子提取")

def extract_factors(alpha_pool, train_calculator, valid_calculator, test_calculator, save_dir):
    """提取并保存训练生成的因子，并进行质量验证"""
    import os
    import pickle
    from datetime import datetime
    
    print(f"\n{'='*80}")
    print(f"📦 开始提取因子 (因子池大小: {alpha_pool.size})")
    print(f"{'='*80}\n")
    
    if alpha_pool.size == 0:
        print("⚠️  警告: 因子池为空，没有因子可提取")
        return None
    
    # 创建因子保存目录
    factors_dir = os.path.join(save_dir, "extracted_factors")
    os.makedirs(factors_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 先进行因子质量分析
    print("🔍 正在进行因子质量分析...\n")
    analysis_df = analyze_factor_pool(
        alpha_pool, train_calculator, valid_calculator, test_calculator, top_k=min(10, alpha_pool.size)
    )
    
    # 保存分析结果
    if not analysis_df.empty:
        analysis_file = os.path.join(factors_dir, f"factor_analysis_{timestamp}.csv")
        analysis_df.to_csv(analysis_file, index=False, encoding='utf-8-sig')
        print(f"\n📊 因子分析结果已保存到: {analysis_file}\n")
    
    # 提取每个因子的详细信息
    factors_info = []
    
    for i in range(alpha_pool.size):
        try:
            # 获取因子表达式
            expr = alpha_pool.exprs[i] if hasattr(alpha_pool, 'exprs') else None
            weight = alpha_pool.weights[i] if hasattr(alpha_pool, 'weights') else 0.0
            
            if expr is None:
                continue
            
            # 进行详细验证
            validation_result = validate_factor_quality(
                expr, train_calculator, valid_calculator, test_calculator, verbose=False
            )
            
            # 计算因子值
            factor_values = test_calculator._eval_expr(expr)
            
            factor_info = {
                'index': i,
                'weight': float(weight),
                'expression_str': str(expr),
                'factor_values_shape': factor_values.shape,
                'factor_values': factor_values,
                'validation': validation_result
            }
            
            factors_info.append(factor_info)
            
        except Exception as e:
            print(f"⚠️  因子 {i} 提取失败: {e}")
    
    # 保存因子信息（使用pkl格式）
    # 为什么用pkl而不是CSV？
    # 1. 因子值DataFrame很大（时间×合约），pkl可以完整保存DataFrame对象（包括索引、数据类型等）
    # 2. 包含复杂的验证结果字典，CSV无法很好地保存嵌套结构
    # 3. 加载速度快，适合后续程序化处理
    # 注意：因子值矩阵会另外保存为CSV格式（长格式），便于查看和分析
    factors_file = os.path.join(factors_dir, f"factors_{timestamp}.pkl")
    with open(factors_file, 'wb') as f:
        pickle.dump(factors_info, f)
    
    print(f"\n✅ 成功提取 {len(factors_info)} 个因子")
    print(f"📁 因子详细信息已保存到: {factors_file} (pkl格式，包含完整DataFrame和验证结果)")
    print(f"💡 提示: 使用以下代码加载因子信息：")
    print(f"   import pickle")
    print(f"   with open('{factors_file}', 'rb') as f:")
    print(f"       factors_info = pickle.load(f)")
    
    # 保存因子值矩阵（CSV格式，便于查看）
    if factors_info:
        factor_matrix_file = os.path.join(factors_dir, f"factor_matrix_{timestamp}.csv")
        try:
            # 合并所有因子值
            # 注意：每个因子的factor_values是 (时间, 合约) 的DataFrame
            # 我们需要将每个因子转换为长格式，然后合并
            all_factors_long = []
            for info in factors_info:
                factor_df = info['factor_values'].copy()
                # 重置索引，将时间作为列
                factor_df = factor_df.reset_index()
                # 转换为长格式：date, symbol, factor_value
                factor_long = factor_df.melt(
                    id_vars=[factor_df.columns[0]],  # 第一列是时间索引
                    var_name='symbol',
                    value_name=f"factor_{info['index']}"
                )
                factor_long.rename(columns={factor_df.columns[0]: 'date'}, inplace=True)
                all_factors_long.append(factor_long)
            
            if all_factors_long:
                # 按 date 和 symbol 合并所有因子
                combined_factors = all_factors_long[0]
                for factor_long in all_factors_long[1:]:
                    combined_factors = pd.merge(
                        combined_factors, 
                        factor_long, 
                        on=['date', 'symbol'], 
                        how='outer'
                    )
                
                combined_factors.to_csv(factor_matrix_file, index=False)
                print(f"📊 因子值矩阵已保存到: {factor_matrix_file}")
                print(f"   格式: 长格式 (date, symbol, factor_0, factor_1, ...)")
        except Exception as e:
            print(f"⚠️  保存因子值矩阵失败: {e}")
            import traceback
            traceback.print_exc()
    
    return factors_info, analysis_df


def validate_factor_quality(
    expr,
    train_calculator,
    valid_calculator,
    test_calculator,
    verbose=True
):
    """
    全面验证因子质量，包括：
    1. IC 和 RankIC 分析（训练集、验证集、测试集）
    2. 多空收益分析
    3. 因子分布和统计特征
    4. 因子稳定性（跨数据集一致性）
    5. 因子可解释性
    
    返回: dict 包含所有验证指标
    """
    import numpy as np
    import pandas as pd
    
    results = {
        'expression': str(expr),
        'train': {},
        'valid': {},
        'test': {},
        'stability': {},
        'statistics': {}
    }
    
    # 1. 计算各数据集的 IC 和 RankIC
    for name, calc in [('train', train_calculator), ('valid', valid_calculator), ('test', test_calculator)]:
        try:
            # IC 和 RankIC
            ic, rank_ic = calc.calc_single_all_ret(expr)
            ic_ls, ls = calc.calc_single_IC_ret_with_ls(expr)
            
            # 计算因子值
            factor_values = calc._eval_expr(expr)
            target = calc.get_target()
            
            # 对齐数据
            factor_values, target = factor_values.align(target, join='inner', axis=0)
            
            # 因子统计
            factor_flat = factor_values.values.flatten()
            factor_flat = factor_flat[~np.isnan(factor_flat)]
            
            results[name] = {
                'ic': float(ic) if np.isfinite(ic) else np.nan,
                'rank_ic': float(rank_ic) if np.isfinite(rank_ic) else np.nan,
                'long_short_return': float(ls) if np.isfinite(ls) else np.nan,
                'mean': float(np.nanmean(factor_flat)) if len(factor_flat) > 0 else np.nan,
                'std': float(np.nanstd(factor_flat)) if len(factor_flat) > 0 else np.nan,
                'min': float(np.nanmin(factor_flat)) if len(factor_flat) > 0 else np.nan,
                'max': float(np.nanmax(factor_flat)) if len(factor_flat) > 0 else np.nan,
                'nan_ratio': float(np.isnan(factor_values.values).sum() / factor_values.size),
                'valid_points': int((~np.isnan(factor_values.values)).sum())
            }
        except Exception as e:
            if verbose:
                print(f"⚠️  计算 {name} 集指标失败: {e}")
            results[name] = {'error': str(e)}
    
    # 2. 因子稳定性分析
    try:
        train_ic = results['train'].get('ic', np.nan)
        valid_ic = results['valid'].get('ic', np.nan)
        test_ic = results['test'].get('ic', np.nan)
        
        # IC 衰减（训练集 -> 验证集 -> 测试集）
        ic_decay_train_valid = train_ic - valid_ic if np.isfinite(train_ic) and np.isfinite(valid_ic) else np.nan
        ic_decay_valid_test = valid_ic - test_ic if np.isfinite(valid_ic) and np.isfinite(test_ic) else np.nan
        ic_decay_train_test = train_ic - test_ic if np.isfinite(train_ic) and np.isfinite(test_ic) else np.nan
        
        # IC 符号一致性
        ic_sign_consistent = (
            np.sign(train_ic) == np.sign(valid_ic) == np.sign(test_ic)
            if all(np.isfinite([train_ic, valid_ic, test_ic])) else False
        )
        
        results['stability'] = {
            'ic_decay_train_valid': float(ic_decay_train_valid) if np.isfinite(ic_decay_train_valid) else np.nan,
            'ic_decay_valid_test': float(ic_decay_valid_test) if np.isfinite(ic_decay_valid_test) else np.nan,
            'ic_decay_train_test': float(ic_decay_train_test) if np.isfinite(ic_decay_train_test) else np.nan,
            'ic_sign_consistent': bool(ic_sign_consistent),
            'ic_std': float(np.nanstd([train_ic, valid_ic, test_ic])) if all(np.isfinite([train_ic, valid_ic, test_ic])) else np.nan
        }
    except Exception as e:
        if verbose:
            print(f"⚠️  稳定性分析失败: {e}")
    
    # 3. 因子质量评分
    try:
        # 基础评分（0-100）
        score = 0
        
        # IC 绝对值评分（40分）
        avg_ic = np.nanmean([abs(results['train'].get('ic', 0)), 
                             abs(results['valid'].get('ic', 0)), 
                             abs(results['test'].get('ic', 0))])
        if avg_ic > 0.05:
            score += 40
        elif avg_ic > 0.03:
            score += 30
        elif avg_ic > 0.01:
            score += 20
        elif avg_ic > 0:
            score += 10
        
        # 稳定性评分（30分）
        if results['stability'].get('ic_sign_consistent', False):
            score += 15
        decay = abs(results['stability'].get('ic_decay_train_test', 1))
        if decay < 0.01:
            score += 15
        elif decay < 0.03:
            score += 10
        elif decay < 0.05:
            score += 5
        
        # 多空收益评分（20分）
        avg_ls = np.nanmean([results['train'].get('long_short_return', 0),
                             results['valid'].get('long_short_return', 0),
                             results['test'].get('long_short_return', 0)])
        if avg_ls > 0.1:
            score += 20
        elif avg_ls > 0.05:
            score += 15
        elif avg_ls > 0:
            score += 10
        
        # 数据质量评分（10分）
        nan_ratio = np.nanmean([results['train'].get('nan_ratio', 1),
                                results['valid'].get('nan_ratio', 1),
                                results['test'].get('nan_ratio', 1)])
        if nan_ratio < 0.01:
            score += 10
        elif nan_ratio < 0.05:
            score += 7
        elif nan_ratio < 0.1:
            score += 5
        
        results['quality_score'] = float(score)
        results['quality_level'] = (
            '优秀' if score >= 80 else
            '良好' if score >= 60 else
            '一般' if score >= 40 else
            '较差' if score >= 20 else
            '很差'
        )
    except Exception as e:
        if verbose:
            print(f"⚠️  质量评分失败: {e}")
    
    # 4. 打印结果
    if verbose:
        print("\n" + "="*80)
        print(f"因子验证报告: {str(expr)[:60]}...")
        print("="*80)
        
        print("\n📊 IC 和 RankIC 分析:")
        for name in ['train', 'valid', 'test']:
            data = results[name]
            if 'error' not in data:
                print(f"  {name.upper():6s}: IC={data.get('ic', np.nan):7.4f}, "
                      f"RankIC={data.get('rank_ic', np.nan):7.4f}, "
                      f"LS={data.get('long_short_return', np.nan):7.4f}")
        
        print("\n📈 因子稳定性:")
        stability = results['stability']
        print(f"  IC 衰减 (训练->验证): {stability.get('ic_decay_train_valid', np.nan):7.4f}")
        print(f"  IC 衰减 (验证->测试): {stability.get('ic_decay_valid_test', np.nan):7.4f}")
        print(f"  IC 衰减 (训练->测试): {stability.get('ic_decay_train_test', np.nan):7.4f}")
        print(f"  IC 符号一致性: {'✅' if stability.get('ic_sign_consistent', False) else '❌'}")
        
        print("\n📉 因子统计特征:")
        for name in ['train', 'valid', 'test']:
            data = results[name]
            if 'error' not in data:
                print(f"  {name.upper():6s}: mean={data.get('mean', np.nan):8.4f}, "
                      f"std={data.get('std', np.nan):8.4f}, "
                      f"nan_ratio={data.get('nan_ratio', np.nan):6.2%}")
        
        if 'quality_score' in results:
            print(f"\n⭐ 因子质量评分: {results['quality_score']:.1f}/100 ({results['quality_level']})")
        
        print("="*80 + "\n")
    
    return results


def analyze_factor_pool(
    alpha_pool,
    train_calculator,
    valid_calculator,
    test_calculator,
    top_k=10
):
    """
    分析因子池中所有因子的质量，并返回排名
    
    返回: DataFrame 包含所有因子的验证结果
    """
    import pandas as pd
    
    if alpha_pool.size == 0:
        print("⚠️  因子池为空")
        return pd.DataFrame()
    
    print(f"\n🔍 开始分析因子池中的 {alpha_pool.size} 个因子...\n")
    
    all_results = []
    
    for i in range(alpha_pool.size):
        expr = alpha_pool.exprs[i]
        weight = alpha_pool.weights[i]
        
        if expr is None:
            continue
        
        print(f"分析因子 {i+1}/{alpha_pool.size}...", end='\r')
        
        try:
            result = validate_factor_quality(
                expr, train_calculator, valid_calculator, test_calculator, verbose=False
            )
            result['index'] = i
            result['weight'] = float(weight)
            all_results.append(result)
        except Exception as e:
            print(f"\n⚠️  因子 {i} 分析失败: {e}")
    
    if not all_results:
        print("\n⚠️  没有成功分析的因子")
        return pd.DataFrame()
    
    # 转换为 DataFrame
    df_data = []
    for r in all_results:
        row = {
            'index': r['index'],
            'weight': r['weight'],
            'expression': r['expression'],
            'train_ic': r['train'].get('ic', np.nan),
            'valid_ic': r['valid'].get('ic', np.nan),
            'test_ic': r['test'].get('ic', np.nan),
            'train_rank_ic': r['train'].get('rank_ic', np.nan),
            'valid_rank_ic': r['valid'].get('rank_ic', np.nan),
            'test_rank_ic': r['test'].get('rank_ic', np.nan),
            'train_ls': r['train'].get('long_short_return', np.nan),
            'valid_ls': r['valid'].get('long_short_return', np.nan),
            'test_ls': r['test'].get('long_short_return', np.nan),
            'ic_decay_train_test': r['stability'].get('ic_decay_train_test', np.nan),
            'ic_sign_consistent': r['stability'].get('ic_sign_consistent', False),
            'quality_score': r.get('quality_score', 0),
            'quality_level': r.get('quality_level', '未知')
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # 按质量评分排序
    df = df.sort_values('quality_score', ascending=False)
    
    print(f"\n✅ 完成分析，共 {len(df)} 个有效因子\n")
    
    # 显示 Top K
    print(f"🏆 Top {min(top_k, len(df))} 因子:")
    print("="*120)
    for idx, row in df.head(top_k).iterrows():
        print(f"\n排名 {idx+1}: 因子 #{int(row['index'])} (质量评分: {row['quality_score']:.1f}/100 - {row['quality_level']})")
        print(f"  权重: {row['weight']:.6f}")
        print(f"  IC: 训练={row['train_ic']:7.4f}, 验证={row['valid_ic']:7.4f}, 测试={row['test_ic']:7.4f}")
        print(f"  RankIC: 训练={row['train_rank_ic']:7.4f}, 验证={row['valid_rank_ic']:7.4f}, 测试={row['test_rank_ic']:7.4f}")
        print(f"  多空收益: 训练={row['train_ls']:7.4f}, 验证={row['valid_ls']:7.4f}, 测试={row['test_ls']:7.4f}")
        print(f"  IC 衰减: {row['ic_decay_train_test']:7.4f}, 符号一致性: {'✅' if row['ic_sign_consistent'] else '❌'}")
        print(f"  表达式: {row['expression'][:100]}...")
        print("-"*120)
    
    return df

if __name__ == "__main__":
    for s in [5, 310, 24, 10, 10086]:
        main(seed=s, pool=20, steps=60_000)