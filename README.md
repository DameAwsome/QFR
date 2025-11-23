# QFR - Quantitative Factor Research with Reinforcement Learning

基于强化学习的量化因子研究项目，用于加密货币交易。

## 项目结构

```
FQR+TRLSinCrypto/
├── CPT_train_maskable_ppo_reward_shaping.py  # 主训练脚本
├── alphagen/                                  # 因子生成模块
│   ├── data/                                  # 数据表达式和计算器
│   ├── models/                                # 因子池模型
│   ├── rl/                                    # 强化学习环境
│   └── utils/                                 # 工具函数
├── alphagen_qlib/                             # QLib适配器
├── sb3cr_contrib/                             # Stable-Baselines3贡献模块
│   ├── common/                                # 通用工具
│   └── ppo_reward_shaping/                    # PPO奖励塑形算法
└── checkpoints/                               # 训练检查点和因子提取结果
    └── ppo_reward_shaping_crypto/
        ├── alpha_pool_*.pkl                   # 因子池对象（Git LFS）
        └── extracted_factors/                 # 提取的因子
            ├── factor_analysis_*.csv          # 因子分析报告
            ├── factor_matrix_*.csv            # 因子值矩阵（Git LFS）
            └── factors_*.pkl                  # 因子详细信息（Git LFS）
```

## 主要功能

1. **强化学习训练**: 使用PPO算法训练因子生成模型
2. **因子池管理**: 自动管理和优化因子池
3. **因子提取**: 训练完成后自动提取和验证因子
4. **因子分析**: 生成详细的因子质量分析报告

## 使用方法

### 训练模型

```bash
cd FQR+TRLSinCrypto
python CPT_train_maskable_ppo_reward_shaping.py
```

### 提取因子

```bash
python extract_factors_manual.py --auto_find
```

### 查看因子表格

```bash
python view_factor_table.py
```

## 依赖

- Python 3.10+
- PyTorch
- Stable-Baselines3
- Pandas
- NumPy

## Git LFS

大文件（.pkl, .csv）使用 Git LFS 跟踪。确保已安装 Git LFS：

```bash
git lfs install
```

## 许可证

[添加许可证信息]

