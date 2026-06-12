# Dancher-Tools

基于 PyTorch 的轻量级深度学习训练框架。提供 `compile → fit → evaluate` 工作流、DDP 多卡训练、AMP 混合精度、tmux 会话管理和 checkpoint 管理。

## 快速开始

```bash
git clone https://github.com/DancherLab/dancher_tools.git
pip install -r dancher_tools/requirements.txt
```

### 最小示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dancher_tools import Core

model = nn.Linear(10, 2)
wrapper = Core()
wrapper.model_name = "my_model"

# 同时包装模型（通过子类或直接赋值）
class MyModel(Core):
    def __init__(self):
        super().__init__()
        self.model_name = "my_model"
        self.net = nn.Linear(10, 2)

    def forward(self, x):
        return self.net(x)

model = MyModel()
model.compile(
    criterion=F.mse_loss,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    amp=True,  # 可选：启用混合精度
)

model.fit(
    train_data=train_dataset,
    val_data=val_dataset,
    num_epochs=500,
    batch_size=4,
    model_save_dir="./checkpoints",
)
```

## API 参考

### Core 类

所有模型的核心基类，继承 `nn.Module`。

#### `compile(criterion, optimizer=None, scheduler=None, metrics=None, loss_weights=None, amp=False)`

配置训练参数。

| 参数 | 类型 | 说明 |
|------|------|------|
| `criterion` | callable 或 list | 损失函数，多个时自动组合为 `CombinedLoss` |
| `optimizer` | Optimizer | 优化器 |
| `scheduler` | LRScheduler | 学习率调度器 |
| `metrics` | dict[str, callable] | 评估指标 `{name: fn(pred, target) → float}` |
| `loss_weights` | list[float] | 多损失函数的权重 |
| `amp` | bool | 启用混合精度训练 (float16 autocast + GradScaler) |

#### `fit(train_data, val_data, num_epochs, batch_size, model_save_dir, ...)`

训练模型。

| 参数 | 说明 |
|------|------|
| `train_data` | Dataset、DataLoader 或 DataModule |
| `val_data` | 验证集 |
| `num_epochs` | 训练轮数 |
| `batch_size` | 批大小 |
| `model_save_dir` | 检查点保存目录 |
| `patience` | 早停耐心值 (default: 15) |
| `delta` | 早停最小改善量 (default: 0.01) |
| `grad_clip` | 梯度裁剪 (default: 1.0) |

#### `evaluate(data_loader, verbose=False) → (loss, metrics)`

评估模型，返回平均损失和指标。

#### `save(model_dir, mode="best")` / `load(model_dir, mode="best", specified_path=None)`

保存/加载检查点（包含 model、optimizer、scheduler、scaler 状态）。

#### `transfer(specified_path, strict=False)`

迁移学习：加载兼容的权重，跳过不匹配的层。

#### 可覆盖的 Hook

```python
class MyModel(Core):
    def _training_step(self, batch):
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self._forward(inputs)
        return self.criterion(outputs, targets)

    def _eval_step(self, batch):
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        metrics = {name: fn(outputs, targets) for name, fn in self.metrics.items()}
        return loss, metrics
```

### SRWrapper

通用的模型包装器，自动处理 eval 时的反归一化：

```python
from dancher_tools import SRWrapper

wrapper = SRWrapper(
    model=my_model,
    model_name="sr_model",
    target_mean=10.5,
    target_std=2.3,
)
wrapper.compile(criterion=F.mse_loss, optimizer=optimizer)
wrapper.fit(train_data=train_ds, val_data=val_ds, ...)
```

`EvalMixin` 在 eval 时自动将输出反归一化到物理单位后计算指标。可通过 `eval_criterion` 属性自定义 eval 损失函数（默认 L1）。

### MmapArrayDataset

内存映射的数据集，避免将整个数组加载到 RAM：

```python
from dancher_tools import MmapArrayDataset

train_ds = MmapArrayDataset(
    input_path="train_input.npy",
    target_path="train_target.npy",
    normalize_target=True,
    add_channel_dim=True,  # 自动添加通道维 (1, ...)
)
stats = train_ds.get_stats()

test_ds = MmapArrayDataset(
    input_path="test_input.npy",
    target_path="test_target.npy",
    input_stats=stats["input"],
    normalize_target=False,
)
```

### DataModule

自定义数据加载：

```python
from dancher_tools import DataModule

class MyData(DataModule):
    def setup(self):
        self.train_ds = ...
        self.val_ds = ...

dm = MyData(batch_size=4, num_workers=2)
dm.setup()
```

### 工具函数

```python
from dancher_tools import CombinedLoss, EarlyStopping, is_ddp, ddp_info, ddp_cleanup
```

- `CombinedLoss(losses, weights)` — 加权组合多个损失函数
- `EarlyStopping(patience, delta)` — 早停策略
- `is_ddp()` — 检测是否在 DDP 环境中
- `ddp_info()` — 获取 rank/world_size/local_rank
- `ddp_cleanup()` — 清理 DDP 进程组

## DDP 多卡训练

```bash
# 单卡
python train.py --model my_model

# 多卡（通过 torchrun）
torchrun --nproc_per_node=4 train.py --model my_model
```

框架自动检测 DDP 环境（`RANK` 环境变量），处理：
- 进程组初始化（NCCL backend）
- `DistributedDataParallel` 包装
- `DistributedSampler` 数据分片
- 跨 rank 指标同步

## Shell 脚本

框架提供通用的服务器训练基础设施：

```
scripts/
├── _common.sh       # 通用 env/conda/tmux/torchrun 基础设施
├── _parse_yaml.sh   # 零依赖 YAML 解析器
└── run.sh           # 入口模板
```

### 使用方法

**1. 创建 YAML 配置文件**

```yaml
# configs/my_experiment.yaml
model: my_model
batch_size: 4
epochs: 500
lr: 1e-3
gpu_ids: "0"
```

**2. 创建项目的 `_common.sh`**

```bash
#!/bin/bash
# scripts/_common.sh — 定义命令构建函数

_build_train_cmd() {
    _CMD=(python train.py --model "${MODEL}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}")
    _wrap_torchrun  # 自动处理多卡 torchrun
}

_build_eval_cmd() {
    _CMD=(python evaluate.py --model "${MODEL}")
    export _CUDA="${GPU_IDS:-0}"
}

# 加载框架基础设施
source "$(dirname "${BASH_SOURCE[0]}")/../dancher_tools/scripts/_common.sh"
```

**3. 创建 `run.sh`**

```bash
#!/bin/bash
CONFIG=my_experiment
MODE=train
GPU_IDS=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -euo pipefail

# ... CLI arg parsing ...

# 加载 YAML 配置
source "${SCRIPT_DIR}/../dancher_tools/scripts/_parse_yaml.sh"
_parse_yaml "${PROJECT_ROOT}/configs/${CONFIG}.yaml"

# 映射 YAML key → shell 变量
MODEL="${model}"
EPOCHS="${epochs:-500}"
BATCH_SIZE="${batch_size:-4}"

source "${SCRIPT_DIR}/_common.sh"
```

**4. 运行**

```bash
bash scripts/run.sh my_experiment train
```

### 特性

- **自动 conda 激活**：三层 fallback（hook → conda.sh → PATH）
- **tmux 持久会话**：detach 后训练继续，idle 会话自动重启
- **torchrun 自动编排**：`GPU_IDS=0,1,2,3` 自动转为多卡训练
- **`.env` 加载**：从项目父目录读取，`CONDA_PATH` 必填
- **安全引号处理**：JSON 参数等特殊字符不会破坏命令

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `CONDA_PATH` | 是 | conda 安装路径 |
| `CONDA_ENV` | 否 | conda 环境名 (default: base) |
| `NO_TMUX` | 否 | 设为任意值禁用 tmux |

## 项目结构

```
my_project/
├── dancher_tools/         # 框架代码
├── configs/               # YAML 配置文件
│   └── my_experiment.yaml
├── scripts/               # 训练脚本
│   ├── run.sh             # 入口
│   └── _common.sh         # 命令构建
├── models/                # 自定义模型
├── train.py               # 训练脚本
└── evaluate.py            # 评估脚本
```
