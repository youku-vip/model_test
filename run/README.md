# Isaac 0.5 双臂 EEF —— 微调与部署（repo/run）

`repo/run/` 下的脚本，基于仓库现有 docker/ 工作流（`debug_bimanual.sh` /
`train_bimanual_8gpu.sh` / `patch_package_for_bimanual.py`）整理成两个可直接运行的
脚本，专门适配 **LeRobot v3.0** 双臂数据集（`meta/info.json` 中
`codebase_version: "v3.0"`、`data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
嵌套布局、3 路相机 head/left/right、34D state / 20D action / 10fps）。

```
repo/run/
├── finetune_isaac05.py   # 流匹配(Flow Matching)微调：校验数据 -> patch 包 -> 多卡训练（支持续训）
├── deploy_isaac.py       # WebSocket 部署服务（与 docker/serve_isaac.py 同协议）
├── train_a100.sh         # A100-80GB 编排：4 卡冒烟测试 -> 8 卡正式训练（容器内一键）
├── probe_env.py          # 容器内环境探测（只读）：GPU/版本/仓库/模型/数据集/词表/补丁状态
├── viz_server.py         # 离线训练可视化（默认端口 8600，纯 stdlib，无外网可用）
├── wandb_server.sh       # 本地 wandb 面板（默认 8600，需 docker；有外网时用）
└── README.md
```

运行环境：**生产镜像容器内**（`isaac:gpu`，`PRODUCTION_TORCH=1`：torch 2.10+cu128 /
transformers 5.5.4）。你拿到的是**已经跑起来的容器**（训练代码/数据集/模型都已挂载），
所有操作与修改都只能在容器内进行；不需要、也无法在宿主执行 docker run。

### 已有容器：直接探测路径 → 训练

```bash
# 1) 一次性体检：确认容器内真实路径（GPU/运行时版本/仓库/模型/数据集/词表/补丁状态）
python /code/run/probe_env.py

# 2) 一键编排：4×A100-80GB 冒烟(20步) -> 8×A100-80GB 正式训练(20000步)
#    数据集/基础包自动探测；路径不符时用环境变量覆盖（见脚本头部）
bash /code/run/train_a100.sh
```

- 容器内用户 `user_lerobot`（uid 1000）与宿主 `zoomlion`（uid 1000）一致，常规
  挂载可直接写；若提示不可写，容器内执行 `sudo chown -R
  user_lerobot:user_lerobot /code /model` 或改用可写挂载。
- 基础包/词表/数据集默认自动探测（`/data/isaac/model/lerobot_policy`、
  `/model/lerobot_policy`、`/checkpoint/lerobot_policy`；`/vocab/vocab.json`；
  `/data/local/bimanual`、`/data/ds` 等），也可用 `--base-package` / `--dataset-dir`
  / `--vocab` 显式指定。
- 如果 `/code/run` 不存在（挂载的不是完整仓库），先确认仓库挂载点，把脚本路径换掉
  或用 probe 输出的路径运行。
- 训练/部署/续训全部用 `/code/run/` 下脚本，无需在宿主做任何事。

> 参考：如果确实还没有容器，才需要从宿主 `docker run` 启动（挂载规范见
> `docker/README.md` 与 `train_a100.sh` 末尾注释）。

---

## 1. 微调（Flow Matching，action-expert-only）

Isaac-0.5 是 36B 稀疏 null-MoE VLA：全量训练需要 FP32 参数存储，而 grouped_mm
专家调度要求 BF16，二者冲突（`modeling_perceptron_isaac.py` 的训练闸门会直接
raise）。因此微调采用 **冻结 VLM+vision、只训练 MolmoAct 动作专家头** 的
`train_expert_only=true` 方式，专家输出按 `loss_plan=text_ntp_fast_flow_action`
走 **流匹配（flow matching）** 损失 —— 这正是 `docker/patches/
isaac05-expert-only-training.patch` 放开的路子。

```bash
# 0) 一次性准备（若还没做）：lerobot 源码打 expert-only 训练补丁
#    （脚本会在训练前自动检测并应用，也可手动执行）
git -C /code/lerobot apply /code/docker/patches/isaac05-expert-only-training.patch

# 1) 冒烟验证：20 步短跑（等价 debug_bimanual.sh）
QWEN35_VOCAB_PATH=/vocab/vocab.json python /code/run/finetune_isaac05.py \
    --dataset-dir /data/local/bimanual --smoke

# 2) 正式训练：8×H100，20000 步，有效 batch = 1×4×8 = 32
QWEN35_VOCAB_PATH=/vocab/vocab.json python /code/run/finetune_isaac05.py \
    --dataset-dir /data/local/bimanual \
    --output-dir /data/outputs/isaac-finetune \
    --steps 20000 --save-freq 2500 --gpus 0,1,2,3,4,5,6,7
```

脚本行为（均可重跑、幂等）：

| 步骤 | 说明 |
| --- | --- |
| 数据集校验 | `meta/info.json` 必须 v3.0，`observation.state`=34D、`action`=20D、三相机、fps 一致、`meta/tasks.parquet` 存在 |
| 统计量 | 从 `data/chunk-*/file-*.parquet` 读全量帧，逐维算 q01/q99 → 写 patched 包 `isaac_stats.json`；若数据集缺 `meta/stats.json` 一并补齐（训练必须） |
| 包 patch | 复制基础包为可写副本，patch `config.json` **和** `policy_preprocessor.json` / `policy_postprocessor.json`（pack step 的相机/尺寸/维度/fps 是渲染权威，只改 config.json 会崩） |
| 源码补丁 | 检测 `_require_native_training_supported` 是否带 `train_expert_only` 豁免，缺失则 `git apply` |
| 训练 | `accelerate launch --num_processes=<GPU数> --mixed_precision=bf16 -m lerobot.scripts.lerobot_train --policy.path=<patched包> --policy.train_expert_only=true ...` |

常用参数：`--dataset-dir`（必填）、`--output-dir`、`--gpus`、`--steps`、`--save-freq`、
`--grad-accum`、`--action-dim 20`、`--proprio-dim 34`、`--cameras head,left,right`、
`--image-size 240,424`、`--fps 10`、`--base-package /data/isaac/model/lerobot_policy`、
`--patch-dir`（默认 `<模型目录>/lerobot_policy_patched`，必须与模型目录同级，否则
`hf_model_path=".."` 便携布局无法解析）。

> 注意：`--patch-dir` 默认是模型目录下的 `lerobot_policy_patched`（例如
> `/data/isaac/model/lerobot_policy_patched`），这是便携 isaac_0_5 包布局的要求。

训练产物：`<output>/checkpoints/last/pretrained_model`（含 config.json、
isaac_stats.json、processor 序列化状态、微调后的专家权重），可直接喂给
`PerceptronIsaacPolicy.from_pretrained`。

> 数据量提示：动作目标按 `chunk_size`（50）帧采样，短于 50 帧的 episode 会被
> 采样器丢弃（脚本会打印提示）。

### RTC（实时分块）训练/推理

基础模型的动作专家本就带 RTC（模型 config 里 `action_expert.rtc_max_delay_steps=12`、
`rtc_probability=0.5`）。微调时传回同样参数即可延续 RTC：

```bash
python .../finetune_isaac05.py --dataset-dir <数据集> --gpus 0,1 --smoke \
    --allow-unqualified-device --n-obs-steps 1 \
    --rtc-max-delay-steps 12 --rtc-probability 0.5 \
    --rtc-prefix-length 4          # 推理时每次携带 4 行已执行动作前缀（可选）
```

- 训练：`flow_rtc_max_delay_steps>0` 且 `rtc_probability>0` 时，`_flow_loss` 对每个训练
  chunk 随机采样一个已执行前缀长度（`--rtc-delay-sampling`：uniform/poisson/exponential），
  前缀行钉在 clean 目标、只对后缀行做 flow 回归；脚本自动给模型侧
  `DiTActionExpertHead.forward` 打 RTC 前缀补丁（幂等）。
- 推理：`--rtc-prefix-length N` 写进 `rtc_prefix_length`，`select_action` 会用上一
  chunk 的延续行做前缀；`deploy_isaac.py` 请求可带可选 `action_prefix`（[P, action_dim] 原始动作）
  和 `prefix_length` 实现 RTC 闭环调用。
- 注意：全局 flow 分母（`get_loss_denominators`）在 forward 前计算、不含 RTC 采样，
  对 flow loss 是一个近似的常数缩放（不改变梯度方向）。

### 训练可视化（端口 8600；支持无外网）

生产环境**没有外网**时，云端 wandb 和 `wandb server`（首次需拉后端镜像）都不可用。
推荐用**纯离线可视化服务** `viz_server.py`，它直接解析训练日志（`<output>/train.log`，
由微调脚本自动 tee 生成），只依赖 Python 标准库：

```bash
# 另开终端：离线可视化面板（默认 8600）
python /code/run/viz_server.py --log-dir /data/algorithm/repo/outputs/isaac-finetune/smoke --port 8600
# 说明：训练日志在 <output>.train.log（如 .../smoke.train.log），viz_server 会自动找到

# 训练（日志自动写 <output>/train.log，面板实时更新）
python /code/run/finetune_isaac05.py --dataset-dir <数据集> --gpus 0,1 --smoke \
    --allow-unqualified-device
```

浏览器打开 `http://<host>:8600` 实时看 `loss/flow_loss`、`loss/text_loss`、`lr`、
`grad_norm` 等曲线。数据接口：`GET /metrics`（JSON）、`GET /log`（最近日志）。

**wandb**（有外网时可选）：`--wandb` 系列参数已内置（`--wandb-project`、
`--wandb-entity`、`--wandb-run-id`、`--wandb-mode`、`--wandb-host`、`--wandb-api-key`）。
无外网时 `--wandb` 默认走 `offline`，数据写 `<output>/wandb`，之后有网再
`wandb sync <output>/wandb` 上传；本地面板 `bash /code/run/wandb_server.sh 8600`
需 docker 且要能拉到后端镜像。

> 8600 端口双用途：`deploy_isaac.py` 部署服务默认端口也是 8600。若两者同机同端口，
> 错开：可视化用 `--port 8601`，或部署服务用 `--port 8601`。

### A100-80GB：4 卡冒烟测试 + 8 卡正式训练

便携模型远程代码（`modeling_isaac05.py::require_qualified_runtime`）默认只允许
SM90/H100 加载（transformers 5.5.4 / CUDA 12.8 / H100）。A100(SM80) 训练需加
`--allow-unqualified-device`：脚本会向模型目录写入一个幂等豁免（环境开关
`ISAAC_ALLOW_UNQUALIFIED_DEVICE=1` 时跳过 capability/设备名检查；transformers 与
CUDA 版本检查保留，因此仍须使用生产镜像 `PRODUCTION_TORCH=1`）。该开关只影响
训练时的模型加载，生产推理仍默认走 H100 闸门。

```bash
# 一键编排：环境自检 -> 4×A100-80GB 冒烟(20步) -> 8×A100-80GB 正式训练(20000步)
# 可调环境变量：DATASET_DIR / OUTPUT_DIR / SMOKE_GPUS / TRAIN_GPUS /
#               TRAIN_STEPS / SAVE_FREQ / GRAD_ACC / VOCAB / BASE_PACKAGE
bash /code/run/train_a100.sh
```

等价于分步执行：

```bash
# 4×A100-80GB 冒烟测试（20 步，loss 正常/无 NaN/显存不爆）
python /code/run/finetune_isaac05.py --dataset-dir /data/local/bimanual \
    --gpus 0,1,2,3 --smoke --allow-unqualified-device

# 8×A100-80GB 正式训练（20000 步，有效 batch = 1×4×8 = 32）
python /code/run/finetune_isaac05.py --dataset-dir /data/local/bimanual \
    --gpus 0,1,2,3,4,5,6,7 --steps 20000 --save-freq 2500 \
    --grad-accum 4 --allow-unqualified-device

# 续训（从上次检查点继续；--steps 可选覆盖目标步数）
python /code/run/finetune_isaac05.py --dataset-dir /data/local/bimanual \
    --output-dir /data/outputs/isaac-finetune --gpus 0,1,2,3,4,5,6,7 \
    --resume --steps 30000 --allow-unqualified-device
```

## 2. 离线评测（可选）

```bash
python /code/docker/eval_offline.py \
    --policy-path /data/outputs/isaac-finetune/checkpoints/last/pretrained_model \
    --dataset-repo-id local/bimanual --dataset-root /data/local/bimanual \
    --n-frames 100 --episodes 3
```

## 3. 部署（WebSocket 服务）

`deploy_isaac.py` 与 `docker/serve_isaac.py` **同一协议**（msgpack-numpy），输入
输出完全对齐训练数据：

```jsonc
// 请求
{
  "instruction": "task in english",
  "state":        [34] float32,            // 与训练 observation.state 一致
  "images":       { "head": HxWx3 uint8, "left": HxWx3 uint8, "right": HxWx3 uint8 }
}
// 响应
{ "status": "success"|"error", "action": [chunk_size, 20] float32(原始绝对动作),
  "latency": ms, "message": "错误信息(失败时)" }
```

```bash
# 启动前自检（加载 + 预热 + 合成请求推理，不监听端口）
QWEN35_VOCAB_PATH=/vocab/vocab.json python /code/run/deploy_isaac.py \
    --policy-path /data/outputs/isaac-finetune/checkpoints/last/pretrained_model \
    --self-test

# 启动服务
QWEN35_VOCAB_PATH=/vocab/vocab.json python /code/run/deploy_isaac.py \
    --policy-path /data/outputs/isaac-finetune/checkpoints/last/pretrained_model \
    --host 0.0.0.0 --port 8600

# 另开终端，客户端连通性测试
python /code/run/deploy_isaac.py --client --host 127.0.0.1 --port 8600
```

动作语义（与训练一致，`action_representation=absolute`）：每臂 10D =
`[x, y, z, 6D旋转(旋转矩阵前两列), gripper]`，左臂在前、右臂在后；响应中的
`action[0:n_action_steps]` 是可直接执行的前瞻动作，其余为未来预测。默认
`--stateless` 每请求独立推理（无历史泄漏），闭环连续调用可设 `--stateless=false`。

## 4. 常见问题

- **`Isaac-0.5 training is not supported yet`**：lerobot 源码未打 expert-only
  补丁，运行训练脚本会自动应用；手动：`git -C /code/lerobot apply
  /code/docker/patches/isaac05-expert-only-training.patch`。
- **probe 显示「运行时 lerobot 是镜像快照，不是挂载仓库」**：容器内
  `import lerobot` 实际加载了镜像内置快照（如 `/lerobot/src`），而自动补丁打在
  挂载仓库上，补丁不生效。修复（任选其一）：
  ```bash
  export PYTHONPATH=/data/algorithm/repo/lerobot/src:$PYTHONPATH   # 让挂载仓库优先
  # 或直接交给训练脚本：它会在子进程 PYTHONPATH 前置挂载仓库，并把补丁也兜底打到运行时源码
  ```
- **probe 找不到模型/数据集/词表**：容器挂载路径与默认候选不同，容器内先定位再验证：
  ```bash
  find / -maxdepth 6 -type d -name lerobot_policy 2>/dev/null          # 找导入包
  find / -maxdepth 6 -name info.json -path '*meta*' 2>/dev/null | head # 找数据集
  find / -name vocab.json 2>/dev/null | head                           # 找词表
  python /data/algorithm/repo/run/probe_env.py \
      --base-package <模型包路径> --dataset-dir <数据集路径> --vocab <词表路径>
  ```
- **GPU 数量与计划不符（probe 只显示 2 卡）**：`nvidia-smi -L` 确认容器实际可见
  GPU 数；只有 2 卡就 `--gpus 0,1`（smoke 与正式训练都按实际卡数调）。
- **torch 2.11.0+cu128 是否可用**：可用。主闸门只查 `torch.version.cuda==12.8`
  与 transformers 5.5.4（都满足）；`torch==2.10.0+cu128` 的要求只属于不经过的
  MK1 v12 加载器。A100 训练加 `--allow-unqualified-device` 即可。
- **`Isaac-0.5 CUDA inference requires Hopper SM90 / NVIDIA H100`（A100 报错）**：
  便携模型的设备生产闸门；训练加 `--allow-unqualified-device` 自动写入豁免
  （`ISAAC_ALLOW_UNQUALIFIED_DEVICE=1`），并确保用生产镜像（transformers 5.5.4 /
  cu128）。
- **`require_qualified_runtime` 报错（transformers 版本）**：便携模型要求
  transformers 5.5.4 / CUDA 12.8 / SM90（H100），请用 `PRODUCTION_TORCH=1`
  构建的生产镜像。
- **`Checkpoint-local path hf_model_path escapes the package root`**：`--patch-dir`
  必须与模型目录同级（便携包布局，`hf_model_path=".."` 解析到模型根）。
- **`ISAAC normalization stats target_fps mismatch`**：`--fps` 必须与数据集
  `meta/info.json` 的 fps 一致（10）。
