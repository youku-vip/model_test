# Isaac Docker 镜像使用说明

基于 [perceptron-ai-inc/isaac](https://github.com/perceptron-ai-inc/isaac)（Isaac 0.5，
36B 稀疏具身基础模型，内嵌固定的 LeRobot 子模块 `perceptron-ai-inc/lerobot`）构建的
Docker 镜像，用于 **模型部署与训练**。

## 镜像说明

| 镜像 | Dockerfile | 基础镜像 | 用途 |
| ---- | ---------- | -------- | ---- |
| `isaac:gpu` | `Dockerfile` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | GPU 推理 + 训练（含官方 mharmony 集成） |
| `isaac:gpu-production` | `Dockerfile` + `--build-arg PRODUCTION_TORCH=1` | 同上 | H100 生产部署（torch 2.10.0+cu128） |
| `isaac:cpu` | `Dockerfile.cpu` | `python:3.12-slim` | CPU 开发 / 数据转换 / 轻量部署 |

GPU 镜像默认安装的 uv extras：

- `perceptron_isaac_cuda` — Isaac 策略 + CUDA 内核（flash-linear-attention==0.4.1、
  causal-conv1d==1.6.0，无预编译 wheel，镜像内使用 CUDA 12.8 devel 工具链源码编译）
- `training` — `lerobot-train`、数据集、accelerate、wandb
- `peft` — PEFT 微调（VLM backbone）

## 构建

```bash
# 从仓库根目录（包含 lerobot 子模块）
./build.sh                 # GPU 镜像 -> isaac:gpu
./build.sh cpu             # CPU 镜像 -> isaac:cpu

# 自定义构建参数
docker build -f Dockerfile -t isaac:gpu \
    --build-arg HF_ENDPOINT=https://hf-mirror.com \   # huggingface.co 被墙时使用
    --build-arg PRODUCTION_TORCH=1 \                  # 切换为 MK1 v12 生产运行时
    .
```

## 运行（环境-only：代码/权重/数据全部挂载）

```bash
# GPU 推理 / 训练（宿主需有 NVIDIA 驱动 + nvidia-container-toolkit）
docker run -it --rm --gpus all --shm-size 16gb \
    -v /host/isaac:/code \              # 你的 isaac+lerobot 源码（e12389c，PYTHONPATH 优先）
    -v /host/checkpoints:/checkpoint \  # 模型权重/导入包
    -v /host/data:/data \               # 数据集/输出
    -v /path/to/vocab:/vocab:ro \
    -e QWEN35_VOCAB_PATH=/vocab/vocab.json \
    isaac:gpu

# 进入容器后先做环境自检（脚本在 /usr/local/bin，不被挂载覆盖）
isaac-check

# 冒烟测试（官方 mharmony API）
python /opt/isaac/docker/mharmony_smoke_test.py
```

**挂载说明**：
- `/code` = 代码挂载点（`PYTHONPATH=/code/lerobot/src` 使挂载的 lerobot 源码优先于镜像内快照；不挂载则用镜像自带快照，同一提交 e12389c）
- `/checkpoint`、`/data` = 权重与数据
- 镜像内 venv 在 `/opt/venv`，辅助脚本在 `/usr/local/bin` 与 `/opt/isaac/docker`——**都不在挂载覆盖范围内**

## 典型用法

### 1. 导入 Isaac 检查点（部署第一步）

```bash
docker run -it --rm --gpus all --shm-size 16gb \
    -v /host/checkpoints:/checkpoints \
    isaac:gpu \
    lerobot-isaac-import \
      --hf-export /checkpoints/raw-hf-export \
      --deployment-adapter /checkpoints/isaac_deployment_adapter.json \
      --output /checkpoints/isaac-lerobot-package \
      --policy-state-dataset cloud/isaac_yam \
      --normalization-scope yam \
      --allow-fast-remote-code
```

### 2. 微调（Fine-tuning）

```bash
docker run -it --rm --gpus all --shm-size 16gb \
    -v /host/data:/data \
    isaac:gpu \
    lerobot-train \
      --policy.path=/data/isaac-lerobot-package \
      --dataset.repo_id=local/isaac_yam \
      --dataset.root=/data/lerobot-dataset \
      --batch_size=1 --gradient_accumulation_steps=8 \
      --steps=1000 --output_dir=/data/outputs/train/isaac-yam
```

多卡训练：

```bash
docker run -it --rm --gpus all --shm-size 16gb -e CUDA_VISIBLE_DEVICES=0,1,2,3 isaac:gpu \
    accelerate launch --num_processes=4 --mixed_precision=bf16 \
      -m lerobot.scripts.lerobot_train --policy.path=/data/pkg --dataset.repo_id=local/isaac_yam ...
```

### 3. 推理 / 评估

```bash
docker run -it --rm --gpus all --shm-size 16gb isaac:gpu \
    lerobot-eval \
      --policy.path=/data/pkg --policy.device=cuda:0 \
      --env.type=libero --env.task=libero_spatial \
      --eval.batch_size=1 --eval.n_episodes=20 --seed=0 \
      --output_dir=/tmp/eval
```

## 重要注意事项

1. **mharmony 运行时缺口（上游未解决）**：`perceptron_isaac` extra 尚未声明外部
   mharmony 运行时（`genesis.data.mharmony`），该包还未发布到 PyPI。本镜像已安装
   extra 声明的全部依赖，策略模块可正常导入；但**完整渲染 / 训练 / 推理**还需要
   该运行时（参考 `lerobot/docs/PERCEPTRON_ISAAC_PROVENANCE.md`）。部署时需：
   - 挂载或自行安装已发布的 mharmony 发行版；或
   - 等待 Genesis PR #3075 发布独立 mharmony 包后重新构建。
   镜像内 `isaac-check` 会明确报告 mharmony 是否可用。

2. **MK1 生产运行时**：v12 action-parity 运行时要求 torch==2.10.0+cu128、
   transformers==5.5.4、`torch_sdpa_v1` attention 后端与 NVIDIA H100。默认锁文件
   是 torch 2.11.0+cu128（CI 同款）。需要生产 ABI 时用
   `--build-arg PRODUCTION_TORCH=1` 构建（`isaac:gpu-production`）：镜像会以
   torch 2.10.0+cu128 编译 causal-conv1d，并在 uv sync 之后恢复该 ABI 的
   torch/torchvision。注意该路径下视频解码请用 `--dataset.video_backend=pyav`
   （torchcodec 与 torch 2.10 不匹配），且不要再执行 `uv sync`（会移除 torch）。

3. **HuggingFace 网络**：`huggingface.co` 在某些网络不可达（本构建环境即如此）。
   构建或运行时用 `--build-arg HF_ENDPOINT=https://hf-mirror.com`（或容器内
   `export HF_ENDPOINT=...`）切换到镜像源。

4. **模型权重**：Isaac-0.5 权重托管在 HuggingFace
   [PerceptronAI/Isaac-0.5](https://huggingface.co/PerceptronAI/Isaac-0.5)，首次推理
   时自动下载；可挂载 `-v $HOME/.cache/huggingface:/home/user_lerobot/.cache/huggingface`
   复用宿主机缓存。

5. **CUDA 内核编译**：GPU 镜像基于 `-devel` 基础镜像，nvcc/gcc/ninja 常驻，以便
   causal-conv1d 源码构建与 flash-linear-attention 的运行时 JIT。causal-conv1d 1.6.0
   没有 PyPI 预编译 wheel，且其安装脚本会尝试从 github.com 下载（本网络被墙），因此
   镜像改为 `CAUSAL_CONV1D_FORCE_BUILD=TRUE` 源码编译，并预先安装与锁文件一致的
   torch 以保证 ABI 正确。构建时可用 `TORCH_CUDA_ARCH_LIST` 控制编译的 GPU 架构
   （默认 `8.0 8.6 8.9 9.0`；H100 只需 `9.0`）。

6. **受限构建环境**：若 `docker build` 报
   `failed to update builder last activity time ... read-only file system`，先执行
   `export DOCKER_CONFIG=/可写目录/.docker` 再构建。本仓库的 `build.sh` 默认使用
   当前 `DOCKER_CONFIG`。

7. **mharmony 官方集成（lerobot e12389c）**：上游已将独立 mharmony 包（`mharmony[qwen35]==0.1.0`，
   git rev `9a57efb`）正式声明进 `perceptron_isaac` extra——`uv sync` 会自动从 git 构建安装
   （需要 Rust 工具链，镜像已内置）。不再需要 `genesis.data.mharmony` 命名空间或任何 shim。

   **代码版本要求**（官方 pin）：
   ```bash
   git clone https://github.com/perceptron-ai-inc/isaac.git
   cd isaac && git checkout be6507b4aed7472f2029606c22684d4ebc9d73e6
   git submodule update --init --recursive
   git -C lerobot fetch origin main
   git -C lerobot checkout e12389c1f8f591ad05dced4e284d4e92e48c5df4
   ```

   构建（github.com 被墙时用默认 gh-proxy 源）：
   ```bash
   docker build -f Dockerfile -t isaac:gpu .
   ```

   **运行时仍需**：
   - Qwen35Harmony 词表：设置 `QWEN35_VOCAB_PATH=/path/to/vocab.json`（ModelScope 可下载，
     SHA-256 与上游 pin 一致），或容器内 `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")`
   - 冒烟测试：`python docker/mharmony_smoke_test.py`（官方 API：编码加载 → 渲染 →
     `rendered_stream_to_local_tensor_stream`）

## 训练/调试脚本（docker/ 下）

| 脚本 | 用途 |
| --- | --- |
| `debug_bimanual.sh` | 4×A100 短跑验证（自检 + patch 包 + 20 步） |
| `train_bimanual_8gpu.sh` | 8×A100 正式训练 + 离线测试 |
| `patch_package_for_bimanual.py` | 把导入包改成自定义维度（action/proprio/相机/fps）+ 重算统计 |
| `eval_offline.py` | 离线测试（预测 vs 真值 MAE） |
| `mharmony_smoke_test.py` | mharmony 全链路自检 |
