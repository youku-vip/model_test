# Isaac 0.5 / A100-80GB / DeepSpeed

仓库现在额外提供一个 **DeepSpeed ZeRO-3** 后端，作为现有 FSDP 路径的可切换 benchmark。

## 为什么是 ZeRO-3

Isaac-0.5 为 36B 级模型。BF16 参数本身约 72GB。ZeRO-2 只分片 optimizer/gradient state，模型参数仍在每张 GPU 上复制，因此对 A100-80GB 过于紧张；ZeRO-3 才会进一步分片参数。Accelerate 当前支持 ZeRO-1/2/3，并可通过 DeepSpeed config file 配置。citeturn929116search0turn929116search1

## 运行

容器内：

```bash
bash /code/run/train_a100_deepspeed.sh
```

默认配置：

```text
8 x A100-80GB
micro batch = 1 / GPU
grad accumulation = 4
effective batch = 32
Flow MC samples/chunk = 2
max sequence length = 8192
RTC max delay = 12
RTC probability = 0.5
FP32 train storage = off
DeepSpeed = ZeRO-3
```

覆盖路径/规模：

```bash
GPUS=0,1,2,3 \
DATASET_DIR=/data/local/bimanual \
OUTPUT_DIR=/data/outputs/isaac-finetune-ds3-smoke \
STEPS=20 \
SAVE_FREQ=20 \
NUM_WORKERS=8 \
SAMPLES_PER_CHUNK=1 \
bash /code/run/train_a100_deepspeed.sh
```

也可以直接使用包装入口，原 `finetune_isaac05.py` 的参数全部保留：

```bash
python /code/run/finetune_isaac05_deepspeed.py \
  --dataset-dir /data/local/bimanual \
  --gpus 0,1,2,3,4,5,6,7 \
  --steps 20000 \
  --batch-size 1 \
  --grad-accum 4 \
  --train-samples-per-chunk 2 \
  --detach-vlm-activations \
  --loss-plan flow_matching_action_prediction \
  --train-storage-fp32 false \
  --allow-unqualified-device
```

## 与 FSDP 的关系

`run/finetune_isaac05.py --fsdp` 仍然是默认推荐路径。DeepSpeed 入口只是把最终的 `accelerate launch` 后端改成 ZeRO-3，因此数据校验、Isaac 包 patch、RTC、Flow MC、学习率、W&B、resume 等仍由原脚本控制。

DeepSpeed config 在 `run/deepspeed_zero3_a100.json`，开启 BF16、ZeRO-3、通信/梯度优化，并关闭 CPU/NVMe offload，避免把训练瓶颈转移到主机 I/O。

## Checkpoint 注意事项

ZeRO-3 的参数是跨 GPU 分片的。当前 config 显式关闭 `stage3_gather_16bit_weights_on_model_save`，避免在单个 A100 上把约 72GB BF16 参数重新聚合。首次使用 DeepSpeed 路径时，**先跑 20 步 smoke，确认训练进程、显存、loss 和 checkpoint 行为，再作为正式训练后端**。

尤其不要直接假定 ZeRO-3 生成的 checkpoint 与现有 `pretrained_model` 部署目录具有完全相同的单体权重布局；这需要在实际 pinned Perceptron LeRobot 环境中验证。FSDP 路径的现有保存/加载契约仍然是当前生产基线。
