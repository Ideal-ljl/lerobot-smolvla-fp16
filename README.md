# SmolVLA FP16 Hardware Benchmark

本项目用于测试 SmolVLA 在不同硬件平台上的推理性能，重点比较 NVIDIA Jetson Orin NX 与桌面端/服务器端 GPU 在完全相同输入和推理流程下的差异。

项目基于 Hugging Face LeRobot 修改，仅保留实验所需代码和环境信息。模型 checkpoint 与完整数据集不包含在仓库中；仓库已经提供 10 条可直接重放的 FP16 模型输入和参考输出。

## 特性

- SmolVLA 推理链路使用 FP16，不使用 BF16、FP32 或 TF32。
- 使用固定的全零 FP16 latent，不进行高斯采样。
- 10 条测试样本分别来自 LIBERO 数据集的 10 个 task。
- 每条样本包含预处理后的模型输入、noise、normalized action 和反归一化 action。
- 启用 PyTorch 确定性算法，关闭 cuDNN benchmark 和 TF32。
- 重放时执行 bit-exact 检查，并输出每条样本的 GPU 推理延迟。
- CUDA 不可用时直接终止，不回退到 CPU。

## 下载模型和数据集

SmolVLA checkpoint：

- [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)

```bash
hf download lerobot/smolvla_base --local-dir ./smolvla_base
```

LIBERO 数据集（下载版本为 LeRobot v2.1）：

- [IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)

```bash
hf download \
  --repo-type dataset \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  --local-dir ./libero_goal_no_noops_1.0.0_lerobot
```

只运行仓库中已经准备好的 10 条 benchmark 时不需要下载完整数据集，只需要 checkpoint。

## 环境

当前验证环境为 Jetson/JetPack CUDA 12.6、Python 3.10。完整环境快照位于 [`requirements-lerobot-orin.txt`](requirements-lerobot-orin.txt)，其中包含当前 Orin 环境使用的 NVIDIA PyTorch wheel 地址。

```bash
conda create -n lerobot_orin python=3.10 -y
conda activate lerobot_orin
pip install -r requirements-lerobot-orin.txt
pip install -e . --no-deps
```

该 requirements 是当前 Orin 环境的精确导出，不保证其中的 AArch64/Jetson PyTorch wheel 能安装在 x86_64 机器上。其他硬件应先安装与自身 CUDA 和架构对应的 PyTorch，再安装其余依赖，并记录实际版本。

## 运行 10 条 FP16 Benchmark

仓库中的 [`benchmark/smolvla_fp16_10_cases.pt`](benchmark/smolvla_fp16_10_cases.pt) 已包含 10 条预处理后的输入与参考输出。

```bash
conda run -n lerobot_orin python benchmark.py \
  --model ./smolvla_base \
  --cases ./benchmark/smolvla_fp16_10_cases.pt \
  --warmup 1
```

程序会：

1. 检查模型、输入和 noise 是否为 FP16。
2. 预热模型。
3. 使用 CUDA 同步计时，逐条输出推理延迟。
4. 将每条 normalized action 与仓库参考输出进行逐 bit 比较。
5. 检查反归一化输出仍为 FP16。

正常结束时会显示：

```text
All 10 FP16 outputs are bit-exact.
```

不同 GPU 架构、驱动或 CUDA/PyTorch 版本可能使用不同 kernel，因此即使输入完全相同也可能出现末位差异。发生差异时程序会报告最大绝对误差。

## 数据集 v2.1 → v3.0 转换

下载的数据集是 LeRobot v2.1，本项目推理使用的是转换后的 v3.0 数据集。仓库提供了可复现脚本：

```bash
bash scripts/convert_libero_v21_to_v30.sh "$PWD"
```

脚本执行以下流程：

1. 检查本地 v2.1 数据集目录。
2. 创建 `libero_goal_no_noops_1.0.0_lerobot_v21_backup` 完整备份。
3. 调用 LeRobot 官方 `convert_dataset_v21_to_v30` 转换器。
4. 使用 `--push-to-hub=false`，仅在本地转换，不修改上游数据集。

转换后的数据集仍位于：

```text
libero_goal_no_noops_1.0.0_lerobot/
```

其 `meta/info.json` 中的 `codebase_version` 应为 `v3.0`。原始 v2.1 数据保存在备份目录中。

用于维护 benchmark 数据的生成脚本也保留在 [`scripts/build_fp16_benchmark_cases.py`](scripts/build_fp16_benchmark_cases.py)，但普通测试者无需重新生成输入。

## Benchmark 样本

固定数据集索引如下，每个 task 取首条帧：

| Task | Dataset index | Episode | Frame |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 112 | 1 | 0 |
| 2 | 417 | 3 | 0 |
| 3 | 812 | 5 | 0 |
| 4 | 924 | 6 | 0 |
| 5 | 1014 | 7 | 0 |
| 6 | 1407 | 9 | 0 |
| 7 | 1827 | 13 | 0 |
| 8 | 2451 | 19 | 0 |
| 9 | 2551 | 20 | 0 |

## 关键文件

- `benchmark.py`：直接运行 10 条硬件 benchmark。
- `benchmark/smolvla_fp16_10_cases.pt`：FP16 输入和参考输出。
- `requirements-lerobot-orin.txt`：当前 Orin 环境精确依赖快照。
- `scripts/convert_libero_v21_to_v30.sh`：官方 v2.1 → v3.0 转换流程。
- `scripts/build_fp16_benchmark_cases.py`：benchmark 产物维护脚本。
- `src/lerobot/policies/smolvla/`：FP16 SmolVLA 实现。

## 测试记录建议

对比硬件时应记录 GPU/SoC 型号、功耗模式、时钟策略、温度、驱动、CUDA、PyTorch、Transformers、显存峰值和 10 条样本延迟。Orin NX 建议同时保存 `tegrastats` 输出。

## 上游与 License

本项目基于 [huggingface/lerobot](https://github.com/huggingface/lerobot)，保留 Apache License 2.0，详见 [LICENSE](LICENSE)。模型与数据集遵循各自页面声明的许可协议。
