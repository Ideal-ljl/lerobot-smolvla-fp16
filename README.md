# SmolVLA FP16 Hardware Benchmark

本项目用于测试 SmolVLA 在不同硬件平台上的推理性能，重点比较 NVIDIA Jetson Orin NX 与桌面端/服务器端 GPU 在相同输入、相同模型和相同推理流程下的差异。

项目基于 Hugging Face LeRobot 修改，目标不是提供完整的机器人训练框架，而是提供一套可以复现的 SmolVLA FP16 推理流程。模型 checkpoint、数据集和测试输入不会提交到本仓库。

## 主要特性

- SmolVLA 推理链路强制使用 FP16，不使用 BF16、FP32 或 TF32。
- 默认使用全零初始 latent，完全跳过高斯随机采样。
- 支持将预处理后的模型输入、固定 noise 和参考输出保存到单个文件。
- 支持在另一台机器上直接重放输入，并进行逐 bit 输出检查。
- 启用 PyTorch 确定性算法，关闭 cuDNN benchmark 和 TF32。
- CUDA 不可用时直接终止，不静默回退到 CPU。

## 测试目的

建议在所有设备上使用同一份：

- SmolVLA checkpoint
- 重放输入文件
- PyTorch、CUDA、Transformers 和 LeRobot 代码版本
- 推理参数与功耗模式

这样可以尽量将差异限定在硬件、驱动和底层 CUDA kernel 上。不同 GPU 架构或软件栈仍可能产生数值末位差异，因此重放流程会同时保存参考输出并检查是否 bit-exact。

## 环境

已验证环境名称为 `lerobot_orin`，需要：

- Linux
- Python 3.10
- NVIDIA CUDA GPU
- 支持 CUDA 的 PyTorch
- LeRobot 的 SmolVLA 依赖

按照本仓库依赖安装：

```bash
conda create -n lerobot_orin python=3.10 -y
conda activate lerobot_orin
pip install -e ".[smolvla]"
```

Jetson Orin NX 上的 PyTorch 应使用与 JetPack/CUDA 对应的 NVIDIA Jetson 版本，不能直接假设 PyPI wheel 与设备兼容。

## 文件准备

默认目录结构如下：

```text
lerobot/
├── smolvla_base/                         # checkpoint，不上传
├── libero_goal_no_noops_1.0.0_lerobot/  # 数据集，不上传
├── src/lerobot/
└── test.py
```

也可以通过命令行参数指定其他路径：

```bash
python test.py \
  --model /path/to/smolvla_base \
  --dataset-root /path/to/dataset \
  --repo-id libero_goal_no_noops_1.0.0_lerobot \
  --frame-index 0
```

## 首次推理与保存输入

默认使用全零 latent，因此推理过程中不存在高斯采样：

```bash
conda run -n lerobot_orin python test.py \
  --input-file smolvla_replay_input.pt
```

保存文件包含：

- 预处理、归一化和 tokenization 后的模型输入
- FP16 初始 noise/latent
- 本次推理得到的参考 normalized action

如果测试需要高斯分布的初始 latent，可以在一台机器上用固定 seed 生成一次，再将保存的输入复制到其他机器：

```bash
conda run -n lerobot_orin python test.py \
  --seeded-noise \
  --seed 0 \
  --input-file smolvla_replay_input.pt
```

## 在另一台机器重放

将相同 checkpoint 和 `smolvla_replay_input.pt` 放到目标机器，然后执行：

```bash
conda run -n lerobot_orin python test.py \
  --model /path/to/smolvla_base \
  --replay \
  --input-file /path/to/smolvla_replay_input.pt
```

输出完全一致时会显示：

```text
Replay verification: bit-exact output match
```

如果不一致，程序会报错并打印最大绝对误差。这可以帮助区分硬件/软件栈造成的数值差异与输入预处理差异。

## 性能测试建议

为了得到可比较的结果：

1. 先运行一次推理完成模型加载和 CUDA warm-up。
2. 使用 `--replay`，避免把数据解码和预处理时间计入模型推理。
3. 每个平台重复多次，分别记录延迟、吞吐量、显存、功耗和温度。
4. Orin NX 上固定 JetPack、功耗模式和时钟策略，并记录 `tegrastats` 输出。
5. 桌面 GPU 上记录 GPU 型号、驱动、CUDA、PyTorch 版本和功耗上限。
6. 同步 CUDA 后再计时，否则 CPU 侧计时不能代表真实 GPU 延迟。

## 与官方 LeRobot 的主要差异

核心改动位于：

- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `test.py`

改动包括 VLM 权重、RoPE、attention、时间编码、动作头、预处理和后处理的 FP16 化，以及确定性输入保存与重放。

上游项目：[huggingface/lerobot](https://github.com/huggingface/lerobot)

## 不包含的内容

以下内容已通过 `.gitignore` 排除：

- 模型 checkpoint
- LIBERO 数据集及其备份
- 推理重放输入文件
- 本地输出、缓存和日志

请确认你有权使用和分发自行下载的模型与数据集。

## License

本项目保留上游 LeRobot 的 Apache License 2.0。详见 [LICENSE](LICENSE)。
