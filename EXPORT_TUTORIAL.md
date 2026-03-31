# Kokoro ONNX 导出教程

本教程详细介绍如何将 Kokoro 语音合成模型导出为 ONNX 格式，并说明各导出文件的含义与用途。

---

## 目录

- [模型架构简介](#模型架构简介)
- [前置环境要求](#前置环境要求)
- [安装步骤](#安装步骤)
- [准备模型文件](#准备模型文件)
- [导出流程](#导出流程)
- [导出参数说明](#导出参数说明)
- [导出结果说明](#导出结果说明)
- [导出验证](#导出验证)
- [模型转换（ONNX → Axera）](#模型转换onnx--axera)
- [常见问题](#常见问题)

---

## 模型架构简介

Kokoro TTS 模型被拆分为 4 个 ONNX 子模型，以便在 Axera NPU 上高效推理：

```
输入文本
    │
    ▼
┌─────────────────────────────────────────┐
│  Model 1: model1_bert_duration           │
│  BERT 编码 + 时长预测                     │
│  输入: input_ids, ref_s,                 │
│         input_lengths, text_mask         │
│  输出: duration（时长）, d（上下文特征）   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Model 2: model2_f0_n_asr               │
│  F0/能量预测 + 文本对齐                   │
│  输入: en, ref_s, input_ids,             │
│         input_lengths, text_mask,        │
│         pred_aln_trg（对齐矩阵）          │
│  输出: F0_pred（基频）, N_pred（能量）,   │
│         asr（对齐后的声学特征）            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Model 4: model4_har（可与 Model 3 并行）│
│  谐波分析                                │
│  输入: F0_pred                           │
│  输出: har（谐波频谱）                    │
└─────────────────────────────────────────┘
    │
    ▼ （Model 2 + Model 4 的输出汇入）
┌─────────────────────────────────────────┐
│  Model 3: model3_decoder                │
│  解码器（生成音频特征）                   │
│  输入: asr, F0_pred, N_pred, ref_s, har  │
│  输出: x（最终音频特征）                  │
└─────────────────────────────────────────┘
    │
    ▼
  音频输出（WAV）
```

> **注意**：Model 4 和 Model 3 存在依赖关系，Model 4 的输出 `har` 是 Model 3 的输入之一。

---

## 前置环境要求

| 要求 | 版本/说明 |
|------|-----------|
| 操作系统 | Linux（推荐 Ubuntu 20.04/22.04）或 macOS |
| Python | **3.10**（已验证） |
| 内存 | 建议 ≥ 8 GB RAM |
| 磁盘空间 | 导出目录约需 500 MB（含模型权重） |
| CUDA | 可选；CPU 导出即可满足需求 |

---

## 安装步骤

### 1. 克隆仓库（含 LFS 文件）

```bash
# 确保已安装 Git LFS
git lfs install

git clone https://github.com/erquren/erquren_kokoro.git
cd erquren_kokoro

# 拉取 LFS 管理的大文件（音色文件等）
git lfs pull
```

### 2. 创建并激活 Python 虚拟环境

```bash
# 创建虚拟环境
python3.10 -m venv kokoro_export

# 激活（Linux/macOS）
source kokoro_export/bin/activate

# 激活（Windows）
# kokoro_export\Scripts\activate
```

### 3. 安装依赖库

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

主要依赖包括：

| 包名 | 用途 |
|------|------|
| `torch` | PyTorch 深度学习框架 |
| `onnx` | ONNX 模型格式支持 |
| `onnxsim` | ONNX 模型简化工具 |
| `onnxruntime` | ONNX 推理引擎（用于验证） |
| `transformers` | BERT 模型支持 |
| `misaki` | Kokoro 音素化工具 |
| `pypinyin` / `jieba` / `cn2an` | 中文文本处理 |

---

## 准备模型文件

### 下载 Kokoro 模型权重

```bash
mkdir -p checkpoints

# 从 AXERA-TECH 官方 Release 下载模型权重（约 330 MB）
wget -O checkpoints/kokoro-v1_0.pth \
    https://github.com/AXERA-TECH/kokoro.axera/releases/download/v1.0.0/kokoro-v1_0.pth
```

或使用 curl：

```bash
curl -L \
    https://github.com/AXERA-TECH/kokoro.axera/releases/download/v1.0.0/kokoro-v1_0.pth \
    -o checkpoints/kokoro-v1_0.pth
```

下载完成后，`checkpoints/` 目录结构如下：

```
checkpoints/
├── config.json           # 模型配置文件（已在仓库中）
├── kokoro-v1_0.pth       # 模型权重（需下载）
└── voices/               # 音色文件（已通过 Git LFS 拉取）
    ├── af_heart.pt       # 英文女声
    ├── zf_xiaoxiao.pt    # 中文女声（小小）
    ├── zf_xiaoyi.pt      # 中文女声（小艺）
    ├── zm_yunjian.pt     # 中文男声（云间）
    └── ...               # 更多音色
```

---

## 导出流程

### 方式一：使用随机数据（快速验证，无需音色文件）

适合快速测试导出流程是否正常，生成形状固定的 ONNX 模型：

```bash
python export.py \
    --config_file checkpoints/config.json \
    --checkpoint_path checkpoints/kokoro-v1_0.pth \
    --output_dir onnx \
    --input_length 96
```

### 方式二：使用真实中文样本（推荐用于量化）

使用真实文本和音色进行导出，生成的校准数据更贴近真实推理场景，量化效果更好：

```bash
# 中文导出（使用默认中文文本）
python export.py \
    --config_file checkpoints/config.json \
    --checkpoint_path checkpoints/kokoro-v1_0.pth \
    --output_dir onnx \
    --use_real_sample \
    --lang_code z \
    --voice checkpoints/voices/zf_xiaoxiao.pt
```

```bash
# 中文导出（自定义文本）
python export.py \
    --config_file checkpoints/config.json \
    --checkpoint_path checkpoints/kokoro-v1_0.pth \
    --output_dir onnx \
    --use_real_sample \
    --lang_code z \
    --text "致力于打造世界领先的人工智能感知与边缘计算芯片。" \
    --voice checkpoints/voices/zf_xiaoyi.pt
```

### 方式三：使用真实英文样本

```bash
python export.py \
    --config_file checkpoints/config.json \
    --checkpoint_path checkpoints/kokoro-v1_0.pth \
    --output_dir onnx \
    --use_real_sample \
    --lang_code a \
    --text "The sky above the port was the color of television, tuned to a dead channel." \
    --voice checkpoints/voices/af_heart.pt
```

---

## 导出参数说明

| 参数名称 | 简写 | 默认值 | 说明 |
|----------|------|--------|------|
| `--config_file` | `-c` | `checkpoints/config.json` | 模型配置文件路径 |
| `--checkpoint_path` | `-p` | `checkpoints/kokoro-v1_0.pth` | 模型权重文件路径 |
| `--output_dir` | `-o` | `onnx` | ONNX 文件输出目录 |
| `--use_real_sample` | — | False（不使用） | 使用真实文本样本（否则使用随机数据） |
| `--lang_code` | `-l` | `a` | 语言代码：`a`=美式英文，`z`=中文 |
| `--input_length` | — | `96` | 输入音素序列长度（固定形状模型） |
| `--text` | — | None（使用内置默认文本） | 自定义输入文本 |
| `--voice` | — | None（根据语言自动选择） | 音色文件路径（`.pt` 格式） |

### 音色文件说明

音色文件存放在 `checkpoints/voices/` 目录下，命名规则：

| 前缀 | 语言 / 口音 |
|------|------------|
| `af_` | 美式英文女声 |
| `am_` | 美式英文男声 |
| `bf_` | 英式英文女声 |
| `bm_` | 英式英文男声 |
| `zf_` | 中文女声 |
| `zm_` | 中文男声 |
| `jf_` / `jm_` | 日文 |
| `ff_` | 法文 |

---

## 导出结果说明

成功导出后，`onnx/` 目录结构如下：

```
onnx/
├── model1_bert_duration.onnx       # Model 1 原始版本
├── model1_bert_duration_sim.onnx   # Model 1 简化版本 ⭐ 推荐
├── model2_f0_n_asr.onnx            # Model 2 原始版本
├── model2_f0_n_asr_sim.onnx        # Model 2 简化版本 ⭐ 推荐
├── model3_decoder.onnx             # Model 3 原始版本
├── model3_decoder_sim.onnx         # Model 3 简化版本 ⭐ 推荐
├── model4_har.onnx                 # Model 4 原始版本
├── model4_har_sim.onnx             # Model 4 简化版本 ⭐ 推荐
│
├── model1_input_ids.zip            # Model 1 量化校准数据：输入 token id
├── model1_ref_s.zip                # Model 1 量化校准数据：参考风格向量
├── model1_input_lengths.zip        # Model 1 量化校准数据：输入长度
├── model1_text_mask.zip            # Model 1 量化校准数据：文本掩码
│
├── model2_en.zip                   # Model 2 量化校准数据
├── model2_ref_s.zip
├── model2_input_ids.zip
├── model2_input_lengths.zip
├── model2_text_mask.zip
├── model2_pred_aln_trg.zip         # Model 2 量化校准数据：对齐矩阵
│
├── model3_asr.zip                  # Model 3 量化校准数据
├── model3_F0_pred.zip
├── model3_N_pred.zip
├── model3_ref_s.zip
├── model3_har.zip
│
└── model4_F0_pred.zip              # Model 4 量化校准数据
```

### 文件类型说明

| 文件类型 | 说明 | 用途 |
|----------|------|------|
| `*.onnx` | 原始导出的 ONNX 模型 | 可直接用于 onnxruntime 推理验证 |
| `*_sim.onnx` | 经 onnxsim 简化的 ONNX 模型 | **推荐用于 Axera NPU 量化和推理** |
| `*.zip` | 压缩的 NumPy 数组（`.npy` 格式） | 用于 Axera pulsar2 量化校准 |

> ⭐ **`_sim.onnx` 文件**：通过 `onnxsim` 工具对计算图进行了常量折叠、节点融合等优化，
> 在保持数值精度一致的前提下减小模型体积并提升推理效率，是用于量化的首选版本。

### 模型输入输出形状

以 `input_length=96` 为例：

#### Model 1 (`model1_bert_duration_sim.onnx`)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `input_ids` | `[1, 96]` | int64 | 音素 token ID 序列 |
| `ref_s` | `[1, 256]` | float32 | 参考风格向量 |
| `input_lengths` | `[1]` | int64 | 实际输入长度 |
| `text_mask` | `[1, 96]` | bool | 填充掩码 |
| **输出** `duration` | `[1, 96, 2]` | float32 | 每个音素的时长预测 |
| **输出** `d` | `[1, 512, 96]` | float32 | 文本编码特征 |

#### Model 2 (`model2_f0_n_asr_sim.onnx`)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `en` | `[1, 512, 192]` | float32 | 对齐后的文本特征 |
| `ref_s` | `[1, 256]` | float32 | 参考风格向量 |
| `input_ids` | `[1, 96]` | int64 | 音素 token ID 序列 |
| `input_lengths` | `[1]` | int64 | 实际输入长度 |
| `text_mask` | `[1, 96]` | bool | 填充掩码 |
| `pred_aln_trg` | `[1, 96, 192]` | float32 | CTC 对齐矩阵 |
| **输出** `F0_pred` | `[1, 192]` | float32 | 基频（F0）预测 |
| **输出** `N_pred` | `[1, 192]` | float32 | 能量（N）预测 |
| **输出** `asr` | `[1, 64, 192]` | float32 | 声学特征 |

#### Model 4 (`model4_har_sim.onnx`)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `F0_pred` | `[1, 192]` | float32 | 基频预测 |
| **输出** `har` | `[1, 20, 1152]` | float32 | 谐波频谱 |

#### Model 3 (`model3_decoder_sim.onnx`)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `asr` | `[1, 64, 192]` | float32 | 声学特征 |
| `F0_pred` | `[1, 192]` | float32 | 基频预测 |
| `N_pred` | `[1, 192]` | float32 | 能量预测 |
| `ref_s` | `[1, 256]` | float32 | 参考风格向量 |
| `har` | `[1, 20, 1152]` | float32 | 谐波频谱 |
| **输出** `x` | `[1, 1, 57600]` | float32 | 合成音频（采样点） |

> **注意**：以上形状基于 `input_length=96`，若使用不同的 `input_length`，
> 时间维度（192 = input_length × 2，57600 = input_length × 600）会按比例变化。

---

## 导出验证

导出完成后，可使用 `onnxruntime` 对模型进行推理验证：

```python
import numpy as np
import onnxruntime as ort

# 以 input_length=96 为例

# ── Model 1 验证 ──────────────────────────────────────────────────
sess1 = ort.InferenceSession("onnx/model1_bert_duration_sim.onnx")
out1 = sess1.run(None, {
    "input_ids":     np.random.randint(1, 10, (1, 96), dtype=np.int64),
    "ref_s":         np.random.randn(1, 256).astype(np.float32),
    "input_lengths": np.array([96], dtype=np.int64),
    "text_mask":     np.zeros((1, 96), dtype=bool),
})
print("Model 1 duration shape:", out1[0].shape)  # 期望: (1, 96, 2)
print("Model 1 d shape:        ", out1[1].shape)  # 期望: (1, 512, 96)

# ── Model 4 验证 ──────────────────────────────────────────────────
sess4 = ort.InferenceSession("onnx/model4_har_sim.onnx")
out4 = sess4.run(None, {
    "F0_pred": np.random.randn(1, 192).astype(np.float32),
})
print("Model 4 har shape:", out4[0].shape)  # 期望: (1, 20, 1152)

print("验证完成！")
```

---

## 模型转换（ONNX → Axera）

将 ONNX 模型转换为 Axera NPU 可运行的 `.axmodel` 格式，需使用 `pulsar2` 工具。

> **前提**：已安装 Axera pulsar2 工具链（参见 Axera 官方文档）。

### 转换命令示例

```bash
# 转换 Model 1（以 input_length=96 为例）
pulsar2 build --config config/kokoro_1.json

# 转换 Model 2
pulsar2 build --config config/kokoro_2.json

# 转换 Model 3
pulsar2 build --config config/kokoro_3.json
```

> **注意**：Model 4 无需单独转换，其功能已集成到推理流程中。

### 配置文件说明

`config/` 目录下的 JSON 配置文件指定了量化参数，关键字段说明：

| 字段 | 说明 |
|------|------|
| `input` | 输入的 `_sim.onnx` 文件路径 |
| `output_dir` | axmodel 输出目录 |
| `quant.input_configs` | 各输入张量的量化校准数据（对应 `.zip` 文件） |
| `quant.layer_configs.data_type` | 量化精度（`U16` = 16位无符号整数） |
| `quant.calibration_method` | 量化校准方法（`MinMax`） |
| `npu_mode` | NPU 并行模式（`NPU3` = 3核并行） |
| `target_hardware` | 目标硬件（`AX650`） |

---

## 通过 GitHub Actions 自动导出（CI/CD）

本仓库提供了 GitHub Actions 工作流 `.github/workflows/export_onnx_release.yml`，
可以自动完成导出并发布到 GitHub Release。

### 触发步骤

1. 进入仓库的 **Actions** 标签页
2. 选择 **"导出 ONNX 模型并发布到 Release"** 工作流
3. 点击 **"Run workflow"** 按钮
4. 填写参数：
   - **`tag_name`**：Release 标签（如 `v1.0.0`）
   - **`input_length`**：输入序列长度（默认 `96`）
   - **`release_notes`**：可选的 Release 说明
5. 点击 **"Run workflow"** 启动

工作流将自动：
- 下载模型权重
- 运行 ONNX 导出脚本
- 验证导出的模型
- 将所有 ONNX 文件和校准数据打包为 ZIP
- 创建 GitHub Release 并上传压缩包

---

## 常见问题

### Q1：导出时报 `ModuleNotFoundError: No module named 'misaki'`

```bash
pip install "misaki>=0.7.0,<0.9.0"
```

### Q2：导出时报 `espeak not found`

phonemizer 需要系统安装 espeak-ng：

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak
```

### Q3：`onnxsim` 简化失败，没有生成 `_sim.onnx`

这是非致命错误，原始 `.onnx` 文件仍然可用。可手动运行简化：

```bash
python -m onnxsim onnx/model1_bert_duration.onnx onnx/model1_bert_duration_sim.onnx
```

### Q4：如何选择 `input_length`？

- 默认值 **96** 适合大多数中英文短句（约 10-20 个汉字或 15-30 个英文单词）
- 若需处理更长的文本，可增大此值（但会增大模型尺寸）
- 推理时，输入文本的音素数量必须恰好等于导出时指定的 `input_length`

### Q5：`--use_real_sample` 和不使用的区别？

| 对比项 | 随机数据（默认） | 真实样本 (`--use_real_sample`) |
|--------|-----------------|-------------------------------|
| 导出速度 | 更快 | 稍慢（需要运行 G2P 文本转音素转换） |
| ONNX 模型 | 完全相同 | 完全相同 |
| 校准数据质量 | 随机，量化效果可能较差 | 真实分布，**量化效果更好** |
| 推荐场景 | 快速验证导出流程 | **用于 NPU 量化的最终导出** |

### Q6：如何验证导出的 ONNX 模型是否正确？

```bash
python -c "
import onnx
for i, name in enumerate(['model1_bert_duration', 'model2_f0_n_asr', 
                           'model3_decoder', 'model4_har'], 1):
    path = f'onnx/{name}_sim.onnx'
    onnx.checker.check_model(onnx.load(path))
    print(f'Model {i} ({name}_sim.onnx): 验证通过')
"
```
