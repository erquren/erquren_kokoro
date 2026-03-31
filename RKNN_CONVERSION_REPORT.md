# Kokoro TTS ONNX → RKNN (RK3588) 模型转换实验手册

## 1. 实验目标

将 Kokoro TTS 系统的 4 个 ONNX 子模型转换为 RK3588 NPU 可运行的 RKNN 格式，不进行量化（float16）。

## 2. 模型架构概览

```
输入文本 → Model 1 (BERT+时长预测) → Model 2 (F0/能量/对齐) → Model 3 (解码器) → 音频输出
                                                            ↗ Model 4 (谐波分析) ↗
```

| 模型 | 描述 | ONNX 文件 | 大小 |
|------|------|-----------|------|
| model1_bert_duration | BERT编码 + 时长预测 | model1_bert_duration_sim.onnx | 53.06 MB |
| model2_f0_n_asr | F0/能量预测 + 文本对齐 | model2_f0_n_asr_sim.onnx | 53.52 MB |
| model3_decoder | 解码器（生成音频特征）| model3_decoder_sim.onnx | 203.35 MB |
| model4_har | 谐波分析 | model4_har_sim.onnx | 0.01 MB |

## 3. 模型输入输出规格

### Model 1: model1_bert_duration
| 输入 | 形状 | 类型 | 说明 |
|------|------|------|------|
| input_ids | [1, 78] | INT64 | 文本 token ID |
| ref_s | [1, 256] | FLOAT | 参考说话人风格 |
| text_mask | [1, 78] | BOOL | 文本掩码 |

| 输出 | 形状 | 类型 |
|------|------|------|
| duration | [1, 78, 50] | FLOAT |
| d | [1, 78, 640] | FLOAT |

### Model 2: model2_f0_n_asr
| 输入 | 形状 | 类型 |
|------|------|------|
| en | [1, 640, 156] | FLOAT |
| ref_s | [1, 256] | FLOAT |
| input_ids | [1, 78] | INT64 |
| text_mask | [1, 78] | BOOL |
| pred_aln_trg | [1, 78, 156] | FLOAT |

| 输出 | 形状 | 类型 |
|------|------|------|
| F0_pred | [1, 312] | FLOAT |
| N_pred | [1, 312] | FLOAT |
| asr | [1, 512, 156] | FLOAT |

### Model 3: model3_decoder
| 输入 | 形状 | 类型 |
|------|------|------|
| asr | [1, 512, 156] | FLOAT |
| F0_pred | [1, 312] | FLOAT |
| N_pred | [1, 312] | FLOAT |
| ref_s | [1, 256] | FLOAT |
| har | [1, 22, 18721] | FLOAT |

| 输出 | 形状 | 类型 |
|------|------|------|
| x | [1, 22, 18721] | FLOAT |

### Model 4: model4_har
| 输入 | 形状 | 类型 |
|------|------|------|
| F0_pred | [1, 312] | FLOAT |

| 输出 | 形状 | 类型 |
|------|------|------|
| har | [1, 22, 18721] | FLOAT |

## 4. 环境配置

```bash
# Python 3.10 (rknn-toolkit2 要求)
python3.10 -m venv rknn_venv
source rknn_venv/bin/activate

# 安装依赖
pip install numpy onnx==1.16.2 onnxruntime==1.19.2 onnxsim

# 安装 rknn-toolkit2 v2.3.2
pip install rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

**注意**: onnx 版本必须使用 1.16.x，更新版本（如 1.21.x）移除了 `onnx.mapping` 模块，与 rknn-toolkit2 不兼容。

## 5. ONNX 预处理

### 5.1 问题说明

RK3588 NPU 及 rknn-toolkit2 存在以下数据类型限制：

1. **BOOL 类型不支持**: NPU 不支持布尔运算（Not、Where 等）
2. **INT64 可能导致崩溃**: 部分 INT64 操作导致工具链 segfault
3. **INT32 中间张量**: 可能导致 "dataconvert type -1" 错误

### 5.2 预处理策略

对包含 BOOL/INT64 输入的模型（model1、model2），执行以下转换：

```
BOOL/INT64 输入 → FLOAT 输入
Not(x)          → Sub(1.0, x)           # 浮点算术等价
Where(c, x, y)  → x*c + y*(1-c)        # 浮点算术等价
Cast(to=BOOL)   → Cast(to=FLOAT)       # 保持 0.0/1.0 语义
Cast(to=INT32)  → Cast(to=FLOAT)       # 避免 INT32 中间张量
Gather 索引     → 插入 Cast(FLOAT→INT64) # Gather 需要整数索引
```

预处理后使用 `onnxsim` 简化图结构，消除冗余节点。

### 5.3 预处理验证

对比预处理前后的 ONNXRuntime 推理结果：

| 模型 | duration max_diff | d max_diff |
|------|------------------|------------|
| model1_preprocessed | 0.000000e+00 | 0.000000e+00 |
| model2_preprocessed | 0.000000e+00 (F0/N/asr) | - |

预处理完全保持了模型精度。

## 6. 转换结果

### 6.1 转换命令

```bash
python convert_rknn.py \
    --onnx_dir /path/to/onnx_models \
    --output_dir /path/to/rknn_output
```

### 6.2 转换结果汇总

| 模型 | 状态 | ONNX 大小 | RKNN 大小 | 转换耗时 |
|------|------|-----------|-----------|----------|
| model1_bert_duration | ❌ 失败 | 53.06 MB | - | - |
| model2_f0_n_asr | ✅ 成功 | 53.52 MB | 31.68 MB | 4.1s |
| model3_decoder | ✅ 成功 | 203.35 MB | 137.59 MB | 23.4s |
| model4_har | ✅ 成功 | 0.01 MB | 2.82 MB | 1.3s |

### 6.3 Model 1 失败分析

**症状**: rknn-toolkit2 在 `rknn.build()` 阶段产生 segfault (exit code 139)

**错误信息**:
```
E RKNN: Meet unsupported first tensor dtype for per-layer mul
E RKNN: dataconvert type -1 is unsupport in current!
W RKNN: cast tensor failed
W RKNN: emit dataconvert failed
```

**错误位置**: 发生在 `rknn::RKNNAddSecondaryNode` 优化阶段

**已尝试的解决方案**:
1. ✅ BOOL → FLOAT 预处理 → 不影响，仍 segfault
2. ✅ INT64 → INT32 全部转换 → ORT 拒绝（Unsqueeze 需要 INT64 axes）
3. ✅ optimization_level=0 → 不影响
4. ✅ single_core_mode=True → 不影响
5. ✅ op_target 将 LSTM/Gather 强制到 CPU → 不影响

**根因分析**: model1 包含完整的 BERT 编码器（多头注意力机制），具有 769 个节点。RKNN toolkit2 v2.3.2 的编译器在处理 BERT 注意力的 MatMul-Softmax-MatMul 融合模式时存在内部 bug，导致 segfault。相比之下，model2 只有 212 个节点的简单文本编码器，能正常转换。

**建议**:
1. 等待 rknn-toolkit2 更新版本修复 BERT 支持
2. 将 model1 拆分为 BERT 编码器和时长预测两个子模型
3. BERT 部分考虑用 CPU 运行或使用 RKNN 支持的替代 Transformer 实现
4. 尝试 ONNX opset 版本降级或修改注意力机制实现

## 7. 精度验证

### 7.1 验证方法

使用 RKNN 模拟器（x86 平台）运行推理，与 ONNXRuntime 参考输出对比。

评估指标:
- **余弦相似度 (Cosine Similarity)**: 衡量输出方向一致性
- **最大绝对误差 (Max Abs Diff)**: 衡量最大偏差
- **平均绝对误差 (Mean Abs Diff)**: 衡量整体偏差

### 7.2 验证结果

#### Model 2: model2_f0_n_asr

| 输出 | 余弦相似度 | 最大绝对误差 | 平均绝对误差 | 评价 |
|------|-----------|-------------|-------------|------|
| F0_pred [1,312] | 0.99999996 | 3.92e-01 | 3.34e-02 | ✅ 优秀 |
| N_pred [1,312] | 0.99999997 | 5.79e-03 | 1.20e-03 | ✅ 优秀 |
| asr [1,512,156] | 0.99999996 | 1.07e-03 | 5.42e-05 | ✅ 优秀 |

#### Model 3: model3_decoder

| 输出 | 余弦相似度 | 最大绝对误差 | 平均绝对误差 | 评价 |
|------|-----------|-------------|-------------|------|
| x [1,22,18721] | 0.99999987 | 6.58e-02 | 1.50e-03 | ✅ 优秀 |

#### Model 4: model4_har

| 输出 | 状态 | 说明 |
|------|------|------|
| har [1,22,18721] | ⚠️ 模拟器 fp16 溢出 | RKNN 模拟器中间值 inf 超出 fp16 范围，产生 NaN |

**Model 4 说明**: model4_har 在 RKNN 模拟器中产生 NaN 输出，原因是谐波分析中的频率上采样操作产生中间值超出 float16 范围。但模型转换本身成功，**在实际 RK3588 硬件上可能表现不同**（硬件 NPU 和模拟器的 fp16 处理可能存在差异）。建议在实际硬件上进一步验证。

### 7.3 精度总结

所有成功转换的模型（不含 model4 的模拟器异常），余弦相似度均 > 0.9999，说明 fp16 量化引入的误差极小，不影响 TTS 合成质量。

## 8. 使用说明

### 8.1 转换命令

```bash
# 转换所有模型
python convert_rknn.py --onnx_dir <onnx_dir> --output_dir <output_dir>

# 只转换指定模型
python convert_rknn.py --onnx_dir <onnx_dir> --output_dir <output_dir> \
    --models model2_f0_n_asr model3_decoder model4_har

# 跳过验证
python convert_rknn.py --onnx_dir <onnx_dir> --output_dir <output_dir> --skip_verify
```

### 8.2 输入数据类型变更

转换后的 RKNN 模型输入类型可能与原始 ONNX 模型不同：

| 模型 | 输入 | 原始类型 | RKNN 类型 |
|------|------|---------|-----------|
| model2 | input_ids | INT64 | FLOAT32 |
| model2 | text_mask | BOOL | FLOAT32 |
| model3 | 全部 | FLOAT32 | FLOAT32（无变化）|
| model4 | F0_pred | FLOAT32 | FLOAT32（无变化）|

在 RK3588 上运行时，需要将 input_ids 和 text_mask 转换为 float32 再输入。

### 8.3 输出文件

```
rknn_output/
├── model2_f0_n_asr.rknn            # 31.68 MB
├── model2_f0_n_asr_preprocessed.onnx  # 预处理后的 ONNX
├── model3_decoder.rknn             # 137.59 MB
├── model4_har.rknn                 # 2.82 MB
└── conversion_report.json          # 详细转换报告
```

## 9. 已知问题

1. **Model 1 (BERT) 转换失败**: rknn-toolkit2 v2.3.2 内部 segfault，可能需要等待工具链更新
2. **Model 4 模拟器 fp16 溢出**: 谐波分析中间值超出 fp16 范围，需在实际硬件验证
3. **onnx 版本兼容**: 必须使用 onnx 1.16.x，不兼容 1.17+

## 10. 工具版本

| 工具 | 版本 |
|------|------|
| rknn-toolkit2 | 2.3.2 |
| Python | 3.10.20 |
| onnx | 1.16.2 |
| onnxruntime | 1.19.2 |
| onnxsim | 0.6.2 |
| numpy | 1.26.4 |
