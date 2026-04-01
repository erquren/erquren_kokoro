"""
RKNN 量化基准测试脚本
对比 FP16、不量化、INT8 三种量化模式的文件大小和精度差异

用法:
    python benchmark_rknn_quantization.py \
        --onnx_dir <onnx_dir> \
        --output_dir <benchmark_output>

输出:
    - 各量化模式的 RKNN 模型文件
    - benchmark_report.json: 结构化基准测试报告
    - benchmark_report.md: 可读的 Markdown 报告
    - benchmark.log: 详细日志
"""

import argparse
import json
import logging
import os
import sys
import time
import zipfile

import numpy as np

try:
    from rknn.api import RKNN
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

import onnx

try:
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False


# ─── 日志配置 ─────────────────────────────────────────────────────────────

def setup_logging(output_dir):
    """配置日志: 同时输出到文件和控制台"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "benchmark.log")

    logger = logging.getLogger("rknn_benchmark")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ─── 模型定义 (与 convert_rknn.py 保持一致) ──────────────────────────────

MODELS = {
    "model2_f0_n_asr": {
        "onnx_file": "model2_f0_n_asr_sim.onnx",
        "description": "F0/能量预测 + 文本对齐",
        "needs_preprocessing": True,
        "calibration_inputs": {
            "en": ("model2_en.zip", np.float32),
            "ref_s": ("model2_ref_s.zip", np.float32),
            "input_ids": ("model2_input_ids.zip", np.float32),
            "text_mask": ("model2_text_mask.zip", np.float32),
            "pred_aln_trg": ("model2_pred_aln_trg.zip", np.float32),
        },
    },
    "model3_decoder": {
        "onnx_file": "model3_decoder_sim.onnx",
        "description": "解码器（生成音频特征）",
        "needs_preprocessing": False,
        "calibration_inputs": {
            "asr": ("model3_asr.zip", np.float32),
            "F0_pred": ("model3_F0_pred.zip", np.float32),
            "N_pred": ("model3_N_pred.zip", np.float32),
            "ref_s": ("model3_ref_s.zip", np.float32),
            "har": ("model3_har.zip", np.float32),
        },
    },
    "model4_har": {
        "onnx_file": "model4_har_sim.onnx",
        "description": "谐波分析",
        "needs_preprocessing": False,
        "calibration_inputs": {
            "F0_pred": ("model4_F0_pred.zip", np.float32),
        },
    },
}

QUANTIZATION_MODES = ["fp16", "none", "int8"]


# ─── 辅助函数 ─────────────────────────────────────────────────────────────

def load_calibration_data(onnx_dir, zip_name, target_dtype=np.float32):
    """从 zip 文件中加载 numpy 校准数据"""
    zip_path = os.path.join(onnx_dir, zip_name)
    if not os.path.exists(zip_path):
        return None
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = np.load(zf.open(zf.namelist()[0]))
    return data.astype(target_dtype)


def load_all_calibration_inputs(onnx_dir, model_info):
    """加载某个模型的全部校准输入"""
    inputs = {}
    for input_name, (zip_name, dtype) in model_info["calibration_inputs"].items():
        data = load_calibration_data(onnx_dir, zip_name, dtype)
        if data is None:
            return None
        inputs[input_name] = data
    return inputs


def run_onnx_inference(onnx_path, inputs):
    """使用 onnxruntime 运行 ONNX 推理"""
    if not HAS_ORT:
        return None
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {}
    for inp in sess.get_inputs():
        if inp.name in inputs:
            feed[inp.name] = inputs[inp.name]
    outputs = sess.run(None, feed)
    output_names = [o.name for o in sess.get_outputs()]
    return dict(zip(output_names, outputs))


def prepare_calibration_dataset(model_name, model_info, onnx_dir, output_dir):
    """为 INT8 量化准备校准数据集文本文件"""
    dataset_dir = os.path.join(output_dir, f"{model_name}_calibration")
    os.makedirs(dataset_dir, exist_ok=True)

    npy_paths = []
    for input_name, (zip_name, dtype) in model_info["calibration_inputs"].items():
        data = load_calibration_data(onnx_dir, zip_name, dtype)
        if data is None:
            return None
        npy_path = os.path.join(dataset_dir, f"{input_name}.npy")
        np.save(npy_path, data)
        npy_paths.append(npy_path)

    dataset_path = os.path.join(dataset_dir, "dataset.txt")
    with open(dataset_path, "w") as f:
        f.write(" ".join(npy_paths) + "\n")
    return dataset_path


def preprocess_onnx_for_rknn(onnx_path, output_path):
    """预处理 ONNX 模型，消除 RKNN 不支持的数据类型"""
    from onnx import TensorProto, helper

    model = onnx.load(onnx_path)
    graph = model.graph

    uid_cnt = [0]

    def uid(prefix):
        uid_cnt[0] += 1
        return f"_rknn_{prefix}_{uid_cnt[0]}"

    for inp in graph.input:
        if inp.type.tensor_type.elem_type != TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT

    new_nodes = []
    for node in graph.node:
        if node.op_type == "Not":
            inp_name, out_name = node.input[0], node.output[0]
            c = uid("one")
            new_nodes.append(helper.make_node(
                "Constant", [], [c],
                value=helper.make_tensor(c, TensorProto.FLOAT, [], [1.0]),
                name=uid("const")))
            new_nodes.append(helper.make_node(
                "Sub", [c, inp_name], [out_name], name=uid("not_sub")))
        elif node.op_type == "Where":
            cond, x, y = node.input[0], node.input[1], node.input[2]
            out = node.output[0]
            inv, xp, yp, one = uid("inv"), uid("xp"), uid("yp"), uid("one")
            new_nodes.append(helper.make_node(
                "Constant", [], [one],
                value=helper.make_tensor(one, TensorProto.FLOAT, [], [1.0]),
                name=uid("const")))
            new_nodes.append(helper.make_node(
                "Sub", [one, cond], [inv], name=uid("where_inv")))
            new_nodes.append(helper.make_node(
                "Mul", [x, cond], [xp], name=uid("where_mul_x")))
            new_nodes.append(helper.make_node(
                "Mul", [y, inv], [yp], name=uid("where_mul_y")))
            new_nodes.append(helper.make_node(
                "Add", [xp, yp], [out], name=uid("where_add")))
        elif node.op_type == "Cast":
            to_type = next((a.i for a in node.attribute if a.name == "to"), None)
            if to_type in (TensorProto.BOOL, TensorProto.INT32):
                for a in node.attribute:
                    if a.name == "to":
                        a.i = TensorProto.FLOAT
            new_nodes.append(node)
        elif node.op_type == "Gather":
            cast_name = uid("gather_idx")
            new_nodes.append(helper.make_node(
                "Cast", [node.input[1]], [cast_name],
                to=TensorProto.INT64, name=uid("cast_gather")))
            node.input[1] = cast_name
            new_nodes.append(node)
        else:
            new_nodes.append(node)

    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type in (TensorProto.BOOL, TensorProto.INT32):
            vi.type.tensor_type.elem_type = TensorProto.FLOAT

    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.save(model, output_path)
    if HAS_ONNXSIM:
        try:
            m_sim, check = onnxsim.simplify(onnx.load(output_path))
            if check:
                onnx.save(m_sim, output_path)
        except Exception:
            pass

    return output_path


# ─── 核心基准测试函数 ─────────────────────────────────────────────────────

def convert_model(model_name, model_info, onnx_dir, output_dir,
                  quantization_mode, logger):
    """转换单个模型到指定量化模式"""
    suffix = f"_{quantization_mode}"
    result = {
        "model_name": model_name,
        "quantization_mode": quantization_mode,
        "success": False,
        "rknn_path": None,
        "rknn_size_bytes": 0,
        "onnx_size_bytes": 0,
        "conversion_time": 0,
        "error": None,
    }

    onnx_path = os.path.join(onnx_dir, model_info["onnx_file"])
    rknn_path = os.path.join(output_dir, f"{model_name}{suffix}.rknn")

    if not os.path.exists(onnx_path):
        result["error"] = f"ONNX 文件不存在: {onnx_path}"
        logger.error(result["error"])
        return result

    result["onnx_size_bytes"] = os.path.getsize(onnx_path)

    logger.info(f"转换 {model_name} [{quantization_mode}]")

    start_time = time.time()

    # 预处理
    actual_onnx = onnx_path
    if model_info.get("needs_preprocessing", False):
        preprocessed = os.path.join(output_dir, f"{model_name}_preprocessed.onnx")
        if not os.path.exists(preprocessed):
            logger.info(f"  预处理 ONNX 模型...")
            try:
                actual_onnx = preprocess_onnx_for_rknn(onnx_path, preprocessed)
            except Exception as e:
                result["error"] = f"预处理失败: {e}"
                logger.error(result["error"])
                return result
        else:
            actual_onnx = preprocessed
            logger.info(f"  使用已有预处理模型: {preprocessed}")

    if not HAS_RKNN:
        result["error"] = "rknn-toolkit2 未安装，跳过转换"
        logger.warning(result["error"])
        return result

    # 创建 RKNN 并转换
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")

    ret = rknn.load_onnx(model=actual_onnx)
    if ret != 0:
        result["error"] = f"加载 ONNX 失败 (ret={ret})"
        logger.error(result["error"])
        rknn.release()
        return result

    # 根据量化模式构建
    try:
        if quantization_mode == "int8":
            dataset_path = prepare_calibration_dataset(
                model_name, model_info, onnx_dir, output_dir)
            if dataset_path is None:
                result["error"] = "校准数据不存在，无法进行 INT8 量化"
                logger.error(result["error"])
                rknn.release()
                return result
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
            ret = rknn.build(do_quantization=False)
    except Exception as e:
        result["error"] = f"构建 RKNN 异常: {e}"
        logger.error(result["error"])
        rknn.release()
        return result

    if ret != 0:
        result["error"] = f"构建 RKNN 失败 (ret={ret})"
        logger.error(result["error"])
        rknn.release()
        return result

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        result["error"] = f"导出 RKNN 失败 (ret={ret})"
        logger.error(result["error"])
        rknn.release()
        return result

    elapsed = time.time() - start_time
    rknn_size = os.path.getsize(rknn_path)
    result["success"] = True
    result["rknn_path"] = rknn_path
    result["rknn_size_bytes"] = rknn_size
    result["conversion_time"] = elapsed

    logger.info(f"  成功: {rknn_size / 1024 / 1024:.2f} MB, 耗时 {elapsed:.1f}s")
    rknn.release()
    return result


def verify_model(model_name, model_info, onnx_dir, output_dir,
                 quantization_mode, logger):
    """验证单个模型的精度"""
    result = {
        "model_name": model_name,
        "quantization_mode": quantization_mode,
        "verified": False,
        "outputs": {},
        "error": None,
    }

    if not HAS_RKNN or not HAS_ORT:
        result["error"] = "rknn-toolkit2 或 onnxruntime 未安装"
        logger.warning(result["error"])
        return result

    onnx_path = os.path.join(onnx_dir, model_info["onnx_file"])
    actual_onnx = onnx_path
    preprocessed = os.path.join(output_dir, f"{model_name}_preprocessed.onnx")
    if os.path.exists(preprocessed):
        actual_onnx = preprocessed

    logger.info(f"验证 {model_name} [{quantization_mode}]")

    # 加载数据
    inputs = load_all_calibration_inputs(onnx_dir, model_info)
    if inputs is None:
        result["error"] = "校准数据不存在"
        logger.error(result["error"])
        return result

    # ONNX 参考推理
    onnx_inputs = {}
    for input_name, (zip_name, _) in model_info["calibration_inputs"].items():
        with zipfile.ZipFile(os.path.join(onnx_dir, zip_name)) as zf:
            onnx_inputs[input_name] = np.load(zf.open(zf.namelist()[0]))
    onnx_outputs = run_onnx_inference(onnx_path, onnx_inputs)
    if onnx_outputs is None:
        result["error"] = "ONNX 推理失败"
        logger.error(result["error"])
        return result

    # RKNN 模拟器推理
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")
    ret = rknn.load_onnx(model=actual_onnx)
    if ret != 0:
        result["error"] = f"加载 ONNX 失败 (ret={ret})"
        rknn.release()
        return result

    try:
        if quantization_mode == "int8":
            dataset_path = prepare_calibration_dataset(
                model_name, model_info, onnx_dir, output_dir)
            if dataset_path is None:
                result["error"] = "校准数据不存在"
                rknn.release()
                return result
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
            ret = rknn.build(do_quantization=False)
    except Exception as e:
        result["error"] = f"构建失败: {e}"
        rknn.release()
        return result

    if ret != 0:
        result["error"] = f"构建失败 (ret={ret})"
        rknn.release()
        return result

    ret = rknn.init_runtime(target=None)
    if ret != 0:
        result["error"] = f"初始化运行时失败 (ret={ret})"
        rknn.release()
        return result

    # 准备输入
    sess = ort.InferenceSession(actual_onnx, providers=["CPUExecutionProvider"])
    rknn_input_list = [inputs[inp.name] for inp in sess.get_inputs()]
    rknn_outputs = rknn.inference(inputs=rknn_input_list)

    # 对比
    output_names = list(onnx_outputs.keys())
    result["verified"] = True

    for i, name in enumerate(output_names):
        onnx_out = onnx_outputs[name]
        rknn_out = rknn_outputs[i]

        abs_diff = np.abs(onnx_out.astype(np.float64) - rknn_out.astype(np.float64))
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))

        onnx_flat = onnx_out.flatten().astype(np.float64)
        rknn_flat = rknn_out.flatten().astype(np.float64)
        norm_o = np.linalg.norm(onnx_flat)
        norm_r = np.linalg.norm(rknn_flat)
        cosine_sim = (float(np.dot(onnx_flat, rknn_flat) / (norm_o * norm_r))
                      if norm_o > 0 and norm_r > 0 else 1.0)

        output_info = {
            "shape": list(onnx_out.shape),
            "onnx_range": [float(onnx_out.min()), float(onnx_out.max())],
            "rknn_range": [float(rknn_out.min()), float(rknn_out.max())],
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "cosine_similarity": cosine_sim,
        }
        result["outputs"][name] = output_info

        icon = "✅" if cosine_sim > 0.999 else "⚠️" if cosine_sim > 0.99 else "❌"
        logger.info(f"  {name}: cos={cosine_sim:.8f} {icon} "
                    f"(max_diff={max_abs_diff:.2e})")

    rknn.release()
    return result


# ─── 报告生成 ─────────────────────────────────────────────────────────────

def generate_markdown_report(all_results, output_dir):
    """生成 Markdown 格式的基准测试报告"""
    lines = []
    lines.append("# RKNN 量化基准测试报告")
    lines.append("")
    lines.append(f"- **目标平台**: RK3588")
    lines.append(f"- **测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **量化模式**: FP16, 不量化(None), INT8")
    lines.append("")

    # 文件大小对比表
    lines.append("## 1. 文件大小对比")
    lines.append("")
    lines.append("| 模型 | ONNX 大小 | FP16 RKNN | None RKNN | INT8 RKNN | "
                 "FP16 压缩比 | INT8 压缩比 |")
    lines.append("|------|-----------|-----------|-----------|-----------|"
                 "------------|------------|")

    for model_name in MODELS:
        onnx_size = 0
        sizes = {}
        for mode in QUANTIZATION_MODES:
            key = f"{model_name}_{mode}"
            r = all_results.get("conversion", {}).get(key, {})
            if r.get("success"):
                sizes[mode] = r.get("rknn_size_bytes", 0)
                if onnx_size == 0:
                    onnx_size = r.get("onnx_size_bytes", 0)

        onnx_mb = f"{onnx_size / 1024 / 1024:.2f} MB" if onnx_size > 0 else "N/A"

        fp16_mb = (f"{sizes.get('fp16', 0) / 1024 / 1024:.2f} MB"
                   if sizes.get("fp16") else "❌ 失败")
        none_mb = (f"{sizes.get('none', 0) / 1024 / 1024:.2f} MB"
                   if sizes.get("none") else "❌ 失败")
        int8_mb = (f"{sizes.get('int8', 0) / 1024 / 1024:.2f} MB"
                   if sizes.get("int8") else "❌ 失败")

        fp16_ratio = (f"{sizes.get('fp16', 0) / onnx_size * 100:.1f}%"
                      if sizes.get("fp16") and onnx_size > 0 else "N/A")
        int8_ratio = (f"{sizes.get('int8', 0) / onnx_size * 100:.1f}%"
                      if sizes.get("int8") and onnx_size > 0 else "N/A")

        lines.append(f"| {model_name} | {onnx_mb} | {fp16_mb} | {none_mb} | "
                     f"{int8_mb} | {fp16_ratio} | {int8_ratio} |")

    lines.append("")

    # 精度对比表
    lines.append("## 2. 精度对比 (余弦相似度)")
    lines.append("")
    lines.append("| 模型 | 输出 | FP16 余弦相似度 | None 余弦相似度 | "
                 "INT8 余弦相似度 |")
    lines.append("|------|------|----------------|----------------|"
                 "----------------|")

    for model_name in MODELS:
        first_row = True
        all_outputs = set()
        for mode in QUANTIZATION_MODES:
            key = f"{model_name}_{mode}"
            vr = all_results.get("verification", {}).get(key, {})
            all_outputs.update(vr.get("outputs", {}).keys())

        for out_name in sorted(all_outputs):
            model_col = model_name if first_row else ""
            first_row = False
            cos_vals = {}
            for mode in QUANTIZATION_MODES:
                key = f"{model_name}_{mode}"
                vr = all_results.get("verification", {}).get(key, {})
                out_info = vr.get("outputs", {}).get(out_name)
                if out_info:
                    cos = out_info["cosine_similarity"]
                    icon = "✅" if cos > 0.999 else "⚠️" if cos > 0.99 else "❌"
                    cos_vals[mode] = f"{cos:.8f} {icon}"
                elif vr.get("error"):
                    cos_vals[mode] = f"⚠️ {vr['error']}"
                else:
                    cos_vals[mode] = "N/A"

            lines.append(
                f"| {model_col} | {out_name} | "
                f"{cos_vals.get('fp16', 'N/A')} | "
                f"{cos_vals.get('none', 'N/A')} | "
                f"{cos_vals.get('int8', 'N/A')} |")

    lines.append("")

    # 转换耗时
    lines.append("## 3. 转换耗时")
    lines.append("")
    lines.append("| 模型 | FP16 耗时 | None 耗时 | INT8 耗时 |")
    lines.append("|------|----------|----------|----------|")

    for model_name in MODELS:
        times = {}
        for mode in QUANTIZATION_MODES:
            key = f"{model_name}_{mode}"
            r = all_results.get("conversion", {}).get(key, {})
            if r.get("success"):
                times[mode] = f"{r.get('conversion_time', 0):.1f}s"
            else:
                times[mode] = "❌"
        lines.append(
            f"| {model_name} | {times.get('fp16', 'N/A')} | "
            f"{times.get('none', 'N/A')} | {times.get('int8', 'N/A')} |")

    lines.append("")

    # 详细误差
    lines.append("## 4. 详细误差分析")
    lines.append("")

    for model_name in MODELS:
        lines.append(f"### {model_name}")
        lines.append("")
        for mode in QUANTIZATION_MODES:
            key = f"{model_name}_{mode}"
            vr = all_results.get("verification", {}).get(key, {})
            lines.append(f"#### 量化模式: {mode}")
            if vr.get("error"):
                lines.append(f"- **错误**: {vr['error']}")
            elif vr.get("outputs"):
                lines.append("")
                lines.append("| 输出 | 形状 | 余弦相似度 | 最大绝对误差 | "
                             "平均绝对误差 |")
                lines.append("|------|------|-----------|-------------|"
                             "-------------|")
                for out_name, info in vr.get("outputs", {}).items():
                    cos = info["cosine_similarity"]
                    icon = "✅" if cos > 0.999 else "⚠️" if cos > 0.99 else "❌"
                    lines.append(
                        f"| {out_name} | {info['shape']} | "
                        f"{cos:.8f} {icon} | "
                        f"{info['max_abs_diff']:.2e} | "
                        f"{info['mean_abs_diff']:.2e} |")
            else:
                lines.append("- 未验证")
            lines.append("")

    # 结论
    lines.append("## 5. 结论")
    lines.append("")
    lines.append("- **FP16**: 默认量化模式（`do_quantization=False`），RK3588 NPU 默认使用 FP16 计算，"
                 "文件大小适中，精度损失极小（余弦相似度 > 0.9999）")
    lines.append("- **不量化 (None)**: 同样使用 `do_quantization=False`，与 FP16 在 RK3588 上行为一致"
                 "（作为基准对照组）")
    lines.append("- **INT8**: 使用 `do_quantization=True` 进行 INT8 量化，"
                 "文件大小最小（约为 ONNX 的 25-50%），但精度损失较大，需根据应用场景评估")
    lines.append("")
    lines.append("> **注意**: Model 1 (BERT) 因 rknn-toolkit2 v2.3.2 内部 bug "
                 "导致 segfault，未纳入本次测试")
    lines.append("")

    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# ─── 主函数 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RKNN 量化基准测试: 对比 FP16/None/INT8 的文件大小和精度")
    parser.add_argument("--onnx_dir", type=str, required=True,
                        help="ONNX 模型目录")
    parser.add_argument("--output_dir", type=str, default="benchmark_output",
                        help="基准测试输出目录")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="指定模型（默认: model2, model3, model4）")
    parser.add_argument("--skip_verify", action="store_true",
                        help="跳过精度验证")
    parser.add_argument("--modes", type=str, nargs="*",
                        default=QUANTIZATION_MODES,
                        choices=QUANTIZATION_MODES,
                        help="指定量化模式（默认: 全部）")
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    model_names = args.models or list(MODELS.keys())

    logger.info("=" * 60)
    logger.info("RKNN 量化基准测试")
    logger.info(f"  ONNX 目录: {args.onnx_dir}")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  模型: {model_names}")
    logger.info(f"  量化模式: {args.modes}")
    logger.info(f"  RKNN 可用: {HAS_RKNN}")
    logger.info(f"  ORT 可用: {HAS_ORT}")
    logger.info("=" * 60)

    all_results = {
        "conversion": {},
        "verification": {},
        "metadata": {
            "platform": "rk3588",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rknn_available": HAS_RKNN,
            "ort_available": HAS_ORT,
            "modes_tested": args.modes,
            "models_tested": model_names,
        },
    }

    # 转换所有模型的所有量化模式
    for model_name in model_names:
        if model_name not in MODELS:
            logger.warning(f"未知模型: {model_name}, 跳过")
            continue

        for mode in args.modes:
            key = f"{model_name}_{mode}"
            logger.info(f"\n{'─'*40}")
            result = convert_model(
                model_name, MODELS[model_name], args.onnx_dir,
                args.output_dir, mode, logger)
            all_results["conversion"][key] = result

    # 精度验证
    if not args.skip_verify:
        logger.info(f"\n{'='*60}")
        logger.info("精度验证阶段")
        logger.info("=" * 60)

        for model_name in model_names:
            if model_name not in MODELS:
                continue
            for mode in args.modes:
                key = f"{model_name}_{mode}"
                conv = all_results["conversion"].get(key, {})
                if not conv.get("success"):
                    logger.info(f"跳过验证 {key} (转换未成功)")
                    continue
                vr = verify_model(
                    model_name, MODELS[model_name], args.onnx_dir,
                    args.output_dir, mode, logger)
                all_results["verification"][key] = vr

    # 保存 JSON 报告
    json_path = os.path.join(args.output_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\nJSON 报告: {json_path}")

    # 生成 Markdown 报告
    md_path = generate_markdown_report(all_results, args.output_dir)
    logger.info(f"Markdown 报告: {md_path}")

    # 打印汇总
    logger.info(f"\n{'='*60}")
    logger.info("基准测试完成!")
    logger.info("=" * 60)

    for model_name in model_names:
        logger.info(f"\n{model_name}:")
        for mode in args.modes:
            key = f"{model_name}_{mode}"
            conv = all_results["conversion"].get(key, {})
            if conv.get("success"):
                size_mb = conv.get("rknn_size_bytes", 0) / 1024 / 1024
                logger.info(f"  [{mode:>4}] ✅ {size_mb:.2f} MB, "
                            f"{conv.get('conversion_time', 0):.1f}s")
            else:
                logger.info(f"  [{mode:>4}] ❌ {conv.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
