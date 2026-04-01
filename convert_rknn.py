"""
ONNX → RKNN 模型转换脚本
将 Kokoro TTS 的 4 个 ONNX 子模型转换为 RK3588 可运行的 RKNN 格式

用法:
    # 需要 Python 3.10 + rknn-toolkit2 2.3.2
    python convert_rknn.py --onnx_dir <onnx_dir> --output_dir <rknn_output_dir>

依赖:
    - rknn-toolkit2 >= 2.3.2 (cp310)
    - numpy, onnx, onnxruntime, onnxsim
"""

import argparse
import os
import sys
import json
import time
import zipfile
import shutil
import numpy as np

# ─── 导入依赖 ────────────────────────────────────────────────────────────
try:
    from rknn.api import RKNN
except ImportError:
    print("ERROR: rknn-toolkit2 未安装")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import onnx
from onnx import TensorProto, helper

try:
    import onnxsim
except ImportError:
    onnxsim = None


# ─── 模型定义 ────────────────────────────────────────────────────────────
MODELS = {
    "model1_bert_duration": {
        "onnx_file": "model1_bert_duration_sim.onnx",
        "description": "BERT 编码 + 时长预测",
        "needs_preprocessing": True,
        "calibration_inputs": {
            "input_ids": ("model1_input_ids.zip", np.float32),
            "ref_s": ("model1_ref_s.zip", np.float32),
            "text_mask": ("model1_text_mask.zip", np.float32),
        },
    },
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


# ─── ONNX 预处理 ─────────────────────────────────────────────────────────

def preprocess_onnx_for_rknn(onnx_path, output_path):
    """
    预处理 ONNX 模型，消除 RKNN 不支持的数据类型。

    RKNN NPU 不支持 BOOL 和 INT64 类型的计算节点。
    本函数将模型中所有 BOOL/INT64 的输入转为 FLOAT，
    并替换 Not/Where 等 BOOL 操作为等价的浮点算术操作。

    具体操作:
      - 所有 non-FLOAT 输入 → FLOAT
      - Not(x) → Sub(1.0, x)
      - Where(cond, x, y) → x*cond + y*(1-cond)
      - Cast(to=BOOL) → Cast(to=FLOAT)
      - Cast(to=INT32) → Cast(to=FLOAT)
      - Gather 索引: 插入 Cast(FLOAT→INT64)
      - 所有 BOOL/INT32 value_info → FLOAT
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    uid_cnt = [0]

    def uid(prefix):
        uid_cnt[0] += 1
        return f"_rknn_{prefix}_{uid_cnt[0]}"

    # Step 1: 修改输入类型
    for inp in graph.input:
        if inp.type.tensor_type.elem_type != TensorProto.FLOAT:
            old_name = TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
            print(f"  Input {inp.name}: {old_name} → FLOAT")

    # Step 2: 替换节点
    new_nodes = []
    for node in graph.node:
        if node.op_type == "Not":
            # Not(x) → Sub(1.0, x)
            inp_name, out_name = node.input[0], node.output[0]
            c = uid("one")
            new_nodes.append(helper.make_node(
                "Constant", [], [c],
                value=helper.make_tensor(c, TensorProto.FLOAT, [], [1.0]),
                name=uid("const")))
            new_nodes.append(helper.make_node(
                "Sub", [c, inp_name], [out_name], name=uid("not_sub")))
            print(f"  Not({inp_name}) → Sub(1.0, x)")

        elif node.op_type == "Where":
            # Where(cond, x, y) → x * cond + y * (1 - cond)
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
            print(f"  Where({cond}, ...) → arithmetic")

        elif node.op_type == "Cast":
            to_type = next((a.i for a in node.attribute if a.name == "to"), None)
            if to_type in (TensorProto.BOOL, TensorProto.INT32):
                for a in node.attribute:
                    if a.name == "to":
                        a.i = TensorProto.FLOAT
                print(f"  Cast({node.input[0]}) → FLOAT (was {TensorProto.DataType.Name(to_type)})")
            new_nodes.append(node)

        elif node.op_type == "Gather":
            # Gather 索引必须为 INT，插入 Cast(FLOAT→INT64)
            cast_name = uid("gather_idx")
            new_nodes.append(helper.make_node(
                "Cast", [node.input[1]], [cast_name],
                to=TensorProto.INT64, name=uid("cast_gather")))
            node.input[1] = cast_name
            new_nodes.append(node)
            print(f"  Gather: Cast({node.input[1]}→INT64) for indices")

        else:
            new_nodes.append(node)

    # Step 3: 更新 value_info
    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type in (TensorProto.BOOL, TensorProto.INT32):
            vi.type.tensor_type.elem_type = TensorProto.FLOAT

    # 替换节点
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Step 4: 简化
    onnx.save(model, output_path)
    if onnxsim is not None:
        try:
            m_sim, check = onnxsim.simplify(onnx.load(output_path))
            if check:
                onnx.save(m_sim, output_path)
                print(f"  onnxsim: {len(model.graph.node)} → {len(m_sim.graph.node)} nodes")
        except Exception as e:
            print(f"  onnxsim warning: {e}")

    print(f"  Saved: {output_path}")
    return output_path


# ─── 辅助函数 ─────────────────────────────────────────────────────────────

def load_calibration_data(onnx_dir, zip_name, target_dtype=np.float32):
    """从 zip 文件中加载 numpy 校准数据"""
    zip_path = os.path.join(onnx_dir, zip_name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = np.load(zf.open(zf.namelist()[0]))
    return data.astype(target_dtype)


def load_all_calibration_inputs(onnx_dir, model_info):
    """加载某个模型的全部校准输入"""
    inputs = {}
    for input_name, (zip_name, dtype) in model_info["calibration_inputs"].items():
        inputs[input_name] = load_calibration_data(onnx_dir, zip_name, dtype)
    return inputs


def run_onnx_inference(onnx_path, inputs):
    """使用 onnxruntime 运行 ONNX 推理"""
    if ort is None:
        return None
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {}
    for inp in sess.get_inputs():
        feed[inp.name] = inputs[inp.name]
    outputs = sess.run(None, feed)
    output_names = [o.name for o in sess.get_outputs()]
    return dict(zip(output_names, outputs))


# ─── 转换 ────────────────────────────────────────────────────────────────

def convert_single_model(model_name, model_info, onnx_dir, output_dir,
                         quantization_mode="fp16", verbose=False):
    """
    转换单个 ONNX 模型为 RKNN 格式。

    参数:
        quantization_mode: 量化模式
            - "fp16": FP16 量化（默认，do_quantization=False）
            - "none": 不量化，保持原始精度
            - "int8": INT8 量化（需要校准数据）
    """
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
        print(f"[ERROR] {result['error']}")
        return result

    result["onnx_size_bytes"] = os.path.getsize(onnx_path)

    print(f"\n{'='*60}")
    print(f"转换模型: {model_name} ({model_info['description']})")
    print(f"量化模式: {quantization_mode}")
    print(f"{'='*60}")

    start_time = time.time()

    # Step 1: 预处理（如需要）
    actual_onnx = onnx_path
    if model_info.get("needs_preprocessing", False):
        print("[Step 1] 预处理 ONNX 模型...")
        preprocessed_path = os.path.join(output_dir, f"{model_name}_preprocessed.onnx")
        try:
            actual_onnx = preprocess_onnx_for_rknn(onnx_path, preprocessed_path)
        except Exception as e:
            result["error"] = f"预处理失败: {e}"
            print(f"[ERROR] {result['error']}")
            return result
    else:
        print("[Step 1] 无需预处理")

    # Step 2: RKNN 转换
    print("[Step 2] 创建 RKNN 对象并配置...")
    rknn = RKNN(verbose=verbose)
    rknn.config(target_platform="rk3588")

    print("[Step 3] 加载 ONNX 模型...")
    ret = rknn.load_onnx(model=actual_onnx)
    if ret != 0:
        result["error"] = f"加载 ONNX 模型失败 (ret={ret})"
        print(f"[ERROR] {result['error']}")
        rknn.release()
        return result

    # Step 4: 构建 RKNN 模型（根据量化模式选择参数）
    if quantization_mode == "int8":
        print("[Step 4] 构建 RKNN 模型 (do_quantization=True, INT8)...")
        # 准备校准数据集
        dataset_path = _prepare_calibration_dataset(
            model_name, model_info, onnx_dir, output_dir)
        try:
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        except Exception as e:
            result["error"] = f"构建 RKNN 模型异常 (INT8): {e}"
            print(f"[ERROR] {result['error']}")
            rknn.release()
            return result
    elif quantization_mode == "none":
        # "none" 与 "fp16" 在 RK3588 上行为一致，因为 NPU 默认使用 FP16 计算。
        # 两者都使用 do_quantization=False，保留作为基准对照。
        print("[Step 4] 构建 RKNN 模型 (do_quantization=False, 不量化)...")
        try:
            ret = rknn.build(do_quantization=False)
        except Exception as e:
            result["error"] = f"构建 RKNN 模型异常 (none): {e}"
            print(f"[ERROR] {result['error']}")
            rknn.release()
            return result
    else:
        # fp16 (默认)
        print("[Step 4] 构建 RKNN 模型 (do_quantization=False, FP16)...")
        try:
            ret = rknn.build(do_quantization=False)
        except Exception as e:
            result["error"] = f"构建 RKNN 模型异常: {e}"
            print(f"[ERROR] {result['error']}")
            rknn.release()
            return result

    if ret != 0:
        result["error"] = f"构建 RKNN 模型失败 (ret={ret})"
        print(f"[ERROR] {result['error']}")
        rknn.release()
        return result

    print("[Step 5] 导出 RKNN 模型...")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        result["error"] = f"导出 RKNN 模型失败 (ret={ret})"
        print(f"[ERROR] {result['error']}")
        rknn.release()
        return result

    elapsed = time.time() - start_time
    rknn_size = os.path.getsize(rknn_path)
    result["success"] = True
    result["rknn_path"] = rknn_path
    result["rknn_size_bytes"] = rknn_size
    result["conversion_time"] = elapsed

    onnx_size = result["onnx_size_bytes"]
    print(f"\n[SUCCESS] 模型转换成功!")
    print(f"  ONNX 大小: {onnx_size / 1024 / 1024:.2f} MB")
    print(f"  RKNN 大小: {rknn_size / 1024 / 1024:.2f} MB")
    print(f"  量化模式: {quantization_mode}")
    print(f"  转换耗时: {elapsed:.2f} 秒")

    rknn.release()
    return result


def _prepare_calibration_dataset(model_name, model_info, onnx_dir, output_dir):
    """为 INT8 量化准备校准数据集文本文件"""
    dataset_dir = os.path.join(output_dir, f"{model_name}_calibration")
    os.makedirs(dataset_dir, exist_ok=True)

    npy_paths = []
    for input_name, (zip_name, dtype) in model_info["calibration_inputs"].items():
        data = load_calibration_data(onnx_dir, zip_name, dtype)
        npy_path = os.path.join(dataset_dir, f"{input_name}.npy")
        np.save(npy_path, data)
        npy_paths.append(npy_path)

    # RKNN toolkit 要求 dataset 为文本文件，每行列出一组输入的 npy 路径（空格分隔）
    dataset_path = os.path.join(dataset_dir, "dataset.txt")
    with open(dataset_path, "w") as f:
        f.write(" ".join(npy_paths) + "\n")

    print(f"  校准数据集: {dataset_path}")
    return dataset_path


# ─── 精度验证 ─────────────────────────────────────────────────────────────

def verify_single_model(model_name, model_info, onnx_dir, output_dir,
                        quantization_mode="fp16"):
    """
    精度验证: 使用 RKNN 模拟器对比 ONNX 和 RKNN 推理输出。
    
    注意: RKNN 模拟器要求通过 load_onnx + build 初始化（不能用 load_rknn）。
    """
    verify_result = {
        "model_name": model_name,
        "quantization_mode": quantization_mode,
        "verified": False,
        "outputs": {},
        "error": None,
    }

    onnx_path = os.path.join(onnx_dir, model_info["onnx_file"])
    actual_onnx = onnx_path

    # 如果有预处理版本，使用预处理版本
    preprocessed = os.path.join(output_dir, f"{model_name}_preprocessed.onnx")
    if os.path.exists(preprocessed):
        actual_onnx = preprocessed

    print(f"\n{'='*60}")
    print(f"精度验证: {model_name} (量化模式: {quantization_mode})")
    print(f"{'='*60}")

    # 加载校准数据
    print("[1] 加载校准数据...")
    inputs = load_all_calibration_inputs(onnx_dir, model_info)

    # ONNX 推理（使用原始模型获取参考输出）
    print("[2] 运行 ONNX 推理 (参考)...")
    onnx_inputs = {}
    model_dir = onnx_dir
    for input_name, (zip_name, _) in model_info["calibration_inputs"].items():
        # 加载原始类型数据用于 ONNX 推理
        with zipfile.ZipFile(os.path.join(model_dir, zip_name)) as zf:
            onnx_inputs[input_name] = np.load(zf.open(zf.namelist()[0]))
    onnx_outputs = run_onnx_inference(onnx_path, onnx_inputs)
    if onnx_outputs is None:
        verify_result["error"] = "onnxruntime 不可用"
        return verify_result

    # RKNN 推理（模拟器）
    print(f"[3] 构建 RKNN 模拟器 (量化模式: {quantization_mode})...")
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")
    ret = rknn.load_onnx(model=actual_onnx)
    if ret != 0:
        verify_result["error"] = f"加载 ONNX 失败 (ret={ret})"
        rknn.release()
        return verify_result

    if quantization_mode == "int8":
        dataset_path = _prepare_calibration_dataset(
            model_name, model_info, onnx_dir, output_dir)
        ret = rknn.build(do_quantization=True, dataset=dataset_path)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        verify_result["error"] = f"构建 RKNN 失败 (ret={ret})"
        rknn.release()
        return verify_result

    ret = rknn.init_runtime(target=None)
    if ret != 0:
        verify_result["error"] = f"初始化运行时失败 (ret={ret})"
        rknn.release()
        return verify_result

    print("[4] 运行 RKNN 推理 (模拟器)...")
    # 按预处理模型的输入顺序排列
    sess = ort.InferenceSession(actual_onnx, providers=["CPUExecutionProvider"])
    rknn_input_list = []
    for inp in sess.get_inputs():
        rknn_input_list.append(inputs[inp.name])

    rknn_outputs = rknn.inference(inputs=rknn_input_list)

    # 对比
    output_names = list(onnx_outputs.keys())
    verify_result["verified"] = True

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
        cosine_sim = float(np.dot(onnx_flat, rknn_flat) / (norm_o * norm_r)) if norm_o > 0 and norm_r > 0 else 1.0

        output_info = {
            "shape": list(onnx_out.shape),
            "onnx_range": [float(onnx_out.min()), float(onnx_out.max())],
            "rknn_range": [float(rknn_out.min()), float(rknn_out.max())],
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "cosine_similarity": cosine_sim,
        }
        verify_result["outputs"][name] = output_info

        status = "✅" if cosine_sim > 0.999 else "⚠️" if cosine_sim > 0.99 else "❌"
        print(f"\n  输出 '{name}': {status}")
        print(f"    形状: {onnx_out.shape}")
        print(f"    ONNX: [{onnx_out.min():.6f}, {onnx_out.max():.6f}]")
        print(f"    RKNN: [{rknn_out.min():.6f}, {rknn_out.max():.6f}]")
        print(f"    最大绝对误差: {max_abs_diff:.6e}")
        print(f"    余弦相似度:   {cosine_sim:.8f}")

    rknn.release()
    return verify_result


# ─── 主函数 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS: ONNX → RKNN (RK3588)")
    parser.add_argument("--onnx_dir", type=str, required=True, help="ONNX 模型目录")
    parser.add_argument("--output_dir", type=str, default="rknn_output", help="RKNN 输出目录")
    parser.add_argument("--skip_verify", action="store_true", help="跳过精度验证")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="指定模型（默认全部）")
    parser.add_argument("--quantization", type=str, default="fp16",
                        choices=["fp16", "none", "int8"],
                        help="量化模式: fp16(默认), none(不量化), int8(INT8量化)")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_names = args.models or list(MODELS.keys())

    quant_label = {
        "fp16": "FP16 (float16)",
        "none": "无量化 (保持原始精度)",
        "int8": "INT8 量化",
    }

    print("=" * 60)
    print("Kokoro TTS: ONNX → RKNN 模型转换")
    print(f"  目标平台: RK3588")
    print(f"  量化模式: {quant_label.get(args.quantization, args.quantization)}")
    print(f"  ONNX 目录: {args.onnx_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  模型: {model_names}")
    print("=" * 60)

    # 转换
    conversion_results = {}
    for name in model_names:
        if name not in MODELS:
            print(f"[WARNING] 未知模型: {name}")
            continue
        result = convert_single_model(
            name, MODELS[name], args.onnx_dir, args.output_dir,
            quantization_mode=args.quantization, verbose=args.verbose)
        conversion_results[name] = result

    # 验证
    verify_results = {}
    if not args.skip_verify:
        print(f"\n\n{'='*60}")
        print("精度验证")
        print("=" * 60)
        for name in model_names:
            if name not in MODELS or not conversion_results.get(name, {}).get("success"):
                continue
            vr = verify_single_model(
                name, MODELS[name], args.onnx_dir, args.output_dir,
                quantization_mode=args.quantization)
            verify_results[name] = vr

    # 汇总
    print(f"\n\n{'='*60}")
    print("转换结果汇总")
    print("=" * 60)

    report = {
        "platform": "rk3588",
        "quantization": quant_label.get(args.quantization, args.quantization),
        "quantization_mode": args.quantization,
        "toolkit": "rknn-toolkit2 v2.3.2",
        "conversion": {},
        "verification": {},
    }

    for name, res in conversion_results.items():
        status = "✅ 成功" if res["success"] else f"❌ 失败: {res.get('error', 'unknown')}"
        print(f"  {name}: {status}")
        if res["success"]:
            rknn_mb = res.get("rknn_size_bytes", 0) / 1024 / 1024
            onnx_mb = res.get("onnx_size_bytes", 0) / 1024 / 1024
            print(f"    RKNN: {res['rknn_path']} ({rknn_mb:.2f} MB), "
                  f"ONNX: {onnx_mb:.2f} MB, 耗时: {res['conversion_time']:.1f}s")
        report["conversion"][name] = {
            "success": res["success"],
            "rknn_path": res.get("rknn_path"),
            "rknn_size_bytes": res.get("rknn_size_bytes", 0),
            "onnx_size_bytes": res.get("onnx_size_bytes", 0),
            "time_seconds": res.get("conversion_time"),
            "error": res.get("error"),
        }

    if verify_results:
        print(f"\n精度验证:")
        for name, vr in verify_results.items():
            if vr.get("error"):
                print(f"  {name}: ⚠️ {vr['error']}")
            else:
                for out_name, info in vr.get("outputs", {}).items():
                    cos = info["cosine_similarity"]
                    icon = "✅" if cos > 0.999 else "⚠️" if cos > 0.99 else "❌"
                    print(f"  {name}/{out_name}: cos={cos:.6f} {icon}")
            report["verification"][name] = vr

    report_path = os.path.join(
        args.output_dir, f"conversion_report_{args.quantization}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
