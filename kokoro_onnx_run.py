"""
kokoro_onnx_run.py - Kokoro TTS ONNX模型推理验证脚本

功能:
    验证导出的4个ONNX子模型的正确性和推理性能。
    使用导出时保存的校准数据(npy)作为输入，运行完整的TTS推理流水线，
    生成音频文件并输出性能统计。

模型架构 (4个ONNX子模型):
    - Model1 (model1_bert_duration): BERT文本编码 + 时长预测
    - Model2 (model2_f0_n_asr):     基频(F0)/能量(N)预测 + 文本对齐(ASR)
    - Model3 (model3_decoder):       解码器，生成频谱
    - Model4 (model4_har):           谐波(Harmonics)合成

推理流水线:
    text_ids + ref_s ──→ Model1 ──→ duration, d
                                      │
                         (时长处理 + 对齐矩阵计算)
                                      │
                                      ▼
                         Model2 ──→ F0_pred, N_pred, asr
                                      │
                         Model4 ──→ har (谐波源信号)
                                      │
                                      ▼
                         Model3 ──→ x (频谱)
                                      │
                         (iSTFT后处理)
                                      │
                                      ▼
                                    audio (24kHz WAV)

ONNX模型下载:
    https://github.com/erquren/erquren_kokoro/releases/download/v1.0.0-rknn/kokoro_onnx_input96.zip

用法:
    # 下载并解压ONNX模型
    wget https://github.com/erquren/erquren_kokoro/releases/download/v1.0.0-rknn/kokoro_onnx_input96.zip
    unzip kokoro_onnx_input96.zip -d onnx

    # 运行验证 (使用校准数据)
    python kokoro_onnx_run.py -d onnx -o output.wav

    # 自定义参数
    python kokoro_onnx_run.py -d onnx -o output.wav --speed 1.0 --fade_out 0.3 --warmup 3
"""

import argparse
import os
import time
import zipfile
import io
import numpy as np
import soundfile as sf
import onnxruntime as ort

from typing import Tuple


# ===== 全局常量 =====
SAMPLE_RATE = 24000       # 采样率 24kHz
DEFAULT_SPEED = 1.0       # 默认语速倍率
DEFAULT_FADE_OUT = 0.3    # 默认淡出时长(秒)


# ===== 工具函数 =====

def load_npy_from_zip(zip_path: str) -> np.ndarray:
    """
    从zip压缩包中加载npy数组数据。

    导出脚本(export.py)会将校准数据保存为zip包内的npy文件，
    此函数用于读取这些数据。

    Args:
        zip_path: zip文件路径

    Returns:
        加载的numpy数组
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        npy_files = [n for n in zf.namelist() if n.endswith('.npy')]
        if not npy_files:
            raise ValueError(f"zip文件中未找到.npy文件: {zip_path}")
        with zf.open(npy_files[0]) as f:
            return np.load(io.BytesIO(f.read()))


def apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    对音频末尾应用线性淡出效果，避免截断造成的爆音。

    Args:
        audio: 音频波形数组
        fade_samples: 淡出的采样点数

    Returns:
        应用淡出后的音频
    """
    if fade_samples <= 0 or len(audio) <= fade_samples:
        return audio
    fade = np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
    audio = audio.copy()
    audio[-fade_samples:] *= fade
    return audio


def find_model_file(onnx_dir: str, base_name: str) -> str:
    """
    在指定目录中查找ONNX模型文件。

    优先查找经过onnxsim简化的版本(_sim.onnx)，因为简化后的模型
    更适合推理部署。如果不存在，则回退到原始导出版本(.onnx)。

    Args:
        onnx_dir: ONNX模型所在目录
        base_name: 模型基础名称 (不含后缀)

    Returns:
        模型文件的完整路径

    Raises:
        FileNotFoundError: 两种版本都不存在时抛出
    """
    # 优先使用onnxsim简化后的模型
    sim_path = os.path.join(onnx_dir, f"{base_name}_sim.onnx")
    if os.path.exists(sim_path):
        return sim_path
    # 回退到原始导出模型
    orig_path = os.path.join(onnx_dir, f"{base_name}.onnx")
    if os.path.exists(orig_path):
        return orig_path
    raise FileNotFoundError(
        f"未找到模型文件: {base_name}[_sim].onnx (目录: {onnx_dir})"
    )


def print_model_info(session: ort.InferenceSession, name: str):
    """
    打印ONNX模型的输入输出信息，用于调试和验证模型接口。

    Args:
        session: ONNX推理会话
        name: 模型名称 (用于显示)
    """
    print(f"    {name}:")
    print(f"      输入:")
    for inp in session.get_inputs():
        print(f"        - {inp.name}: shape={inp.shape}, type={inp.type}")
    print(f"      输出:")
    for out in session.get_outputs():
        print(f"        - {out.name}: shape={out.shape}, type={out.type}")


# ===== ONNX推理引擎 =====

class OnnxInferenceEngine:
    """
    ONNX推理引擎 - 使用onnxruntime运行Kokoro TTS的4个ONNX子模型。

    该类完整复现了AX推理引擎的逻辑，包括：
    - 短文本自动复制扩展（适配FIXED_SEQ_LEN的固定输入长度）
    - 时长预测与对齐矩阵计算
    - 基于iSTFT的音频后处理
    - 性能计时统计

    与AX版本的主要区别：
    - 全部使用onnxruntime进行推理
    - ONNX模型的input_lengths作为显式输入传入
    - 自动适配模型实际需要的输入参数

    Attributes:
        FIXED_SEQ_LEN: 固定输入序列长度（音素数），与导出时一致
        N_FFT: iSTFT的FFT窗口大小（来自config的gen_istft_n_fft）
        HOP_LENGTH: iSTFT的帧移大小（来自config的gen_istft_hop_size）
        DOUBLE_INPUT_THRESHOLD: 短文本复制扩展的阈值
    """

    FIXED_SEQ_LEN = 96
    N_FFT = 20
    HOP_LENGTH = 5
    DOUBLE_INPUT_THRESHOLD = 32

    def __init__(self, onnx_dir: str):
        """
        初始化ONNX推理引擎，加载4个ONNX子模型。

        Args:
            onnx_dir: 包含ONNX模型文件的目录路径
        """
        self.onnx_dir = onnx_dir
        providers = ['CPUExecutionProvider']

        # 按模型名称查找并加载ONNX文件（优先使用_sim版本）
        model_names = {
            'model1': "model1_bert_duration",
            'model2': "model2_f0_n_asr",
            'model3': "model3_decoder",
            'model4': "model4_har",
        }

        model1_path = find_model_file(onnx_dir, model_names['model1'])
        model2_path = find_model_file(onnx_dir, model_names['model2'])
        model3_path = find_model_file(onnx_dir, model_names['model3'])
        model4_path = find_model_file(onnx_dir, model_names['model4'])

        print(f"    Model1: {os.path.basename(model1_path)}")
        self.session1 = ort.InferenceSession(model1_path, providers=providers)
        print(f"    Model2: {os.path.basename(model2_path)}")
        self.session2 = ort.InferenceSession(model2_path, providers=providers)
        print(f"    Model3: {os.path.basename(model3_path)}")
        self.session3 = ort.InferenceSession(model3_path, providers=providers)
        print(f"    Model4: {os.path.basename(model4_path)}")
        self.session4 = ort.InferenceSession(model4_path, providers=providers)

        # 性能计时器
        self.model1_time = 0.0
        self.model2_time = 0.0
        self.model3_time = 0.0
        self.har_time = 0.0
        self.inference_count = 0

        # iSTFT后处理的预计算参数
        # Hann窗函数: w(n) = 0.5 * (1 - cos(2*pi*n/N))
        self._n_overlap = self.N_FFT // self.HOP_LENGTH
        self._window = (0.5 * (1.0 - np.cos(
            2.0 * np.pi * np.arange(self.N_FFT) / self.N_FFT
        ))).astype(np.float32)
        # 预计算窗口平方矩阵，用于overlap-add归一化
        self._win_sq = (self._window ** 2).reshape(self._n_overlap, self.HOP_LENGTH)
        self._ws_cache = {}

    def _prepare_model_inputs(
        self, session: ort.InferenceSession, inputs_dict: dict
    ) -> dict:
        """
        根据模型的实际输入定义，自动筛选并转换输入数据类型。

        不同的ONNX模型可能有不同的输入集合和数据类型要求，
        此函数通过读取模型的input metadata来自动匹配。

        Args:
            session: ONNX推理会话
            inputs_dict: 候选输入字典 {name: ndarray}

        Returns:
            适配后的输入字典（仅包含模型需要的输入，且类型正确）
        """
        # ONNX类型字符串到numpy dtype的映射
        type_map = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(int64)': np.int64,
            'tensor(int32)': np.int32,
            'tensor(bool)': np.bool_,
            'tensor(uint8)': np.uint8,
        }

        feed = {}
        for inp in session.get_inputs():
            if inp.name in inputs_dict:
                value = inputs_dict[inp.name]
                expected_dtype = type_map.get(inp.type)
                if expected_dtype is not None:
                    value = value.astype(expected_dtype)
                feed[inp.name] = value
        return feed

    def _get_window_sum(self, n_frames: int) -> np.ndarray:
        """
        计算iSTFT的窗口归一化因子（带缓存）。

        在overlap-add合成中，需要除以窗函数的平方和来归一化，
        避免窗函数造成的幅度变化。

        Args:
            n_frames: 帧数

        Returns:
            窗口平方和数组，长度为 N_FFT + HOP_LENGTH * (n_frames - 1)
        """
        if n_frames not in self._ws_cache:
            out_len = self.N_FFT + self.HOP_LENGTH * (n_frames - 1)
            ws = np.zeros(out_len, dtype=np.float64)
            for j in range(self._n_overlap):
                offset = j * self.HOP_LENGTH
                tile = np.tile(self._win_sq[j], n_frames)
                end = min(offset + len(tile), out_len)
                ws[offset:end] += tile[:end - offset]
            ws[ws < 1e-8] = 1e-8
            self._ws_cache[n_frames] = ws
        return self._ws_cache[n_frames]

    def _compute_external_preprocessing(
        self, input_ids: np.ndarray, actual_len: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算模型输入的前处理数据: input_lengths 和 text_mask。

        text_mask是一个布尔掩码，True表示padding位置（即超出actual_len的部分）。
        用于告诉BERT模型哪些位置不需要关注。

        Args:
            input_ids: 输入ID序列, shape=(1, FIXED_SEQ_LEN)
            actual_len: 实际有效音素数

        Returns:
            input_lengths: 每个batch的有效长度, shape=(batch_size,)
            text_mask: 文本掩码, shape=(1, FIXED_SEQ_LEN), True=padding
        """
        if actual_len is None:
            actual_len = self.FIXED_SEQ_LEN
        input_lengths = np.full((input_ids.shape[0],), actual_len, dtype=np.int64)
        # 生成位置索引并与长度比较，得到padding掩码
        text_mask = (
            np.arange(self.FIXED_SEQ_LEN)[np.newaxis, :]
            >= input_lengths[:, np.newaxis]
        )
        return input_lengths, text_mask

    def _prepare_input_ids(
        self, input_ids: np.ndarray, actual_len: int
    ) -> Tuple[np.ndarray, int, bool]:
        """
        预处理输入ID，对短文本进行复制扩展。

        当输入长度小于DOUBLE_INPUT_THRESHOLD时，将有效部分复制一倍，
        以增加模型的输入信息量，适配短文本的推理场景。
        推理完成后，输出的音频会取前半部分。

        Args:
            input_ids: 原始输入ID, shape=(1, FIXED_SEQ_LEN)
            actual_len: 实际有效音素数

        Returns:
            input_ids: 处理后的输入ID
            actual_len: 处理后的有效长度
            is_doubled: 是否进行了复制扩展
        """
        is_doubled = False
        original_actual_len = actual_len

        if actual_len <= self.DOUBLE_INPUT_THRESHOLD:
            is_doubled = True
            valid_content = input_ids[:, :actual_len]
            # 复制有效内容
            input_ids_doubled = np.concatenate(
                [valid_content, valid_content], axis=1
            )
            # 补齐到FIXED_SEQ_LEN
            padding_len = self.FIXED_SEQ_LEN - input_ids_doubled.shape[1]
            if padding_len > 0:
                input_ids = np.concatenate([
                    input_ids_doubled,
                    np.zeros((1, padding_len), dtype=input_ids.dtype)
                ], axis=1)
            else:
                input_ids = input_ids_doubled[:, :self.FIXED_SEQ_LEN]
            actual_len = min(original_actual_len * 2, self.FIXED_SEQ_LEN)

        return input_ids, actual_len, is_doubled

    def _process_duration(
        self, duration: np.ndarray, actual_len: int, speed: float
    ) -> Tuple[np.ndarray, int]:
        """
        处理时长预测结果，将模型输出的原始时长转换为整数帧数。

        处理步骤:
        1. Sigmoid激活: 将原始输出映射到(0,1)区间
        2. 维度求和 + 速度调整: 得到每个音素的持续帧数
        3. 四舍五入 + 裁剪: 保证每个音素至少1帧
        4. 帧数调整: 总帧数固定为 FIXED_SEQ_LEN * 2
        5. Padding分配: 将剩余帧数分配给padding位置

        Args:
            duration: 模型输出的原始时长, shape=(1, seq_len, dim)
            actual_len: 实际有效音素数
            speed: 语速倍率 (>1加速, <1减速)

        Returns:
            pred_dur: 每个位置的帧数, shape=(FIXED_SEQ_LEN,)
            total_frames: 总帧数 (应等于 FIXED_SEQ_LEN * 2)
        """
        # Sigmoid + 求和 + 速度调整
        duration_processed = 1.0 / (1.0 + np.exp(-duration))
        duration_processed = duration_processed.sum(axis=-1) / speed
        pred_dur_original = (
            np.round(duration_processed).clip(min=1).astype(np.int64).squeeze()
        )

        # 分离有效内容部分和padding部分
        pred_dur_actual = pred_dur_original[:actual_len]
        pred_dur_padding = np.zeros(
            self.FIXED_SEQ_LEN - actual_len, dtype=np.int64
        )
        pred_dur = np.concatenate([pred_dur_actual, pred_dur_padding])

        # 调整总帧数到固定值 = FIXED_SEQ_LEN * 2
        fixed_total_frames = self.FIXED_SEQ_LEN * 2
        diff = fixed_total_frames - pred_dur[:actual_len].sum()

        # 如果预测帧数过多，从最长的音素开始减少
        if diff < 0:
            indices = np.argsort(pred_dur[:actual_len])[::-1]
            decreased = 0
            for idx in indices:
                if pred_dur[idx] > 1 and decreased < abs(diff):
                    pred_dur[idx] -= 1
                    decreased += 1
                if decreased >= abs(diff):
                    break

        # 将剩余帧数均匀分配给padding位置
        remaining_frames = fixed_total_frames - pred_dur[:actual_len].sum()
        padding_len = self.FIXED_SEQ_LEN - actual_len
        if remaining_frames > 0 and padding_len > 0:
            frames_per_padding = remaining_frames // padding_len
            remainder = remaining_frames % padding_len
            pred_dur[actual_len:] = frames_per_padding
            if remainder > 0:
                pred_dur[actual_len:actual_len + remainder] += 1

        total_frames = pred_dur.sum()
        return pred_dur, total_frames

    def _create_alignment_matrix(
        self, pred_dur: np.ndarray, total_frames: int
    ) -> np.ndarray:
        """
        根据时长预测结果创建音素-帧对齐矩阵。

        对齐矩阵是一个稀疏的0-1矩阵，表示每个音素对应哪些帧。
        例如，如果音素i持续3帧（从第5帧到第7帧），则
        pred_aln_trg[i, 5:8] = 1。

        Args:
            pred_dur: 每个位置的帧数, shape=(FIXED_SEQ_LEN,)
            total_frames: 总帧数

        Returns:
            pred_aln_trg: 对齐矩阵, shape=(1, FIXED_SEQ_LEN, total_frames)
        """
        indices = np.repeat(np.arange(self.FIXED_SEQ_LEN), pred_dur)
        pred_aln_trg = np.zeros(
            (self.FIXED_SEQ_LEN, total_frames), dtype=np.float32
        )
        if len(indices) > 0:
            pred_aln_trg[indices, np.arange(total_frames)] = 1.0
        return pred_aln_trg[np.newaxis, ...]

    def _trim_audio_by_content(
        self, audio: np.ndarray, actual_content_frames: int,
        total_frames: int, actual_len: int
    ) -> np.ndarray:
        """
        根据实际内容帧数占比裁剪音频，去除padding对应的音频部分。

        由于模型输入是固定长度(FIXED_SEQ_LEN)，其中部分是padding，
        生成的音频中也包含padding对应的部分，需要裁剪掉。

        Args:
            audio: 完整音频波形
            actual_content_frames: 实际内容对应的帧数
            total_frames: 总帧数（包含padding帧）
            actual_len: 实际有效音素数

        Returns:
            裁剪后的音频（仅保留有效内容部分）
        """
        padding_len = self.FIXED_SEQ_LEN - actual_len
        if padding_len > 0:
            content_ratio = actual_content_frames / total_frames
            audio_len_to_keep = int(len(audio) * content_ratio)
            return audio[:audio_len_to_keep]
        return audio

    def _postprocess_x_to_audio(self, x: np.ndarray) -> np.ndarray:
        """
        将Model3输出的频谱x通过iSTFT转换为时域音频波形。

        Model3输出的x包含两部分:
        - 前半部分 x[:, :n_freq, :]: log幅度谱
        - 后半部分 x[:, n_freq:, :]: 相位的sin值

        通过exp恢复幅度，由sin推导cos，构造复数STFT矩阵，
        然后使用iFFT + overlap-add合成时域信号。

        Args:
            x: 频谱输出, shape=(1, n_freq*2, n_frames)

        Returns:
            audio: 音频波形, shape=(n_samples,), dtype=float32
        """
        n_freq = self.N_FFT // 2 + 1

        # 分离幅度谱和相位
        spec = np.exp(x[0, :n_freq, :])                       # 幅度谱 (从log域恢复)
        phase_sin = np.sin(x[0, n_freq:, :])                   # 相位的sin值
        # 由sin²+cos²=1推导cos (取正值，因为cos在主值区间可能为正)
        cos_part = np.sqrt(np.clip(1.0 - phase_sin * phase_sin, 0.0, 1.0))
        # 构造复数STFT: magnitude * (cos + j*sin)
        stft_matrix = spec * (cos_part + 1j * phase_sin)

        n_frames = stft_matrix.shape[1]

        # iFFT: 频域→时域 (每帧独立)
        frames = np.fft.irfft(stft_matrix.T, n=self.N_FFT)
        # 应用Hann窗
        frames *= self._window

        # Overlap-Add合成
        out_len = self.N_FFT + self.HOP_LENGTH * (n_frames - 1)
        output = np.zeros(out_len, dtype=np.float64)
        fr = frames.reshape(n_frames, self._n_overlap, self.HOP_LENGTH)
        for j in range(self._n_overlap):
            offset = j * self.HOP_LENGTH
            data = fr[:, j, :].ravel()
            output[offset:offset + len(data)] += data

        # 窗函数归一化 (除以窗口平方和)
        output /= self._get_window_sum(n_frames)[:out_len]

        # 去除首尾padding (N_FFT//2)
        pad = self.N_FFT // 2
        if pad > 0:
            return output[pad:-pad].astype(np.float32)
        return output.astype(np.float32)

    def inference_single_chunk(
        self,
        input_ids: np.ndarray,
        ref_s: np.ndarray,
        actual_len: int,
        speed: float
    ) -> Tuple[np.ndarray, int, int]:
        """
        对单个文本chunk进行完整的4模型推理。

        完整流程:
        1. 短文本预处理 (可能复制扩展)
        2. 计算input_lengths和text_mask
        3. Model1: BERT编码 + 时长预测 → duration, d
        4. 时长处理: duration → pred_dur → alignment matrix
        5. 矩阵运算: en = d^T @ alignment
        6. Model2: F0/能量预测 + ASR对齐 → F0_pred, N_pred, asr
        7. Model4: 谐波合成 → har
        8. Model3: 解码生成频谱 → x
        9. iSTFT后处理: x → audio

        Args:
            input_ids: 输入音素ID, shape=(1, FIXED_SEQ_LEN)
            ref_s: 参考说话人风格向量, shape=(1, 256)
                   前128维为decoder风格, 后128维为predictor风格
            actual_len: 实际有效音素数 (不含padding)
            speed: 语速倍率

        Returns:
            audio: 生成的音频波形
            actual_content_frames: 有效内容对应的帧数
            total_frames: 总帧数
        """
        self.inference_count += 1

        # Step 1: 短文本预处理
        input_ids, actual_len, is_doubled = self._prepare_input_ids(
            input_ids, actual_len
        )

        # Step 2: 计算前处理数据
        input_lengths, text_mask = self._compute_external_preprocessing(
            input_ids, actual_len=actual_len
        )

        # ===== Step 3: Model1 - BERT编码 + 时长预测 =====
        # 输入: input_ids(音素ID), ref_s(风格向量), input_lengths, text_mask
        # 输出: duration(原始时长), d(编码器隐状态)
        model1_inputs = {
            'input_ids': input_ids,
            'ref_s': ref_s,
            'input_lengths': input_lengths,
            'text_mask': text_mask,
        }
        model1_feed = self._prepare_model_inputs(self.session1, model1_inputs)

        t1 = time.time()
        outputs1 = self.session1.run(None, model1_feed)
        self.model1_time += time.time() - t1
        duration, d = outputs1

        # Step 4: 时长处理 + 对齐矩阵
        pred_dur, total_frames = self._process_duration(
            duration, actual_len, speed
        )
        pred_aln_trg = self._create_alignment_matrix(pred_dur, total_frames)

        # Step 5: 计算编码器输出的帧级表示
        # en = d^T @ alignment, 将音素级表示扩展到帧级
        d_transposed = np.transpose(d, (0, 2, 1))
        en = d_transposed @ pred_aln_trg

        # ===== Step 6: Model2 - F0/能量/ASR预测 =====
        # 输入: en(帧级编码), ref_s(风格), input_ids, text_mask, pred_aln_trg
        # 输出: F0_pred(基频), N_pred(能量), asr(对齐后的文本编码)
        model2_inputs = {
            'en': en,
            'ref_s': ref_s,
            'input_ids': input_ids,
            'input_lengths': input_lengths,
            'text_mask': text_mask,
            'pred_aln_trg': pred_aln_trg,
        }
        model2_feed = self._prepare_model_inputs(self.session2, model2_inputs)

        t2 = time.time()
        outputs2 = self.session2.run(None, model2_feed)
        self.model2_time += time.time() - t2
        F0_pred, N_pred, asr = outputs2

        # ===== Step 7: Model4 - 谐波源信号合成 =====
        # 从F0预测生成谐波源信号 (har_source → STFT → har)
        model4_feed = self._prepare_model_inputs(
            self.session4, {'F0_pred': F0_pred}
        )

        t_har = time.time()
        har = self.session4.run(None, model4_feed)[0]
        self.har_time += time.time() - t_har

        # ===== Step 8: Model3 - 解码器 =====
        # 输入: asr(文本编码), F0_pred, N_pred, ref_s(风格), har(谐波)
        # 输出: x (频谱, shape=(1, n_freq*2, n_frames))
        model3_inputs = {
            'asr': asr,
            'F0_pred': F0_pred,
            'N_pred': N_pred,
            'ref_s': ref_s,
            'har': har,
        }
        model3_feed = self._prepare_model_inputs(self.session3, model3_inputs)

        t3 = time.time()
        outputs3 = self.session3.run(None, model3_feed)
        self.model3_time += time.time() - t3
        x = outputs3[0]

        # Step 9: iSTFT后处理 - 频谱转音频
        audio = self._postprocess_x_to_audio(x)
        actual_content_frames = pred_dur[:actual_len].sum()

        # 短文本复制模式: 只取前半部分音频
        if is_doubled:
            audio = audio[:len(audio) // 2]
            actual_content_frames = actual_content_frames // 2
            total_frames = total_frames // 2

        return audio, actual_content_frames, total_frames

    def inference(
        self,
        input_ids: np.ndarray,
        ref_s: np.ndarray,
        actual_len: int,
        speed: float = DEFAULT_SPEED,
        fade_out_duration: float = DEFAULT_FADE_OUT
    ) -> np.ndarray:
        """
        完整推理流程: 输入音素ID和风格向量，输出音频波形。

        Args:
            input_ids: 输入音素ID序列, shape=(1, FIXED_SEQ_LEN)
            ref_s: 参考风格向量, shape=(1, 256)
            actual_len: 实际有效音素数
            speed: 语速倍率 (默认1.0)
            fade_out_duration: 音频末尾淡出时长(秒)

        Returns:
            audio: 生成的音频波形 (24kHz, float32)
        """
        fade_samples = (
            int(SAMPLE_RATE * fade_out_duration) if fade_out_duration > 0 else 0
        )

        # 运行单chunk推理
        audio, actual_content_frames, total_frames = (
            self.inference_single_chunk(input_ids, ref_s, actual_len, speed)
        )

        # 裁剪padding产生的多余音频
        audio_trimmed = self._trim_audio_by_content(
            audio, actual_content_frames, total_frames, actual_len
        )

        # 应用淡出效果
        if fade_samples > 0:
            audio_trimmed = apply_fade_out(audio_trimmed, fade_samples)

        return audio_trimmed

    def print_performance(self):
        """打印各模型的推理耗时统计信息。"""
        if self.inference_count > 0:
            total = (
                self.model1_time + self.model2_time
                + self.model3_time + self.har_time
            )
            print(f"\n推理性能统计 (共{self.inference_count}次):")
            print(
                f"  ├─ Model1 (BERT+Duration): {self.model1_time:.3f}s "
                f"(平均 {self.model1_time / self.inference_count * 1000:.1f}ms)"
            )
            print(
                f"  ├─ Model2 (F0+N+ASR):      {self.model2_time:.3f}s "
                f"(平均 {self.model2_time / self.inference_count * 1000:.1f}ms)"
            )
            print(
                f"  ├─ Model3 (Decoder):        {self.model3_time:.3f}s "
                f"(平均 {self.model3_time / self.inference_count * 1000:.1f}ms)"
            )
            print(
                f"  └─ Model4 (Harmonics):      {self.har_time:.3f}s "
                f"(平均 {self.har_time / self.inference_count * 1000:.1f}ms)"
            )
            print(f"  合计推理耗时: {total:.3f}s")


# ===== 数据加载 =====

def load_calibration_data(
    onnx_dir: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    从ONNX导出目录加载校准数据，用于验证推理。

    导出脚本(export.py)在导出ONNX模型时会同时保存校准数据，
    这些数据以zip格式存储在模型目录中，包含真实的输入样本。

    Args:
        onnx_dir: 包含ONNX模型和校准数据的目录

    Returns:
        input_ids: 输入音素ID, shape=(1, FIXED_SEQ_LEN)
        ref_s: 参考风格向量, shape=(1, 256)
        actual_len: 实际有效音素长度
    """
    # 加载Model1的输入数据
    input_ids_path = os.path.join(onnx_dir, "model1_input_ids.zip")
    ref_s_path = os.path.join(onnx_dir, "model1_ref_s.zip")

    if not os.path.exists(input_ids_path):
        raise FileNotFoundError(
            f"未找到校准数据: {input_ids_path}\n"
            f"请确保ONNX模型包中包含校准数据文件 (model1_*.zip)"
        )

    input_ids = load_npy_from_zip(input_ids_path)
    ref_s = load_npy_from_zip(ref_s_path)

    # 确定实际有效长度
    input_lengths_path = os.path.join(onnx_dir, "model1_input_lengths.zip")
    if os.path.exists(input_lengths_path):
        # 优先使用保存的input_lengths
        input_lengths = load_npy_from_zip(input_lengths_path)
        actual_len = int(input_lengths[0])
    else:
        # 从input_ids推断: 查找最后一个非零元素的位置
        nonzero = np.nonzero(input_ids[0])[0]
        actual_len = (
            int(nonzero[-1] + 1) if len(nonzero) > 0 else input_ids.shape[1]
        )

    return input_ids, ref_s, actual_len


# ===== 主函数 =====

def main():
    """
    主函数: 解析参数，加载模型和数据，运行推理验证。

    返回值:
        0: 验证通过
        1: 验证失败（音频异常）
    """
    parser = argparse.ArgumentParser(
        description="Kokoro TTS ONNX模型推理验证脚本 - "
                    "验证导出的4个ONNX子模型的正确性和推理性能"
    )
    parser.add_argument(
        "--onnx_dir", "-d", type=str, default="onnx",
        help="ONNX模型及校准数据目录路径 (默认: onnx)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output_onnx.wav",
        help="输出音频文件路径 (默认: output_onnx.wav)"
    )
    parser.add_argument(
        "--speed", "-s", type=float, default=DEFAULT_SPEED,
        help=f"语速倍率, >1加速, <1减速 (默认: {DEFAULT_SPEED})"
    )
    parser.add_argument(
        "--fade_out", "-f", type=float, default=DEFAULT_FADE_OUT,
        help=f"音频末尾淡出时长(秒), 设为0禁用 (默认: {DEFAULT_FADE_OUT})"
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=1,
        help="预热推理次数, 用于排除首次推理的冷启动开销 (默认: 1)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Kokoro TTS ONNX 模型推理验证")
    print("=" * 60)

    # ===== 1. 加载ONNX模型 =====
    start_init = time.time()
    print("\n[1/5] 加载ONNX模型...")
    engine = OnnxInferenceEngine(args.onnx_dir)
    init_time = time.time() - start_init
    print(f"  模型加载完成: {init_time:.3f}s")

    # 打印模型输入输出信息
    print("\n[2/5] 模型信息:")
    print_model_info(engine.session1, "Model1 (BERT+Duration)")
    print_model_info(engine.session2, "Model2 (F0+N+ASR)")
    print_model_info(engine.session3, "Model3 (Decoder)")
    print_model_info(engine.session4, "Model4 (Harmonics)")

    # ===== 2. 加载校准数据 =====
    print("\n[3/5] 加载校准数据...")
    input_ids, ref_s, actual_len = load_calibration_data(args.onnx_dir)
    print(f"  input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}")
    print(f"  ref_s:     shape={ref_s.shape}, dtype={ref_s.dtype}")
    print(f"  实际有效长度: {actual_len}/{input_ids.shape[1]}")

    # ===== 3. 预热推理 =====
    if args.warmup > 0:
        print(f"\n[4/5] 预热推理 ({args.warmup}次)...")
        for i in range(args.warmup):
            _ = engine.inference(
                input_ids, ref_s, actual_len,
                speed=args.speed, fade_out_duration=0
            )
        # 重置计时器，排除预热的影响
        engine.model1_time = 0.0
        engine.model2_time = 0.0
        engine.model3_time = 0.0
        engine.har_time = 0.0
        engine.inference_count = 0
        print("  预热完成")
    else:
        print("\n[4/5] 跳过预热")

    # ===== 4. 正式推理 =====
    print("\n[5/5] 正式推理...")
    start_inference = time.time()
    audio = engine.inference(
        input_ids, ref_s, actual_len,
        speed=args.speed,
        fade_out_duration=args.fade_out
    )
    inference_time = time.time() - start_inference

    # ===== 5. 保存音频 =====
    sf.write(args.output, audio, SAMPLE_RATE)
    audio_duration = len(audio) / SAMPLE_RATE

    # ===== 6. 输出结果 =====
    print(f"\n{'=' * 60}")
    print(f"✓ 推理完成")
    print(f"  输出文件:    {args.output}")
    print(f"  音频时长:    {audio_duration:.2f}s")
    print(f"  音频采样点:  {len(audio)}")
    print(f"  采样率:      {SAMPLE_RATE}Hz")
    print(f"  推理耗时:    {inference_time:.3f}s")
    print(f"  RTF(实时因子): {inference_time / audio_duration:.4f}")

    # 性能分解
    engine.print_performance()

    # ===== 7. 音频质量检查 =====
    print(f"\n音频质量检查:")
    max_amp = np.abs(audio).max()
    rms = np.sqrt(np.mean(audio ** 2))
    has_nan = np.isnan(audio).any()
    has_inf = np.isinf(audio).any()

    print(f"  最大振幅:    {max_amp:.6f}")
    print(f"  RMS能量:     {rms:.6f}")
    print(f"  包含NaN:     {'是 ✗' if has_nan else '否 ✓'}")
    print(f"  包含Inf:     {'是 ✗' if has_inf else '否 ✓'}")

    if has_nan or has_inf:
        print("  ✗ 错误: 音频包含无效值(NaN/Inf)")
        print("=" * 60)
        return 1
    elif max_amp < 0.001:
        print("  ⚠ 警告: 音频振幅极小，模型输出可能存在问题")
        print("=" * 60)
        return 1
    else:
        print("  ✓ 音频质量检查通过")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    exit(main())
