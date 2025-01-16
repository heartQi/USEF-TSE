import os
import argparse
import torch
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from collections import OrderedDict
import librosa
import soundfile as sf
import onnxruntime as ort
import onnx
from onnxsim import simplify
from utils.feature import STFT
from utils.feature import iSTFT

def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace("module.", "").replace("convolution_", "convolution_module.")  # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)
    return model
def normalize_audio(audio):
    # 将音频信号的幅度限制在[-1, 1]之间
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def export_to_onnx(model, model_input, onnx_path="model.onnx"):
    model.eval()
    # 导出模型为ONNX格式
    torch.onnx.export(
        model,
        model_input,  # 模型输入
        onnx_path,
        input_names=['mix', 'embd'],  # 输入名称
        output_names=["est_source"],  # 输出名称
        dynamic_axes={
            "mix": {2: "freq", 3: "time"},  # 动态维度
            "embd": {2: "freq", 3: "time"},
            "est_source": {2: "freq", 3: "time"}
        },
        opset_version=12  # ONNX 的 opset 版本
    )
    print(f"ONNX 模型已保存到 {onnx_path}")

def infer_with_onnx(onnx_path, inputs, fs):
    # 加载ONNX模型
    ort_session = ort.InferenceSession(onnx_path)
    est_source = ort_session.run(None, inputs)

    return est_source

def simplify_onnx(onnx_path):
    # 加载 ONNX 模型
    model = onnx.load(onnx_path)  # 替换为实际的文件路径

    # 简化模型
    model_simp, check = simplify(onnx_path)

    # 确认模型简化后有效
    assert check, "Simplified ONNX model could not be validated"

    # 如果验证成功，保存简化后的模型
    onnx.save(model_simp, './ONNX/USEF-TSE_simplified.onnx')  # 替换为保存的文件路径
    print("Simplified model saved successfully!")

def main(config, chkpt_path, mix_path, ref_path):
    """
    主程序，加载配置，加载模型，执行推理
    """
    for f in config, chkpt_path:
        assert os.path.isfile(f), f"没有找到文件：{f}"

    with open(config, 'r') as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    print('INFO: 加载了配置文件：{}'.format(config))

    model = config['modules']['masknet']
    model = load_pretrained_modules(model, chkpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    fs = config['sample_rate']
    onnx_path = "./ONNX/USEF-TSE.onnx"

    # 数据预处理
    win_length = 128
    n_fft = 128
    hop_length = 64
    window = torch.hann_window(win_length).to(device)
    print(f"win_length: {win_length}")

    mix, _ = librosa.load(args.mix_path, sr=fs)
    embd, _ = librosa.load(args.ref_path, sr=fs)
    mix = torch.from_numpy(mix).to(device).unsqueeze(0).unsqueeze(1)
    embd = torch.from_numpy(embd).to(device).unsqueeze(0).unsqueeze(1)
    print(f"Mix shape: {mix.shape}, Embd shape: {embd.shape}")

    mix_std = mix.std(dim=(1, 2), keepdim=True)
    mix = mix / mix_std

    embd_std = embd.std(dim=(1, 2), keepdim=True)
    embd = embd / embd_std

    # 2. 时间域转换到频域
    fft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    ifft = iSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    input_stft = fft(mix)[-1]
    aux_stft = fft(embd)[-1]

    # 3. 将 STFT 结果转换为 `real` 和 `imag` 格式
    input_ri = torch.cat([input_stft.real, input_stft.imag], dim=1)
    aux_ri = torch.cat([aux_stft.real, aux_stft.imag], dim=1)
    input_ri = input_ri.permute(0, 1, 3, 2).contiguous()
    aux_ri = aux_ri.permute(0, 1, 3, 2).contiguous()

    # 导出ONNX模型
    model_inputs = (input_ri.to(device), aux_ri.to(device)) # 元组
    export_to_onnx(model, model_inputs, onnx_path)

    # 使用ONNX模型进行推理
    model_inputs_data = {
        "mix": input_ri.cpu().numpy(),
        "embd": aux_ri.cpu().numpy()
    } # 字典
    output_real_imag = infer_with_onnx(onnx_path, model_inputs_data, fs)

    # 输出的频域数据还原到时间域
    output_real_imag = torch.from_numpy(output_real_imag[0]).to(device)  # 提取第一个输出并转换为 torch.Tensor
    out_r = output_real_imag[:, 0, :, :].permute(0, 2, 1).contiguous()
    out_i = output_real_imag[:, 1, :, :].permute(0, 2, 1).contiguous()
    est_source = ifft((out_r, out_i), input_type="real_imag").unsqueeze(1)

    # 恢复原幅度
    est_source = est_source * mix_std
    est_source = est_source.squeeze().detach().cpu().numpy()
    est_source = normalize_audio(est_source)
    sf.write("estimated_source-onnx.wav", est_source, fs)

    # 简化ONNX模型
    simplify_onnx(onnx_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation')
    parser.add_argument('-c', '--config', type=str, default='',help='config file path')
    parser.add_argument('-p', '--chkpt-path', type=str, default='',help='path to the chosen checkpoint')
    parser.add_argument('-m', '--mix-path', type=str, required=True, help='混合音频文件路径')
    parser.add_argument('-r', '--ref-path', type=str, required=True, help='参考音频文件路径')
    args = parser.parse_args()

    main(args.config, args.chkpt_path, args.mix_path, args.ref_path)

