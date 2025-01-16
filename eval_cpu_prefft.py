import os
import sys

sys.path.append('../../..')
import argparse
import torch
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from collections import OrderedDict
import librosa
import soundfile as sf
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


def main(config, args):
    model = config['modules']['masknet']
    model = load_pretrained_modules(model, args.chkpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    fs = config['sample_rate']
    mix_wav, _ = librosa.load(args.mix_path, sr=fs)
    emb_wav, _ = librosa.load(args.ref_path, sr=fs)

    mix = torch.from_numpy(mix_wav).to(device).unsqueeze(0).unsqueeze(1)
    embd = torch.from_numpy(emb_wav).to(device).unsqueeze(0).unsqueeze(1)
    print(f"Mix shape: {mix.shape}, Embd shape: {embd.shape}")

    # 数据预处理
    win_length = 128
    n_fft = 128
    hop_length = 64
    window = torch.hann_window(win_length).to(device)
    print(f"win_length: {win_length}")
    
    # 1. 输入信号标准化
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

    # 4. 模型前向推理
    output_real_imag = model(input_ri, aux_ri)

    # 5. 输出的频域数据还原到时间域
    out_r = output_real_imag[:, 0, :, :].permute(0, 2, 1).contiguous()
    out_i = output_real_imag[:, 1, :, :].permute(0, 2, 1).contiguous()
    est_source = ifft((out_r, out_i), input_type="real_imag").unsqueeze(1)

    # 6. 恢复原幅度
    est_source = est_source * mix_std
    est_source = est_source.squeeze().detach().cpu().numpy()
    est_source = normalize_audio(est_source)
    sf.write("estimated_source.wav", est_source, fs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation')
    parser.add_argument('-c', '--config', type=str, default='',help='config file path')
    parser.add_argument('-p', '--chkpt-path', type=str, default='',help='path to the chosen checkpoint')
    parser.add_argument('-m', '--mix-path', type=str, required=True, help='混合音频文件路径')
    parser.add_argument('-r', '--ref-path', type=str, required=True, help='参考音频文件路径')
    args = parser.parse_args()

    for f in args.config, args.chkpt_path:
        assert os.path.isfile(f), "No such file: %s" % f

    with open(args.config, 'r') as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    print('INFO: Loaded hparams from: {}'.format(args.config))

    main(config, args)
