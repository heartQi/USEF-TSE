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
    emb_s1, _ = librosa.load(args.ref_path, sr=fs)

    mix = torch.from_numpy(mix_wav).to(device).unsqueeze(0)
    embd = torch.from_numpy(emb_s1).to(device).unsqueeze(0)
    print(f"Mix shape: {mix.shape}, Embd shape: {embd.shape}")

    est_source = model(mix, embd)
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
