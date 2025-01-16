import torch
import torch.nn as nn
import copy

from models.local.TFgridnet import GridNetV2Block


class Tar_Model(nn.Module):
    def __init__(
        self,
        real_att,
        n_freqs,
        hidden_channels,
        n_head,
        emb_dim,
        emb_ks,
        emb_hs,
        num_layers=6,
        eps=1e-5,
    ):
        super(Tar_Model, self).__init__()
        self.num_layers = num_layers

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.deconv = nn.ConvTranspose2d(2 * emb_dim, 2, ks, padding=padding)

        self.att = real_att

        self.dual_mdl = nn.ModuleList([
            copy.deepcopy(
                GridNetV2Block(
                    2 * emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    hidden_channels,
                    n_head,
                    approx_qk_dim=512,
                    activation="prelu",
                )
            )
            for _ in range(num_layers)
        ])

    def forward(self, mix_ri, aux_ri):
        # `mix_ri` 和 `aux_ri` 已经是频域数据 [B, C=2, F, T]
        mix_ri = self.conv(mix_ri)
        aux_ri = self.conv(aux_ri)

        # 通过注意力机制融合辅助信息
        aux_ri = self.att(mix_ri, aux_ri)

        # 合并主混合信号和辅助信号
        x = torch.cat([mix_ri, aux_ri], dim=1)

        # 通过多层 GridNetV2Block 处理
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)

        # 通过逆卷积还原频域数据
        x = self.deconv(x)

        return x
