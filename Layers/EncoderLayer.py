# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021


import torch
from torch import nn
import os
import numpy as np
from Layers.LayerNorm import LayerNorm
from Layers.Attention import MultiHeadedAttention
import matplotlib.pyplot as plt
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    """
    Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, self_attn, feed_forward, feed_forward_macaron, conv_module, dropout_rate, normalize_before=True, concat_after=False, ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """
        Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)


#####################
        # x_q = x.to(device='cuda:0')
        # batch = x_q.shape[0] # save batch size as it is query
        # residual = x
        # self.accent_emb = []
        # for embs in next(os.walk('/nas/projects/vokquant/IMS-Toucan_lang_emb_conformer/Preprocessing/embeds/'))[2]:
        #     self.accent_emb.append(torch.load(os.path.join('/nas/projects/vokquant/IMS-Toucan_lang_emb_conformer/Preprocessing/embeds/', embs )))
        # self.accent_emb = np.concatenate(self.accent_emb)       # created shape: (12, 1, 192)
        # self.accent_emb = torch.Tensor(self.accent_emb).repeat(1,1,1,2)
        # self.accent_emb = self.accent_emb.squeeze(0)
        # x_acc_emb = self.accent_emb.permute(1, 0, 2).repeat(batch,1,1).to(device='cuda:0')
        
        # # check if x_acc_emb and x_q is in cpu or gpu
        # #print("x_acc_emb.device: ", x_acc_emb.device, "x_q.device: ", x_q.device)
        # #print("x_acc_emb.shape: ", x_acc_emb.shape, "x_q.shape: ", x_q.shape)
        # #self.self_attn = MultiHeadedAttention(4, 384, 0.1).to(device='cuda:0') # produces dimension error
        # multihead_attn = nn.MultiheadAttention(384, 16, 0.1, batch_first=True).to(device='cuda:0')
        # attn_output, attn_weights = multihead_attn(x_q, x_acc_emb, x_acc_emb)
        # print("attn_weights.shape: ", attn_weights.shape)
        # # in inference
        # # make all other speakers zeros
        # #x_acc_emb[:,1:,:] = 0

        # # visualize attn_weights in a plot
        # # assuming attn_weights has shape (batch_size, query_length, key_length)
        # attn_weights_0 = attn_weights[0].to(device='cpu')

        # # apply softmax to get values between 0 and 1 that sum to 1
        # attn_weights_softmax_0 = F.softmax(attn_weights_0, dim=1)
        
        # # plot the attention weights as a heatmap
        # plt.imshow(attn_weights_softmax_0.detach().numpy(), cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.savefig('heatmap.png')

        # #print("attn_output.shape: ", attn_output.shape, "attn_weights.shape: ", attn_weights.shape, "\n")
        # x = residual + self.dropout(attn_output)
        # x = self.norm_mha(x)
        # # print("pos_emb.shape: ", pos_emb.shape)
        # # print("mask.shape: ", mask.shape)
        # # x = self.self_attn(x_q, x_acc_emb, x_acc_emb, pos_emb)

        # # side info: it appears, that https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention.forward does a call to F.multi_head_attention_forward(... which in github shows as: https://github.com/pytorch/pytorch/blob/8372c5dc687d622c7d2e0d411f61cd2720fc1052/torch/nn/functional.py#L5029 see:
        # # scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set to :math:`\frac{1}{\sqrt{E}}`.
        # # where E is the embedding dimension. So it should be the same as in the paper SPEAKER-AWARE SPEECH-TRANSFORMER (formula 7)

#####################
        if pos_emb is not None:
            return (x, pos_emb), mask
        return x, mask
