"""
Taken from ESPNet
"""

import torch
import torch.nn.functional as F
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt

from Layers.Attention import RelPositionMultiHeadedAttention
from Layers.Convolution import ConvolutionModule
from Layers.EncoderLayer import EncoderLayer
from Layers.LayerNorm import LayerNorm
from Layers.MultiLayeredConv1d import MultiLayeredConv1d
from Layers.MultiSequential import repeat
from Layers.PositionalEncoding import RelPositionalEncoding
from Layers.Swish import Swish


class Conformer_accent_mha(torch.nn.Module):
    """
    Conformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Conformer positional encoding layer type.
        selfattention_layer_type (str): Conformer attention layer type.
        activation_type (str): Conformer activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(self, idim, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0, input_layer="conv2d", normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1,
                 macaron_style=False, use_cnn_module=False, cnn_module_kernel=31, zero_triu=False, utt_embed=None, connect_utt_emb_at_encoder_out=True,
                 spk_emb_bottleneck_size=128):
        super(Conformer_accent_mha, self).__init__()

        activation = Swish()
        self.conv_subsampling_factor = 1

        if isinstance(input_layer, torch.nn.Module):
            self.embed = input_layer
            self.pos_enc = RelPositionalEncoding(attention_dim, positional_dropout_rate)
        elif input_layer is None:
            self.embed = None
            self.pos_enc = torch.nn.Sequential(RelPositionalEncoding(attention_dim, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before

        self.connect_utt_emb_at_encoder_out = connect_utt_emb_at_encoder_out
        if utt_embed is not None:
            self.hs_emb_projection = torch.nn.Linear(attention_dim + spk_emb_bottleneck_size, attention_dim)
            # embedding projection derived from https://arxiv.org/pdf/1705.08947.pdf
            self.embedding_projection = torch.nn.Sequential(torch.nn.Linear(utt_embed, spk_emb_bottleneck_size),
                                                            torch.nn.Softsign())

        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, attention_dim, attention_dropout_rate, zero_triu)

        # feed-forward module definition
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate,)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(num_blocks, lambda lnum: EncoderLayer(attention_dim, encoder_selfattn_layer(*encoder_selfattn_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                                                                     convolution_layer(*convolution_layer_args) if use_cnn_module else None, dropout_rate,
                                                                     normalize_before, concat_after))
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, xs, masks, utterance_embedding=None, lang_embs=None):
        """
        Encode input sequence.
        Args:
            utterance_embedding: embedding containing lots of conditioning signals
            step: indicator for when to start updating the embedding function
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if self.embed is not None:
            xs = self.embed(xs)

        #print("setting lang_emb to None manually")
        lang_embs = None
        if lang_embs is not None:
            xs = xs + torch.cat([lang_embs,lang_embs],dim=-1).unsqueeze(1) 

        if utterance_embedding is not None and not self.connect_utt_emb_at_encoder_out:
            xs = self._integrate_with_utt_embed(xs, utterance_embedding)

        xs = self.pos_enc(xs)

        xs, masks = self.encoders(xs, masks)    # return (x, pos_emb), mask
        if isinstance(xs, tuple):
            xs = xs[0]

#####################
        x_q = xs.to(device='cuda:0')
        print("x_q.shape: ", x_q.shape)    # x_q.shape:  torch.Size([8, 55, 384])
        batch = x_q.shape[0] # save batch size as it is in the query
        residual = xs
        self.accent_emb = []
        c=0
        for embs in next(os.walk('/nas/projects/vokquant/IMS-Toucan_lang_emb_conformer/Preprocessing/embeds_mls_test/trained_on_12_but_only_wass/'))[2]:
            self.accent_emb.append(torch.load(os.path.join('/nas/projects/vokquant/IMS-Toucan_lang_emb_conformer/Preprocessing/embeds_mls_test/trained_on_12_but_only_wass/', embs )))
            print(str(c) + ": " + str(embs))
            c+=1
        self.accent_emb = np.concatenate(self.accent_emb)       # created shape: (12, 1, 192)
        self.accent_emb = torch.Tensor(self.accent_emb).repeat(1,1,1,2)
        self.accent_emb = self.accent_emb.squeeze(0)
        x_acc_emb = self.accent_emb.permute(1, 0, 2).repeat(batch,1,1).to(device='cuda:0') # [8, 12, 384]
        x_acc_emb[:,:1,:] = 0 # in inference you can try to make all speakers in x_acc_emb zero except one
        x_acc_emb[:,2:,:] = 0
        # x_acc_emb[:,11,:] = 0
        multihead_attn = nn.MultiheadAttention(384, 16, 0.1, batch_first=True).to(device='cuda:0')
        attn_output, attn_weights = multihead_attn(x_q, x_acc_emb, x_acc_emb)
        #print("attn_output.shape: ", attn_output.shape, "attn_weights.shape: ", attn_weights.shape, "\n")  # example: attn_output.shape:  torch.Size([8, 55, 384]) attn_weights.shape:  torch.Size([8, 55, 12])

        visualize_attn_weights = True
        if visualize_attn_weights:
            #print("attn_weights[0]: ", attn_weights[0], "attn_weights.shape: ", attn_weights.shape)
            attn_weights_0 = attn_weights[0].to(device='cpu') # visualize attn_weights in a plot, assuming attn_weights has shape (batch_size, query_length, key_length)
            attn_weights_0 = F.softmax(attn_weights_0, dim=1) # apply softmax to get values between 0 and 1 that sum to 1
            fig, ax = plt.subplots(figsize=(6, 6)) # plot the attention weights as a heatmap
            im = ax.imshow(attn_weights_0.detach().numpy(), cmap='hot', interpolation='nearest', extent=[-0.5, 11.5, -0.5, 54.5])
            ax.set_xticks(range(attn_weights.shape[2]))
            ax.set_xticklabels([str(i) for i in range(attn_weights.shape[2])])
            plt.colorbar(im)
            plt.savefig('heatmap.png')

        attn_output = attn_output.to(device='cuda:0')
        #residual connection around mha:
        attn_output = (x_q + attn_output)

        # create an instance of PositionwiseFeedForward class
        ffn_size = 2048
        ffn = PositionWiseFFN(attn_output.size(-1), ffn_size) # 384, 2048

        # apply position-wise feed-forward network on attn_output
        ffn_output = ffn(attn_output)
        #print("ffn_output.shape: ", ffn_output.shape) # ffn_output.shape:  torch.Size([8, 55, 384])
        # batch_norm = nn.BatchNorm1d(num_features=384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device='cuda:0')
        # attn_output = batch_norm(attn_output.permute(0, 2, 1)).permute(0, 2, 1).to(device='cuda:0')
        
        linear = torch.nn.Linear(2*attn_output.size(-1), attn_output.size(-1)).to(device='cuda:0')
        # concatenate:
        concatenated = torch.cat([xs, ffn_output], dim=-1)
        x = linear(concatenated)

        #print("concatenated.shape: ", concatenated.shape) # concatenated.shape:  torch.Size([8, 55, 768])
        #print("residual.shape: ", residual.shape) # residual.shape:  torch.Size([8, 55, 384])
        #print("self.concat_linear(concatenated).shape: ", self.concat_linear(concatenated).shape) # self.concat_linear(concatenated).shape:  torch.Size([8, 55, 384])
        #print("xs.shape: ", xs.shape) # xs.shape:  torch.Size([8, 55, 384])
        #xs = residual + self.concat_linear(concatenated)
        #xs = np.concatenate([xs, attn_output], axis=-1)
        #xs = residual + self.linear(cat(xs, concatenated))

        # x -> x + linear(concat(x, att(x)))
        #xs = residual + attn_output
        #xs = residual + 0.001*self.dropout(attn_output)
        # OR:
        # x_concat = torch.cat((residual, attn_output), dim=-1)
        # x = residual + self.concat_linear(x_concat)
        #xs = concatenated
        xs = residual + x


#####################

        if self.normalize_before:
            xs = self.after_norm(xs)

        if utterance_embedding is not None and self.connect_utt_emb_at_encoder_out:
            xs = self._integrate_with_utt_embed(xs, utterance_embedding)

        return xs, masks

    def _integrate_with_utt_embed(self, hs, utt_embeddings):
        # project embedding into smaller space
        speaker_embeddings_projected = self.embedding_projection(utt_embeddings)
        # concat hidden states with spk embeds and then apply projection
        speaker_embeddings_expanded = F.normalize(speaker_embeddings_projected).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.hs_emb_projection(torch.cat([hs, speaker_embeddings_expanded], dim=-1))
        return hs

class PositionWiseFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, 2*d_ff).to(device='cuda:0')
        self.fc2 = torch.nn.Linear(d_ff, d_model).to(device='cuda:0')
        self.norm1 = torch.nn.LayerNorm(2*d_ff).to(device='cuda:0')
        self.norm2 = torch.nn.LayerNorm(d_model).to(device='cuda:0')
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.norm1(x)
        x, gate = x.chunk(2, dim=-1)
        x = torch.sigmoid(gate) * x
        x = self.fc2(x)
        x = self.norm2(x)
        x = residual + x
        return x