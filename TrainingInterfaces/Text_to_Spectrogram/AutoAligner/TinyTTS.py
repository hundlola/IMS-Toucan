import torch
import torch.multiprocessing
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch

from Utility.utils import make_non_pad_mask

class TinyTTS(torch.nn.Module):

    def __init__(self,
                 n_mels=80,
                 num_symbols=145,
                 lstm_dim=512):
        super().__init__()
        self.in_proj = torch.nn.Linear(num_symbols, lstm_dim)
        self.rnn1 = torch.nn.LSTM(lstm_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.rnn2 = torch.nn.LSTM(2*lstm_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.out_proj =  torch.nn.Linear(2*lstm_dim, n_mels)
        self.l1_criterion = torch.nn.L1Loss(reduction="none")


    def forward(self, x, lens, ys):
        x = self.in_proj(x)
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.out_proj(x)
        out_masks = make_non_pad_mask(lens).unsqueeze(-1).to(ys.device)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= ys.size(0) * ys.size(2)
        l1_loss = self.l1_criterion(x, ys).mul(out_weights).masked_select(out_masks).sum()
        return l1_loss