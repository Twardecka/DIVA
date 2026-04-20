import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _sn_linear(in_features, out_features, use_spectral_norm):
    layer = nn.Linear(in_features, out_features)
    if use_spectral_norm:
        return nn.utils.spectral_norm(layer)
    return layer


class SpectralGRUCellDIVA(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_spectral_norm):
        super(SpectralGRUCellDIVA, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.x_r = _sn_linear(input_dim, hidden_dim, use_spectral_norm)
        self.h_r = _sn_linear(hidden_dim, hidden_dim, use_spectral_norm)
        self.x_z = _sn_linear(input_dim, hidden_dim, use_spectral_norm)
        self.h_z = _sn_linear(hidden_dim, hidden_dim, use_spectral_norm)
        self.x_n = _sn_linear(input_dim, hidden_dim, use_spectral_norm)
        self.h_n = _sn_linear(hidden_dim, hidden_dim, use_spectral_norm)

    def forward(self, x, h):
        r = th.sigmoid(self.x_r(x) + self.h_r(h))
        z = th.sigmoid(self.x_z(x) + self.h_z(h))
        n = th.tanh(self.x_n(x) + r * self.h_n(h))
        return (1 - z) * n + z * h


class RNNAgentDIVA(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentDIVA, self).__init__()
        self.args = args
        self.use_spectral_norm = getattr(args, "diva_use_spectral_norm", True)
        self.r_max = getattr(args, "diva_r_max", 10.0)

        self.fc1 = _sn_linear(input_shape, args.rnn_hidden_dim, self.use_spectral_norm)
        self.rnn = SpectralGRUCellDIVA(args.rnn_hidden_dim, args.rnn_hidden_dim, self.use_spectral_norm)
        self.fc2 = _sn_linear(args.rnn_hidden_dim, args.n_actions, self.use_spectral_norm)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.r_max * th.tanh(self.fc2(h))
        return q, h
