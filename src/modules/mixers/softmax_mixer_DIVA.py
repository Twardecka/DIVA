import numpy as np
import torch as th
import torch.nn as nn


def _maybe_spectral_norm(layer, use_spectral_norm):
    if use_spectral_norm:
        return nn.utils.spectral_norm(layer)
    return layer


def _build_hypernet(args, state_dim, output_dim, use_spectral_norm):
    hypernet_layers = getattr(args, "hypernet_layers", 1)

    if hypernet_layers == 1:
        return _maybe_spectral_norm(nn.Linear(state_dim, output_dim), use_spectral_norm)
    elif hypernet_layers == 2:
        hypernet_embed = args.hypernet_embed
        return nn.Sequential(
            _maybe_spectral_norm(nn.Linear(state_dim, hypernet_embed), use_spectral_norm),
            nn.ReLU(),
            _maybe_spectral_norm(nn.Linear(hypernet_embed, output_dim), use_spectral_norm),
        )
    elif hypernet_layers > 2:
        raise Exception("Sorry >2 hypernet layers is not implemented!")
    else:
        raise Exception("Error setting number of hypernet layers.")


class SoftmaxMixerDIVA(nn.Module):
    def __init__(self, args):
        super(SoftmaxMixerDIVA, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.use_spectral_norm = getattr(args, "diva_use_spectral_norm", True)
        self.hyper_gates = _build_hypernet(
            args,
            self.state_dim,
            self.n_agents,
            self.use_spectral_norm,
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        logits = self.hyper_gates(states)
        gates = th.softmax(logits, dim=-1)
        q_tot = th.sum(gates * agent_qs, dim=-1, keepdim=True)

        return q_tot.view(bs, -1, 1)
