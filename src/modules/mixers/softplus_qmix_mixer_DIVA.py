import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _maybe_spectral_norm(layer, use_spectral_norm):
    if use_spectral_norm:
        return nn.utils.spectral_norm(layer)
    return layer


def _build_hypernet(args, state_dim, output_dim, use_spectral_norm):
    hypernet_layers = getattr(args, "hypernet_layers", 1)

    if hypernet_layers == 1:
        return _maybe_spectral_norm(nn.Linear(state_dim, output_dim), use_spectral_norm)
    if hypernet_layers == 2:
        hypernet_embed = args.hypernet_embed
        return nn.Sequential(
            _maybe_spectral_norm(nn.Linear(state_dim, hypernet_embed), use_spectral_norm),
            nn.ReLU(),
            _maybe_spectral_norm(nn.Linear(hypernet_embed, output_dim), use_spectral_norm),
        )
    raise Exception("Sorry >2 hypernet layers is not implemented!")


class SoftplusQMixerDIVA(nn.Module):
    def __init__(self, args):
        super(SoftplusQMixerDIVA, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.offset = getattr(args, "diva_softplus_offset", 1e-3)
        self.use_spectral_norm = getattr(args, "diva_use_spectral_norm", True)

        # State-dependent positive gates preserve the DIVA-style gated utilities x = g(s) * q.
        self.hyper_gates = _build_hypernet(
            args,
            self.state_dim,
            self.n_agents,
            self.use_spectral_norm,
        )

        # QMIX-style hidden and output weights increase mixer expressivity while keeping
        # the mapping monotone via positive weights.
        self.hyper_w1 = _build_hypernet(
            args,
            self.state_dim,
            self.n_agents * self.embed_dim,
            self.use_spectral_norm,
        )
        self.hyper_b1 = _build_hypernet(
            args,
            self.state_dim,
            self.embed_dim,
            self.use_spectral_norm,
        )
        self.hyper_w_final = _build_hypernet(
            args,
            self.state_dim,
            self.embed_dim,
            self.use_spectral_norm,
        )
        self.V = _build_hypernet(
            args,
            self.state_dim,
            1,
            self.use_spectral_norm,
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        device = next(self.parameters()).device

        agent_qs = agent_qs.to(device).view(-1, self.n_agents)
        states = states.to(device).reshape(-1, self.state_dim)

        gates = F.softplus(self.hyper_gates(states)) + self.offset
        gated_qs = (gates * agent_qs).view(-1, 1, self.n_agents)

        w1 = F.softplus(self.hyper_w1(states)) + self.offset
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(gated_qs, w1) + b1)

        w_final = F.softplus(self.hyper_w_final(states)) + self.offset
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)

        q_tot = th.bmm(hidden, w_final) + v
        return q_tot.view(bs, -1, 1)
