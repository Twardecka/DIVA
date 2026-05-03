import numpy as np
import torch as th
import torch.nn as nn

from .bounded_sigmoid_qmix_mixer_DIVA import _bounded_positive, _bounded_signed, _build_hypernet


class BoundedSigmoidVDNMixerDIVA(nn.Module):
    def __init__(self, args):
        super(BoundedSigmoidVDNMixerDIVA, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.use_spectral_norm = getattr(args, "diva_use_spectral_norm", True)

        self.offset = getattr(args, "diva_positive_offset", 1e-3)
        self.gate_scale = getattr(args, "diva_gate_scale", 1.0)
        self.v_scale = getattr(args, "diva_v_scale", 5.0)

        self.hyper_gates = _build_hypernet(
            args,
            self.state_dim,
            self.n_agents,
            self.use_spectral_norm,
        )
        self.V = _build_hypernet(
            args,
            self.state_dim,
            1,
            self.use_spectral_norm,
        )
        self.latest_stats = {}

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        device = next(self.parameters()).device

        agent_qs = agent_qs.to(device).view(-1, self.n_agents)
        states = states.to(device).reshape(-1, self.state_dim)

        gates = _bounded_positive(self.hyper_gates(states), self.offset, self.gate_scale)
        gated_qs = gates * agent_qs
        v = _bounded_signed(self.V(states), self.v_scale)
        q_tot = gated_qs.sum(dim=1, keepdim=True) + v

        self.latest_stats = {
            "gate_mean": gates.mean().detach(),
            "gate_std": gates.std(unbiased=False).detach(),
            "gate_min": gates.min().detach(),
            "gate_max": gates.max().detach(),
            "gated_q_mean": gated_qs.mean().detach(),
            "v_mean": v.mean().detach(),
        }

        return q_tot.view(bs, -1, 1)

    def get_logging_stats(self):
        return {name: value.item() for name, value in self.latest_stats.items()}
