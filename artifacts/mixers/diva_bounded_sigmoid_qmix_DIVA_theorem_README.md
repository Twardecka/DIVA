`diva_bounded_sigmoid_qmix_DIVA` theorem-oriented package

Included files:
- `src/modules/mixers/bounded_sigmoid_qmix_mixer_DIVA.py`
- `src/config/algs/diva_bounded_sigmoid_qmix_DIVA.yaml`
- `src/learners/q_learner_DIVA.py`
- `src/modules/agents/rnn_agent_DIVA.py`

Why these files:
- The mixer implements explicit gated utilities `x = g(s) * q`, bounded positive dynamic weights, a monotone hidden layer, and `V(s)`.
- The config keeps `diva_use_spectral_norm: True` and the bounded positive scales enabled.
- The learner hook registers the new mixer.
- The DIVA agent uses spectral normalization and bounded `tanh` outputs, which are part of the utility-side theorem assumptions.

Intended theorem-aligned properties:
- Assumption 1: bounded utilities and bounded gates/weights
- Assumption 2: Lipschitz control via spectrally normalized state-conditioned networks and bounded activations
- Assumption 3: strictly positive gates
- Assumption 4: coordinate-wise monotonicity in the gated utilities

Main run command:

```bash
/home/ludwika/miniconda3/envs/gacg/bin/python src/main.py --config=diva_bounded_sigmoid_qmix_DIVA --env-config=pursuit with seed=1 use_cuda=True
```
