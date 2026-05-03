# DIVA Verification Bundle

This folder contains the files most relevant for verifying the current
theorem-oriented DIVA implementation and the exact variant we are running now.

Current main config:
- `src/config/algs/diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5.yaml`

Recommended reading order:
1. `src/config/algs/diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5.yaml`
   Current algorithm settings, including the bounded scales.
2. `src/modules/agents/rnn_agent_DIVA.py`
   Per-agent recurrent utility network with spectral normalization and bounded
   `tanh` outputs.
3. `src/modules/mixers/bounded_sigmoid_qmix_mixer_DIVA.py`
   Core DIVA mixer: state-dependent positive gates, bounded positive monotone
   weights, bounded signed bias and `V(s)`.
4. `src/learners/q_learner_DIVA.py`
   Training rule, target construction, double Q-learning, TD loss, logging.
5. `src/controllers/basic_controller_DIVA.py`
   How agent inputs are built and how decentralized execution is carried out.
6. `src/components/action_selectors.py`
   Epsilon-greedy action selection and evaluation behavior.
7. `src/runners/episode_runner.py`
   Episode collection, test logging, and metric naming.
8. `src/config/default.yaml`
   Default optimization and training hyperparameters inherited by the run.

Registry helper files:
- `src/modules/agents/__init__.py`
- `src/controllers/__init__.py`
- `src/learners/__init__.py`

These show how the config names map to the concrete DIVA classes:
- `agent: rnn_DIVA`
- `mac: basic_mac_DIVA`
- `learner: q_learner_DIVA`

Theory note:
- `artifacts/mixers/diva_bounded_sigmoid_qmix_DIVA_theorem_README.md`

This note summarizes the intended theorem-aligned assumptions:
- bounded utilities and bounded gates/weights
- Lipschitz control via spectral normalization and bounded activations
- strictly positive gates
- coordinate-wise monotonicity in the gated utilities

Practical summary of the current method:
- Each agent outputs bounded utilities `q_i = r_max * tanh(f_i(...))`
- The mixer computes positive state-dependent gates `g_i(s)`
- Utilities are gated as `x_i = g_i(s) * q_i`
- A QMIX-style monotone hypernetwork mixer combines the gated utilities into
  `Q_tot`
- The model is trained with replay, target networks, double Q-learning, and a
  masked squared TD loss

All copied files preserve their original repo-relative paths so the bundle is
easy to inspect or share for external verification.
