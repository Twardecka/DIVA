runs_baselines
==============

This bundle contains CPU-only 1M jobs with seeds 1, 3, 5, 8.

Groups:
- diva_vdn_like_1m: 12 jobs
  config: diva_bounded_sigmoid_vdn_DIVA
  envs: hallway, disperse, sensor

- baselines_1m: 12 jobs
  configs: qmix, qtran, vdn
  envs: sensor only

- sensor_diva_gpu_version_1m: 4 jobs
  config: diva_bounded_sigmoid_qmix_DIVA_vscale1_clean
  env: sensor

All jobs use the explicit Sacred override:
- t_max=1005000

See job_manifest.csv for the exact mapping from run_N.job to config/env/seed.
