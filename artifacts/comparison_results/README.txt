comparison_results
==================

Generated files:
- csv/<env>_results.csv: one row per Sacred run for gather, hallway, and disperse.
- tb_logs/<env>/<algorithm>-seed<seed>[...]: TensorBoard export with clean per-run names.

Algorithm labels:
- diva_v1 = diva_bounded_sigmoid_qmix_DIVA
- diva_v2 = diva_bounded_sigmoid_qmix_DIVA_scale1_capacity64_gate2

- diva_v3 = results/sacred/13-20 (diva_bounded_sigmoid_qmix_DIVA_vscale1_rmax5)

Run counts by environment:
- gather: 31 runs
- hallway: 29 runs
- disperse: 30 runs
