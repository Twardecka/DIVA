results_all_disperse top4_by_diva_v1
====================================

Selected seeds from diva_v1 for baselines/diva_v1: 8, 3, 7, 4
Selected seeds from diva_v2 for diva_v2: 3, 6, 8, 1
Ranking rule:
- primary: test_battle_won_mean_best
- tie-break 1: test_return_mean_best
- tie-break 2: env progress metric when available (win_group or match)

Contents:
- sacred/<algorithm-seed...>: copied Sacred run folders for the selected seeds only.
- tb_logs/<algorithm-seed...>: clean TensorBoard logs for the selected seeds only.
- disperse_top4_by_diva_v1_results.csv: flat manifest of the selected runs.
