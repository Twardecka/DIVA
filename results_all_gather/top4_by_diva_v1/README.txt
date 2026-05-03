results_all_gather top4_by_diva_v1
==================================

Selected seeds from diva_v1 for baselines/diva_v1: 7, 5, 4, 8
Selected seeds from diva_v2 for diva_v2: 5, 3, 8, 4
Ranking rule:
- primary: test_battle_won_mean_best
- tie-break 1: test_return_mean_best
- tie-break 2: env progress metric when available (win_group or match)

Contents:
- sacred/<algorithm-seed...>: copied Sacred run folders for the selected seeds only.
- tb_logs/<algorithm-seed...>: clean TensorBoard logs for the selected seeds only.
- gather_top4_by_diva_v1_results.csv: flat manifest of the selected runs.
