| run_name                                               | architecture                             | eval_mode           |   AP_all_events |
|:-------------------------------------------------------|:-----------------------------------------|:--------------------|----------------:|
| seq_first_smoke_fullweek_eval_check                    | seq-first + XGB + h20+burst + U0.01      | full_week_unsampled |      0.0066343  |
| seq_first_full_006_h20_burst_fullweek_eval_u001_cat_v4 | seq-first + CATBOOST + h20+burst + U0.01 | full_week_unsampled |      0.00377148 |
| seq_first_full_005_h20_burst_fullweek_eval_u001        | seq-first + XGB + h20+burst + U0.01      | full_week_unsampled |      0.00206579 |
| seq_first_full_004_h20_burst_lbproxy_u001              | seq-first + XGB + h20+burst + U0.01      | sampled_week_only   |      0.811704   |