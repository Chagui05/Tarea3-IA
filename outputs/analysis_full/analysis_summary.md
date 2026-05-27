# Analysis summary

Top ROC AUC by run:

| run                 |      auc |       ap |
|:--------------------|---------:|---------:|
| vae_ssim+l1_gpu20   | 0.651068 | 0.754582 |
| u-net_l1_gpu20      | 0.608516 | 0.788095 |
| vae_l1_gpu20        | 0.501114 | 0.694253 |
| u-net_l2_gpu20      | 0.466331 | 0.647499 |
| vae_ssim_gpu20      | 0.445314 | 0.664465 |
| vae_l2_gpu20        | 0.407784 | 0.66857  |
| u-net_ssim_gpu20    | 0.349283 | 0.605094 |
| u-net_ssim+l1_gpu20 | 0.318559 | 0.614529 |

Per-run/per-class AUC saved in `auc_by_run_class.csv`.
