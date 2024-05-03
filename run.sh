python3 train.py --source_path ./data/kitchen_12 --model_path ./output/kitchen_12_SparseGS --beta 5.0 --lambda_pearson 0.05 --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 --lambda_diffusion 0.001 --step_ratio 0.99 --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 --iterations 30000 --checkpoint_iterations 30000 -r 2
python3 render.py --source_path ./data/kitchen --model_path ./output/kitchen_12_SparseGS --no_load_depth --iteration 30000
python3 metrics.py --model_path ./output/kitchen_12_SparseGS --exclude_path ./data/kitchen_12
