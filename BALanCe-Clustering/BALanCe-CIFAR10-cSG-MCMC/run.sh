CUDA_VISIBLE_DEVICES=0 python -u main.py --method Random > log_random  &
CUDA_VISIBLE_DEVICES=0 python -u main.py --method BADGE > log_badge &
CUDA_VISIBLE_DEVICES=1 python -u main.py --method CoreSet > log_coreset &
CUDA_VISIBLE_DEVICES=1 python -u main.py --method PowerBALD > log_power_bald &
#CUDA_VISIBLE_DEVICES=1 python -u main.py --method PowerBALanCe > log_power_balance &
