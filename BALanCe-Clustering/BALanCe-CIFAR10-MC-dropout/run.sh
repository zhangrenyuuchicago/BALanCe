#CUDA_VISIBLE_DEVICES=0 python -u main.py --method PowerBALD  > log_power_bald &
CUDA_VISIBLE_DEVICES=1 python -u main.py --method PowerBALanCe  > log_power_balance &
#CUDA_VISIBLE_DEVICES=0 python -u main.py --method Random > log_random & 
#CUDA_VISIBLE_DEVICES=1 python -u main.py --method BALD > log_bald_cold1.0
#CUDA_VISIBLE_DEVICES=0 python -u main.py --method Variation-Ratio > log_vr 
#CUDA_VISIBLE_DEVICES=0 python -u main.py --method Mean-STD > log_ms 
