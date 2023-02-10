#CUDA_VISIBLE_DEVICES=0 python -u main.py --method CoreSet > log_coreset 
CUDA_VISIBLE_DEVICES=1 python -u main.py --method BADGE > log_badge & 
