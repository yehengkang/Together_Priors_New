
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 train_yhk.py

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_yhk.py

source /home/omnisky/anaconda3/bin/activate Priors

https://aistudio.google.com/