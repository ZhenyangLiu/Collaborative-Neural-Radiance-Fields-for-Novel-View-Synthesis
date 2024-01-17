CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/leaves_2.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/flower.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10
