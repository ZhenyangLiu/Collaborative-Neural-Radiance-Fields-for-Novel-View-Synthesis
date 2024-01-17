CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/leaves.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/leaves_1.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10
