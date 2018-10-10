declare -a datasets=("nus_wide" "ms_coco")
declare -a init_opts=("imagenet") # random
declare -a losses=("robust_warp_sup" "warp")
declare -a labeled_ratios=("10" "20")
declare -a corruption_ratios=("0" "10" "20" "30" "40" "50")

for dataset in "${datasets[@]}"
do
	for init_opt in "${init_opts[@]}"
	do
		for loss in "${losses[@]}"
		do
			for labeled_ratio in "${labeled_ratios[@]}"
			do
				for corruption_ratio in "${corruption_ratios[@]}"
				do
					python main.py $dataset $init_opt $loss $labeled_ratio $corruption_ratio
				done
			done
		done
	done
done
