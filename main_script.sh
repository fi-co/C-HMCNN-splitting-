# Description: Script to run the main_split.py file with different seeds
for seed in 0 1 2 3 4 5 6 7 8 9
do
    python main_split.py --dataset seq_FUN --seed $seed --device 3 --num_splits 3&
done