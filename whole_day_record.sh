#!/bin/bash
for i in {0..23};
do
python run_multiagents_rltl.py --mode test --load DQN_weights/5-intersections-pen02-wholeday/DQN_SUMO_best_weights --pysumo --whole_day --record --penetration_rate 0.2 --day_time $i
#echo $i
done
