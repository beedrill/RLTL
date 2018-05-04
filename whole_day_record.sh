#!/bin/bash
for i in {0..23};
do
python run_rltl.py --mode test --load DQN_weights/TL-run16/DQN_SUMO_best_weights.hdf5 --pysumo --whole_day --record --penetration_rate 0.2 --day_time $i
#echo $i
done
