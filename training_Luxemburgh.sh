#training:
python run_multiagents_rltl.py --cpu --pysumo --whole_day --phase_representation original --simulator original
#testing:
python run_multiagents_rltl.py --cpu --load DQN_weights/TL-run2/DQN_SUMO_3900000_weights --whole_day --day_time 8 --mode test --phase_representation original
