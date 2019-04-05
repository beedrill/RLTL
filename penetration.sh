name="Lux_26640_sparse"
#python3 run_multiagents_rltl.py --pysumo --dir_name ${name}_pen10 --no_counter --penetration_rate 1
python3 run_multiagents_rltl.py --pysumo --dir_name ${name}_pen8 --no_counter --penetration_rate 0.8 --load DQN_weights/${name}_pen10/DQN_SUMO_best_weights
python3 run_multiagents_rltl.py --pysumo --dir_name ${name}_pen6 --no_counter --penetration_rate 0.6 --load DQN_weights/${name}_pen8/DQN_SUMO_best_weights
python3 run_multiagents_rltl.py --pysumo --dir_name ${name}_pen4 --no_counter --penetration_rate 0.4 --load DQN_weights/${name}_pen6/DQN_SUMO_best_weights
python3 run_multiagents_rltl.py --pysumo --dir_name ${name}_pen2 --no_counter --penetration_rate 0.2 --load DQN_weights/${name}_pen4/DQN_SUMO_best_weights
python3 run_multiagents_rltl.py --pysumo --dir_name ${name}_pen1 --no_counter --penetration_rate 0.1 --load DQN_weights/${name}_pen2/DQN_SUMO_best_weights
python3 run_multiagents_rltl.py --pysumo --dir_name ${name}_pen0 --no_counter --penetration_rate 0 --load DQN_weights/${name}_pen1/DQN_SUMO_best_weights
