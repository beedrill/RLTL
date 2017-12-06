python run_rltl.py --cpu --pysumo --dir_name penetration_add_yellow_test10 --no_counter --penetration_rate 1
python run_rltl.py --cpu --pysumo --dir_name penetration_add_yellow_test8 --no_counter --penetration_rate 0.8 --load DQN_weights/penetration_add_yellow_test10/DQN_SUMO_bestweights.hdf5
python run_rltl.py --cpu --pysumo --dir_name penetration_add_yellow_test6 --no_counter --penetration_rate 0.6 --load DQN_weights/penetration_add_yellow_test8/DQN_SUMO_bestweights.hdf5
python run_rltl.py --cpu --pysumo --dir_name penetration_add_yellow_test4 --no_counter --penetration_rate 0.4 --load DQN_weights/penetration_add_yellow_test6/DQN_SUMO_bestweights.hdf5
python run_rltl.py --cpu --pysumo --dir_name penetration_add_yellow_test2 --no_counter --penetration_rate 0.2 --load DQN_weights/penetration_add_yellow_test4/DQN_SUMO_bestweights.hdf5
python run_rltl.py --cpu --pysumo --dir_name penetration_add_yellow_test1 --no_counter --penetration_rate 0.1 --load DQN_weights/penetration_add_yellow_test2/DQN_SUMO_bestweights.hdf5
python run_rltl.py --cpu --pysumo --dir_name penetration_add_yellow_test0 --no_counter --penetration_rate 0 --load DQN_weights/penetration_add_yellow_test1/DQN_SUMO_bestweights.hdf5
