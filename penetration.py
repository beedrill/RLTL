import os
penetration_rates = [0.2, 0.4,0.6,0.8,1]
names = ['02','04','06','08','10']
time = 8
#prv_dir_name = 'Luxembourg-time8-pen00'
#os.system(cmd)
for pr, n in zip(penetration_rates, names):
    dir_name = 'Luxembourg-time{}-pen{}'.format(time, n)
    append = '--load DQN_weights/{}-detailed/DQN_SUMO_best_weights'.format(dir_name)
    #if prv_dir_name:
        #append = '--load DQN_weights/{}/DQN_SUMO_best_weights'.format(prv_dir_name)


    cmd = 'python run_multiagents_rltl.py --pysumo --no_counter   --dir_name  {}  --penetration_rate {} {}'.format(dir_name, pr, append)
    os.system(cmd)
    prv_dir_name = dir_name
