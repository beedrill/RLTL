import os, sys, subprocess
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    #import traci
    #f.close()
else:
    print('warning: no SUMO_HOME declared')

import traci
from time import time
import random
import numpy as np
from simulator import Simulator, TrafficLightLuxembourg
from parseResult import parse_result

weight_dir_prefix = 'DQN_weights/luxemboug-DUE-12408-pen02-run10/'

net_itr = 150000
gap = 150000
penetration_rate = 0.2

while net_itr<=4500000:
    #something here
    weight_file_name = "{}DQN_SUMO_{}_weights".format(weight_dir_prefix, net_itr)
    os.system('python run_multiagents_rltl.py --load {} --whole_day --mode test --phase_representation original --record --penetration_rate {} --Luxembourg_intersection 0 --sumo --record_file records/record_{}'
    .format(weight_file_name, penetration_rate, net_itr))

    ##
    net_itr += gap
