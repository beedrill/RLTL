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
