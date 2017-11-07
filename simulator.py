import pysumo
from tqdm import tqdm
import random
from time import time

class Simulator():
    def __init__(self, visual = False, map_file = 'map/traffic.net.xml', route_file = 'map/traffic.rou.xml', end_time = 500, additional_file = None):
        self.visual = visual
        self.map_file = map_file
        self.end_time = end_time
        self.route_file = route_file
        self.additional_file = additional_file
        if self.visual == False:
            self.cmd = ['sumo', 
                  '--net-file', self.map_file, 
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
            if not additional_file == None:      
                self.cmd.append('--additional-files', self.additional_file)
                
    def simulation_start(self):
        if self.visual == False:
            pysumo.simulation_start(self.cmd)
            return


actions = ['rGrG','ryry','GrGr','yryr']
def random_action():
	return random.choice(actions)
  
time_start = time()
for i in tqdm(range(500)):
	pysumo.simulation_start(cmd)
#	print 'lanes:', pysumo.tls_getControlledLanes('0');
	print 'all lanes', pysumo.lane_list();
	for j in range(1000):
		pysumo.tls_setstate("0",random_action())
		pysumo.simulation_step()
		ids =  pysumo.lane_onLaneVehicles("0_n_0")
		if ids:
			print ids;
	pysumo.simulation_stop()
time_end = time()
print "pysumo time elapsed: {}".format(time_end-time_start)
