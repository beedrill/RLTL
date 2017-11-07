import pysumo
#commend this line if you dont have pysumo and set visual = True, it should still run traci
import traci
from time import time

class Simulator():
    def __init__(self, visual = False, map_file = 'map/traffic.net.xml', route_file = 'map/traffic.rou.xml', end_time = 500, additional_file = None):
        self.visual = visual
        self.map_file = map_file
        self.end_time = end_time
        self.route_file = route_file
        self.additional_file = additional_file
        self.veh_list = {}
        self.tl_list = {}
        self.time = 0
        lane_list = ['0_e_0', '0_n_0','0_s_0','0_w_0','e_0_0','n_0_0','s_0_0','w_0_0'] # temporary, in the future, get this from the .net.xml file
        self.lane_list = {l:Lane(l,self) for l in lane_list}
        tl_list = ['0'] # temporary, in the future, get this from .net.xml file
        for tlid in tl_list:
            self.tl_list[tlid] = SimpleTrafficLight(tlid, self)
        if self.visual == False:
            self.cmd = ['sumo', 
                  '--net-file', self.map_file, 
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
            if not additional_file == None:      
                self.cmd.append('--additional-files', self.additional_file)
        else:
            self.cmd = ['sumo-gui', 
                  '--net-file', self.map_file, 
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
        if not additional_file == None:      
            self.cmd.append('--additional-files', self.additional_file)
                
    def _simulation_start(self):
        if self.visual == False:
            pysumo.simulation_start(self.cmd)
            return
        else:
            traci.start(self.cmd)
            return
    
        
    def _simulation_end(self):
        if self.visual == False:
            pysumo.simulation_stop()
            return
        else:
            traci.close()
            return
        
    def _simulation_step(self):
        if self.visual == False:
            pysumo.simulation_step()
            self.time += 1
            return
        else:
            self.time += 1
            traci.simulationStep()
            return 
        
    def step(self):
        self._simulation_step()
        for l in self.lane_list:
            self.lane_list[l].step()
        for tlid in self.tl_list:
            self.tl_list[tlid].step()
            
    def start(self):
        self._simulation_start()
    
    def stop(self):
        self._simulation_end()
        
    def print_status(self):
        #print self.veh_list
        print 'current time:', self.time, ' total cars:', len(self.veh_list.keys())
   # def _update_vehicles(self):
   #     if self.visual == False:
   #         self.current_veh_list = pysumo.vehicle_list()
   #         return
        
class Lane():
    def __init__(self,lane_id,simulator,length = 251):
        self.id = lane_id
        self.simulator = simulator
        self.vehicle_list = []
        self.length = length
        
    def _get_vehicles(self):
        if self.simulator.visual == False:
            return pysumo.lane_onLaneVehicles(self.id)
        else:
            return traci.lane.getLastStepVehicleIDs(self.id)
        
    def step(self):
        vidlist = self._get_vehicles()
        for vid in vidlist:
            if not vid in self.simulator.veh_list.keys():
                self.simulator.veh_list[vid]= Vehicle(vid,self.simulator)
                
            self.simulator.veh_list[vid].lane = self
            self.simulator.veh_list[vid].step()
            
            
class Vehicle():
    def __init__(self, vid, simulator, equipped = True,):
        self.id = vid
        self.simulator = simulator
        self.depart_time = simulator.time
        self.latest_time = simulator.time
        self.waiting_time = 0
        self.equipped = equipped
    
    def _update_speed(self):
        if self.simulator.visual == False:
            self.speed = pysumo.vehicle_speed(self.id)
            return 
        else:
            self.speed = traci.vehicle.getSpeed(self.id)
            return
        
    def _update_lane_position(self):
        if self.simulator.visual == False:
            self.lane_position = self.lane.length - pysumo.vehicle_lane_position(self.id)
            return
        else:
            self.lane_position = self.lane.length - traci.vehicle.getLanePosition(self.id)
            return
    
    def step(self):
        self._update_speed()
        self._update_lane_position()
        self.latest_time = self.simulator.time
        if self.speed < 1:
            self.waiting_time += 1
            
class TrafficLight():
    
    
    def __init__(self, tlid, simulator):
        self.id = tlid
        self.simulator = simulator

    
    def _set_phase(self, phase):
        self.current_phase = phase
        if self.simulator.visual == False:
            pysumo.tls_setstate(self.id, self.actions[phase])
            return
        else:
            traci.trafficlights.setRedYellowGreenState(self.id,self.actions[phase])
            return
        
    def step(self):
        print 'specify this method in subclass before use'
        pass

class SimpleTrafficLight(TrafficLight):
    def __init__(self, tlid, simulator, max_time= 30, yellow_time = 3):
        
        TrafficLight.__init__(self, tlid, simulator)
        self.actions = ['rGrG','ryry','GrGr','yryr']
        self.current_pahse = 0 # pahse can be 0, 1, 2, 3
        self.current_phase_time = 0
        self.max_time = max_time
        self.yellow_time = yellow_time
        
    def step(self):
        self.current_phase_time+=1
        if self.current_pahse in [0,2]:
            if self.current_phase_time>self.max_time:
                self.move_to_next_phase()
        else:
            if self.current_phase_time > self.yellow_time:
                self.move_to_next_phase()
                
            
    def move_to_next_phase(self):
        self.current_pahse = (self.current_pahse+1)%len(self.actions)
        self._set_phase(self.current_pahse)
        self.current_phase_time = 0
            
  
if __name__ == '__main__':
    sim = Simulator()
    #sim = Simulator(visual = True)
    # use this commend if you don't have pysumo installed
    sim.start()
    for i in range(0,1000):
        sim.step()
        sim.print_status()
    sim.stop()
        
