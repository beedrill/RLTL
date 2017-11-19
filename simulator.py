import pysumo
# comment this line if you dont have pysumo and set visual = True, it should still run traci
# Todo: another class for another kind of traffic state formation
import traci
from time import time
import numpy as np


class Simulator():
    """
    mimic openAI gym environment
        action_space
        reset?
            restart simulation
        step
            TL agent makes a step
            params:
                action
            returns:
                observation, reward, isterminal, info
    """
    def __init__(self, visual = False, map_file = 'map/traffic.net.xml', route_file = 'map/traffic.rou.xml', end_time = 3600, episode_time = 1000, additional_file = None, gui_setting_file = "map/view.settings.xml"):
        self.visual = visual
        self.map_file = map_file
        self.end_time = end_time
        self.route_file = route_file
        self.additional_file = additional_file
        self.gui_setting_file = gui_setting_file
        self.veh_list = {}
        self.tl_list = {}
        self.time = 0
        lane_list = ['0_e_0', '0_n_0','0_s_0','0_w_0','e_0_0','n_0_0','s_0_0','w_0_0'] # temporary, in the future, get this from the .net.xml file
        self.lane_list = {l:Lane(l,self) for l in lane_list}
        tl_list = ['0'] # temporary, in the future, get this from .net.xml file
        for tlid in tl_list:
            self.tl_list[tlid] = SimpleTrafficLight(tlid, self)
        ###RL parameters
       
        ##############
        self.episode_time = episode_time
        self.action_space = ActionSpaces(len(tl_list), 2) # action = 1 means move to next phase, otherwise means stay in current phase

        if self.visual == False:
            self.cmd = ['sumo', 
                  '--net-file', self.map_file, 
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
            if not additional_file == None:      
                self.cmd+=['--additional-files', self.additional_file]
            
        else:
            self.cmd = ['sumo-gui', 
                  '--net-file', self.map_file, 
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
        if not additional_file == None:      
            self.cmd+=['--additional-files', self.additional_file]
        if not gui_setting_file == None:
            self.cmd+=['--gui-settings-file', self.gui_setting_file]
                
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
        
    def step(self, actions):
        self._simulation_step()
        for l in self.lane_list:
            self.lane_list[l].step()

        observation = []
        reward = []
        i = 0
        for tlid in self.tl_list:
            tl = self.tl_list[tlid]
            #print actions
            tl.step(actions[i])
            observation.append(tl.traffic_state)
            reward.append(self.tl_list[tlid].reward)
            i += 1

        observation = np.array(observation)
        reward = np.array(reward)
        info = (self.time, len(self.veh_list.keys()))
        #if not type( observation[0][0]) in ['int',np.float64]:
        #    print 'something wrong', observation[0][0], type(observation[0][0])
        
        return observation, reward, self.time == self.episode_time, info

    def start(self):
        self._simulation_start()
    
    def stop(self):
        self._simulation_end()
        
    def reset(self):
        self.stop()
        self.time = 0
        self.start()
        observation = []
        reward = []
        info = (self.time, len(self.veh_list.keys()))
        for l in self.lane_list:
            self.lane_list[l].update_lane_reward()
        for tlid in self.tl_list:
            tl = self.tl_list[tlid]
            #print actions
            #tl.step(actions[i])
            tl.move_to_next_phase()
            tl.updateRLParameters()
            observation.append(tl.traffic_state)
            reward.append(self.tl_list[tlid].reward)
            #i += 1
        return observation, reward, self.time == self.episode_time, info
        
        
    def print_status(self):
        #print self.veh_list
        print 'current time:', self.time, ' total cars:', len(self.veh_list.keys()), 'traffic status', self.tl_list['0'].traffic_state, 'reward:', self.tl_list['0'].reward

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
        self.car_number = 0
        self.detected_car_number = 0
        self.lane_reward = 0
        
    def update_lane_reward(self):
        self.lane_reward = 0
        for vid in self.vehicle_list:
            self.lane_reward+=(Vehicle.max_speed - self.simulator.veh_list[vid].speed)/Vehicle.max_speed
        #self.lane_reward = - self.lane_reward

    def _get_vehicles(self):
        if self.simulator.visual == False:
            return pysumo.lane_onLaneVehicles(self.id)
        else:
            return traci.lane.getLastStepVehicleIDs(self.id)
        
    def step(self):
        vidlist = self._get_vehicles()
        self.vehicle_list = vidlist
        self.car_number = len(vidlist)
        self.detected_car_number = 0
        for vid in vidlist:
            if not vid in self.simulator.veh_list.keys():
                self.simulator.veh_list[vid]= Vehicle(vid,self.simulator)
            if self.simulator.veh_list[vid].equipped == True:
                self.detected_car_number += 1
            self.simulator.veh_list[vid].lane = self
            self.simulator.veh_list[vid].step()
        self.update_lane_reward()
            
            
class Vehicle():
    max_speed = 13.9

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
            pysumo.tls_setstate(self.id, self.signal_groups[phase])
            return
        else:
            traci.trafficlights.setRedYellowGreenState(self.id,self.signal_groups[phase])
            return
        
    def step(self):
        print 'specify this method in subclass before use'
        pass


class SimpleTrafficLight(TrafficLight):
    def __init__(self, tlid, simulator, max_phase_time= 30., min_phase_time = 5, yellow_time = 3):
        
        TrafficLight.__init__(self, tlid, simulator)
        self.signal_groups = ['rGrG','ryry','GrGr','yryr']
        self.current_phase = 0  # phase can be 0, 1, 2, 3
        self.current_phase_time = 0
        self.max_time = max_phase_time
        self.min_phase_time = min_phase_time
        self.yellow_time = yellow_time

        # Traffic State 1
        # (car num, .. , dist to TL, .., current phase time)
        self.num_traffic_state = 9
        self.traffic_state = [None for i in range(0, self.num_traffic_state)]

        # Traffic State 2
        # Lanes with car speed in its position
        #self.MAP_SPEED = False
        #self.lane_length = 252
        #self.lanes = 4
        #self.car_length = 4
        #if self.MAP_SPEED:
        #    self.traffic_state = np.zeros((self.lanes, self.lane_length))

        self.reward = None
    
    def updateRLParameters(self):
        lane_list = ['e_0_0','w_0_0','n_0_0','s_0_0']  # temporary, in the future, get this from the .net.xml file
        sim = self.simulator
        self.reward = 0
        
        car_normalizing_number = 5.

        # Traffic State 1
        for i in range(0, 4):
            self.traffic_state[i] = sim.lane_list[lane_list[i]].detected_car_number/car_normalizing_number
            temp = 252
            for vid in sim.lane_list[lane_list[i]].vehicle_list:
                v = sim.veh_list[vid]
                if v.lane_position < temp and v.equipped:
                    temp = sim.veh_list[vid].lane_position
            self.traffic_state[i+4] = temp/float(sim.lane_list[lane_list[i]].length)
            self.reward += sim.lane_list[lane_list[i]].lane_reward
        self.traffic_state[8] = self.current_phase_time/float(self.max_time)
        if self.current_phase in [0,1]:
            self.traffic_state[0]*=-1
            self.traffic_state[1]*=-1
            self.traffic_state[4]*=-1
            self.traffic_state[5]*=-1
        else:
            self.traffic_state[2]*=-1
            self.traffic_state[3]*=-1
            self.traffic_state[6]*=-1
            self.traffic_state[7]*=-1
            self.traffic_state[8]*=-1
        #self.traffic_state[9] = self.current_phase

        # Traffic State 2 I will update this part in another inherited class, I don't want to put this in the same class since it becomes messy
        #if self.MAP_SPEED:
        #    self.traffic_state = np.zeros((self.lanes, self.lane_length))
        #    for i in range(self.lanes):
        #        for vid in sim.lane_list[lane_list[i]].vehicle_list:
        #            v = sim.veh_list[vid]
        #            if v.lane_position < self.lane_length and v.equipped:
        #                self.traffic_state[i, v.lane_position] = v.speed / Vehicle.max_speed
        #        self.reward += sim.lane_list[lane_list[i]].lane_reward

    def step(self, action):
        self.current_phase_time += 1
        # make sure this phrase remain to keep track on current phase time
        
         # rGrG or GrGr
        if self.check_allow_change_phase():
            if action == 1:
            
            
                self.move_to_next_phase()
            #elif self.correct_action(action):
            #    self.move_to_next_phase()
        elif self.current_phase in [1,3]: 
            # yellow phase, action doesn't affect
            if self.current_phase_time > self.yellow_time:
                self.move_to_next_phase()
            # if no appropriate action is given, phase doesn't change
            # if self.current_phase_time > self.yellow_time and self.correct_action(action):
            #     self.move_to_next_phase()
       
        self.updateRLParameters()
        # make sure this method is called last to avoid error

    #def correct_action(self, action):
    #    return action == (self.current_phase + 1) % len(self.actions)
    def check_allow_change_phase(self):
        if self.current_phase in [0, 2]: 
            if self.current_phase_time>self.min_phase_time:
                #print self.current_phase_time, self.min_phase_time
                return True
        return False

    def move_to_next_phase(self):
        self.current_phase = (self.current_phase + 1) % len(self.signal_groups)
        self._set_phase(self.current_phase)
        self.current_phase_time = 0


class ActionSpaces:
    def __init__(self, num_TL, num_actions):
        self.num_TL = num_TL
        self.n = num_actions

    def sample(self):
        return np.random.randint(self.n, size=self.num_TL)

  
if __name__ == '__main__':
    num_episode = 90
    episode_time = 3000
    
    sim = Simulator(episode_time=episode_time)
    #sim = Simulator(visual = True, episode_time=episode_time)
    # use this commend if you don't have pysumo installed
    sim.start()
    for _ in range(num_episode):
        # for i in range(episode_time):
        while True:
            action = sim.action_space.sample()
            next_state, reward, terminal, info = sim.step(action)
            #sim.print_status()
            if terminal:
                sim.reset()
                sim.print_status()
            #    break
    sim.stop()
