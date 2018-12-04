try:
    import libsumo
except ImportError:
    print('libsumo not installed properly, please use traci only')
# comment this line if you dont have pysumo and set visual = True, it should still run traci
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
import numpy as np
import random

TRUNCATE_DISTANCE = 125.

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class SimpleFlowManager():
    def __init__(self, simulator,
                 rush_hour_file = 'map/traffic-dense.rou.xml',
                 normal_hour_file = 'map/traffic-medium.rou.xml',
                 midnight_file = 'map/traffic-sparse.rou.xml'):

        self.rush_hour_file = rush_hour_file
        self.normal_hour_file = normal_hour_file
        self.midnight_file = midnight_file
        self.sim = simulator

    def travel_to_random_time(self):
        t = random.randint(0,23)
        self.travel_to_time(t)


    def travel_to_time(self,t):
        route_file = self.get_carflow(t)
        self.sim.route_file = route_file
        self.sim.current_day_time = t
        self.sim.cmd[4] = route_file
        print('successfully travel to time: ', t)

    def get_carflow(self, t):
        if t >= 0 and t <=7:
            return self.midnight_file

        if t > 7 and t <= 9:
            return self.rush_hour_file

        if t > 9 and t <= 16:
            return self.normal_hour_file

        if t > 16 and t <= 19:
            return self.rush_hour_file

        if t > 19 and t <= 22:
            return self.normal_hour_file

        if t >22 and t <= 24:
            return self.midnight_file

        print('time:', t, 'is not a supported input, put something between 0 to 24')

class HourlyFlowManager(SimpleFlowManager):
    def __init__(self, simulator, file_name = 'map/whole-day-flow/traffic'):
        self.sim = simulator
        self.file_name =file_name

    def get_carflow(self,t):
        return self.file_name+'-{}.rou.xml'.format(int(t))


class Vehicle():
    max_speed = 13.9

    def __init__(self, vid, simulator, equipped = True):
        self.id = vid
        self.simulator = simulator
        self.depart_time = simulator.time
        self.latest_time = simulator.time
        self.waiting_time = 0
        self.equipped = equipped

    def _update_speed(self):
        if self.simulator.visual == False:
            self.speed = libsumo.vehicle_getSpeed(self.id)
            #print 'vehicle_getSpeed:', type(self.speed), self.speed
            return
        else:
            self.speed = traci.vehicle.getSpeed(self.id)
            return

    def _update_lane_position(self):
        if self.simulator.visual == False:
            self.lane_position = self.lane.length - libsumo.vehicle_getLanePosition(self.id)
            #print 'vehicle_getLanePosition:', type(self.lane_position), self.lane_position
            return
        else:
            self.lane_position = self.lane.length - traci.vehicle.getLanePosition(self.id)
            return

    def _update_appearance(self):
        if self.simulator.visual:
            if self.equipped:
                #traci.vehicle.setColor(self.id,(255,0,0,0))
                return

    def step(self):
        self._update_appearance()
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
            libsumo.trafficlight_setRedYellowGreenState(self.id, self.signal_groups[phase])
            return
        else:
            traci.trafficlights.setRedYellowGreenState(self.id,self.signal_groups[phase])
            return

    def step(self):
        print('specify this method in subclass before use')
        pass



class SimpleTrafficLight(TrafficLight):
    def __init__(self, tlid, simulator, max_phase_time= 40., min_phase_time = 5, yellow_time = 3, num_traffic_state = 11, lane_list = [], state_representation = ''):

        TrafficLight.__init__(self, tlid, simulator)
        self.signal_groups = ['rrrrGGGGrrrrGGGG','rrrryyyyrrrryyyy','GGGGrrrrGGGGrrrr','yyyyrrrryyyyrrrr']
        self.normal_phases = [0,2]
        self.yellow_phases = [1,3]

        self.current_phase = 0  # phase can be 0, 1, 2, 3
        self.current_phase_time = 0
        self.max_time = max_phase_time
        self.min_phase_time = min_phase_time
        self.yellow_time = yellow_time

        # Traffic State 1
        # (car num, .. , dist to TL, .., current phase time)
        self.num_traffic_state = num_traffic_state
        self.traffic_state = [None for i in range(0, self.num_traffic_state)]
        self.lane_list = lane_list
        self.state_representation = state_representation
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
        if self.state_representation == 'original':
            self.updateRLParameters_original()
            return
        elif self.state_representation == 'sign':
            self.updateRLParameters_sign()
        else:
            print('no such state representation supported')
            return

    def updateRLParameters_sign(self):
        lane_list = self.lane_list  # temporary, in the future, get this from the .net.xml file
        sim = self.simulator
        self.reward = 0

        car_normalizing_number = 20. #1. # TODO generalize by length / car length

        # Traffic State 1
        for i in range(0, 4):
            self.traffic_state[i] = sim.lane_list[lane_list[i]].detected_car_number/car_normalizing_number
            temp = sim.lane_list[lane_list[i]].length
            for vid in sim.lane_list[lane_list[i]].vehicle_list:
                v = sim.veh_list[vid]
                if v.equipped == False:
                    continue
                if v.lane_position < temp and v.equipped:
                    temp = sim.veh_list[vid].lane_position
            #self.traffic_state[i+4] = temp/float(sim.lane_list[lane_list[i]].length)
            self.traffic_state[i+4] = 1 - temp / TRUNCATE_DISTANCE # TODO generalize
            #self.traffic_state[i+4] = temp
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

        self.traffic_state[9] = 1 if self.current_phase in [1,3] else -1

        if self.simulator.whole_day:
            self.traffic_state[10] = self.simulator.current_day_time/float(24)


        # Traffic State 2 I will update this part in another inherited class, I don't want to put this in the same class since it becomes messy
        #if self.MAP_SPEED:
        #    self.traffic_state = np.zeros((self.lanes, self.lane_length))
        #    for i in range(self.lanes):
        #        for vid in sim.lane_list[lane_list[i]].vehicle_list:
        #            v = sim.veh_list[vid]
        #            if v.lane_position < self.lane_length and v.equipped:
        #                self.traffic_state[i, v.lane_position] = v.speed / Vehicle.max_speed
        #        self.reward += sim.lane_list[lane_list[i]].lane_reward

    def updateRLParameters_original(self):
        lane_list = self.lane_list  # temporary, in the future, get this from the .net.xml file
        sim = self.simulator
        self.reward = 0

        car_normalizing_number = 20. #1. # TODO generalize by length / car length

        # Traffic State 1
        for i in range(0, 4):
            self.traffic_state[i] = sim.lane_list[lane_list[i]].detected_car_number/car_normalizing_number
            temp = sim.lane_list[lane_list[i]].length
            for vid in sim.lane_list[lane_list[i]].vehicle_list:
                v = sim.veh_list[vid]
                if v.equipped == False:
                    continue
                if v.lane_position < temp and v.equipped:
                    temp = sim.veh_list[vid].lane_position
            #self.traffic_state[i+4] = temp/float(sim.lane_list[lane_list[i]].length)
            self.traffic_state[i+4] = 1 - temp / TRUNCATE_DISTANCE # TODO generalize
            #self.traffic_state[i+4] = temp
            self.reward += sim.lane_list[lane_list[i]].lane_reward
        self.traffic_state[8] = self.current_phase_time/float(self.max_time)

        self.traffic_state[9] = self.current_phase

        if self.simulator.whole_day:
            self.traffic_state[10] = self.simulator.current_day_time/float(24)
    def step(self, action):
        self.current_phase_time += 1
        # make sure this phrase remain to keep track on current phase time

         # rGrG or GrGr
        if self.check_allow_change_phase():
            if action == 1 or self.current_phase_time > self.max_time:
           #if action == 1:
                self.move_to_next_phase()
                #elif self.correct_action(action):
            #    self.move_to_next_phase()
        elif self.current_phase in self.yellow_phases:
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
        if self.current_phase in self.normal_phases:
            if self.current_phase_time>self.min_phase_time:
                #print self.current_phase_time, self.min_phase_time
                return True
        return False

    def move_to_next_phase(self):
        self.current_phase = (self.current_phase + 1) % len(self.signal_groups)
        self._set_phase(self.current_phase)
        self.current_phase_time = 0

class TrafficLightLuxembourg(SimpleTrafficLight):
    def __init__(self, tlid, simulator ,
        max_phase_time= 40.,
        min_phase_time = 5,
        yellow_time = 3,
        num_traffic_state = 11,
        lane_list = [],
        state_representation = '',
        signal_groups = ['rrrGGGGgrrrGGGGg', 'rrryyyygrrryyyyg', 'rrrrrrrGrrrrrrrG', 'rrrrrrryrrrrrrry', 'GGgGrrrrGGgGrrrr', 'yygyrrrryygyrrrr', 'rrGrrrrrrrGrrrrr', 'rryrrrrrrryrrrrr']):
        SimpleTrafficLight.__init__(self,tlid, simulator, max_phase_time= max_phase_time, min_phase_time = min_phase_time,
            yellow_time = yellow_time, num_traffic_state = num_traffic_state, lane_list = lane_list, state_representation = state_representation)
        self.signal_groups = signal_groups
        self.yellow_phases = []
        self.normal_phases = []
        for idx, phase in enumerate(self.signal_groups):
            if 'y' in phase.lower():
                self.yellow_phases.append(idx)
            else:
                self.normal_phases.append(idx)


    def updateRLParameters_original(self):
        lane_list = self.lane_list  # temporary, in the future, get this from the .net.xml file
        sim = self.simulator
        self.reward = 0

        #car_normalizing_number = 20. #1. # TODO generalize by length / car length
        n_lane = len(lane_list)

        # Traffic State 1
        for i in range(0, n_lane):
            lane = sim.lane_list[lane_list[i]]
            self.traffic_state[i] = lane.detected_car_number/lane.car_normalizing_number
            #temp = sim.lane_list[lane_list[i]].length
            temp = min(TRUNCATE_DISTANCE, lane.length)

            for vid in lane.vehicle_list:
                v = sim.veh_list[vid]
                if v.equipped == False:
                    continue
                if v.lane_position < temp and v.equipped:
                    temp = sim.veh_list[vid].lane_position
            #self.traffic_state[i+4] = temp/float(sim.lane_list[lane_list[i]].length)
            self.traffic_state[i+n_lane] = 1 - temp / min(TRUNCATE_DISTANCE, lane.length) # TODO generalize
            #self.traffic_state[i+4] = temp
            self.reward += sim.lane_list[lane_list[i]].lane_reward

        self.traffic_state[2*n_lane] = self.current_phase_time/float(self.max_time)

        self.traffic_state[2*n_lane+1] = self.current_phase

        if self.simulator.whole_day:
            #self.traffic_state[2*n_lane+2] = self.simulator.current_day_time/float(24)
            time = self.simulator.time
            if time<1824:
                day_time = 0;
            elif time>88223:
                day_time = 23;
            else:
                day_time = int((time-1824)/3600)

            if not self.simulator.current_day_time == day_time:
                self.simulator.current_day_time = day_time
                print('time comes to {} o clock'.format(day_time))
            self.traffic_state[2*n_lane+2] = day_time/float(24)

            #print(self.traffic_state)
        #print(self.traffic_state)



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
    def __init__(self, visual = False,
                 map_file = 'map/traffic.net.xml',
                 route_file = 'map/traffic.rou.xml',
                 end_time = 3600, episode_time = 1000,
                 additional_file = None,
                 gui_setting_file = "map/view.settings.xml",
                 penetration_rate = 1,
                 num_traffic_state = 11,
                 record_file = "record.txt",
                 whole_day = False,
                 state_representation = 'sign',
                 flow_manager_file_prefix = 'map/whole-day-flow/traffic',
                 traffic_light_module = SimpleTrafficLight,
                 tl_list = ['26640'],
                 config_file = ''):
        self.visual = visual
        self.map_file = map_file
        self.config_file = config_file
        self.end_time = end_time
        self.route_file = route_file
        self.additional_file = additional_file
        self.gui_setting_file = gui_setting_file
        self.veh_list = {}
        self.timely_veh_list = [[] for i in range(0,24)]
        self.tl_list = {}
        self.is_started = False
        self.time = 0
        self.reset_to_same_time = False
        self.state_representation = state_representation
        self.traffic_light_module = traffic_light_module

        self.penetration_rate = penetration_rate
        #lane_list = ['0_e_0', '0_n_0','0_s_0','0_w_0','e_0_0','n_0_0','s_0_0','w_0_0'] # temporary, in the future, get this from the .net.xml file
        #self.lane_list = {l:Lane(l,self,penetration_rate=penetration_rate) for l in lane_list}
        #tl_list = ['0'] # temporary, in the future, get this from .net.xml file
        #self.tl_id_list = tl_list
        self.num_traffic_state = num_traffic_state
        self._init_sumo_info(tl_list = tl_list)
        #for tlid in tl_list:
        #    self.tl_list[tlid] = SimpleTrafficLight(tlid, self, num_traffic_state = self.num_traffic_state)
        ###RL parameters

        ##############
        self.episode_time = episode_time
        self.action_space = ActionSpaces(len(self.tl_list), 2) # action = 1 means move to next phase, otherwise means stay in current phase
        self.whole_day = whole_day
        self.current_day_time = 0 # this is a value from 0 to 24




        if self.visual == False:
            self.cmd = ['sumo',
                  '--net-file', self.map_file,
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
            if not additional_file == None:
                self.cmd+=['--additional-files', self.additional_file]
            if config_file:
                self.cmd = ['sumo', '-c', config_file]

        else:
            self.cmd = ['sumo-gui',
                  '--net-file', self.map_file,
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
            if config_file:
                self.cmd = ['sumo-gui', '-c', config_file]

        if whole_day:
            self.flow_manager = HourlyFlowManager(self, file_name=flow_manager_file_prefix)
            #self.flow_manager.travel_to_random_time() #this will travel to a random current_day_time and modifie the carflow accordingly
        if not additional_file == None:
            self.cmd+=['--additional-files', self.additional_file]
        if not gui_setting_file == None:
            self.cmd+=['--gui-settings-file', self.gui_setting_file]
        self.record_file = record_file

    def step_(self):
        #use step() for standard operation, this is only for normal traffic light
        self._simulation_step()
        for l in self.lane_list:
            self.lane_list[l].step()

        return traci.simulation.getMinExpectedNumber() == 0

    def _init_sumo_info(self, tl_list = []):
        cmd = ['sumo',
                  '--net-file', self.map_file,
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
        sumoProcess = subprocess.Popen("%s %s" % ('sumo', self.config_file), shell=True, stdout=sys.stdout)
        traci.init(port=8813, numRetries=10, host='localhost', label='default')
        #time.sleep(1)
        #tl_list = traci.trafficlights.getIDList()
        #print 'tls:',tl_list
        self.tl_id_list = tl_list
        lane_list = []
        for tlid in tl_list:
            tl_lane_list = remove_duplicates(traci.trafficlights.getControlledLanes(tlid))
            self.tl_list[tlid] = self.traffic_light_module(tlid, self, num_traffic_state = self.num_traffic_state, lane_list = tl_lane_list,state_representation = self.state_representation)
            lane_list = lane_list+tl_lane_list
            #print 'controlled lane', self.tl_list[tlid].lane_list
        #lane_list = traci.lane.getIDList()
        self.lane_list = {}
        for l in lane_list:
            #print 'lane list', lane_list
            if l.startswith(':'):
                continue
            self.lane_list[l] = Lane(l,self,penetration_rate=self.penetration_rate, length = traci.lane.getLength(l))
            print("lane{}, length{}".format(l,self.lane_list[l].length))
            #print 'lane list', l

        #print len(self.lane_list.keys())

        traci.close()
    def _simulation_start(self):
        if self.visual == False:
            libsumo.start(self.cmd)
            return
        else:
            #traci.start(self.cmd)
            print('starting ... ')

            sumoProcess = subprocess.Popen("{} {} {}".format(self.cmd[0], self.cmd[1], self.cmd[2]), shell=True, stdout=sys.stdout)
            traci.init(port=8813, numRetries=10, host='localhost', label='default')
            return


    def _simulation_end(self):
        if self.visual == False:
            libsumo.close()
            return
        else:
            traci.close()
            return

    def _simulation_step(self):

        if self.visual == False:
            libsumo.simulationStep()
            self.time += 1
            return
        else:
            self.time += 1
            traci.simulationStep()
            #print("time:{}".format(self.time))
            return

    def decode_action(self, encoded_action):
        actions = []
        for _ in range(len(self.tl_id_list)):
            if encoded_action & 1 == 1:
                actions.append(1)
            else:
                actions.append(0)
            encoded_action >>= 1
        return actions

    def step(self, actions):
        self._simulation_step()
        # print('lane list',self.lane_list)
        # print('lane list length', len(self.lane_list))
        # input()
        for l in self.lane_list:
            self.lane_list[l].step()

        observation = []
        reward = []
        i = 0
        for tlid in self.tl_id_list:
            tl = self.tl_list[tlid]
            #print actions
            tl.step(actions[i])
            observation.append(tl.traffic_state)
            reward.append(self.tl_list[tlid].reward)
            i += 1

        #print reward
        observation = np.array(observation)
        reward = np.array(reward)
        info = (self.time, len(self.veh_list.keys()))
        #if not type( observation[0][0]) in ['int',np.float64]:
        #    print 'something wrong', observation[0][0], type(observation[0][0])
        #print reward

        return observation, reward, traci.simulation.getMinExpectedNumber() == 0, info

    def start(self):
        self._simulation_start()
        self.is_started = True

    def stop(self):
        if self.is_started == False:
            print('not started yet')
            return
        self._simulation_end()
        self.is_started = False

    def reset(self):
        return self._reset()

    def _reset(self):
        if self.is_started == True:
            self.stop()
        #if self.whole_day and self.reset_to_same_time == False:
        #    self.flow_manager.travel_to_random_time()
        self.veh_list = {}
        self.time = 0
       #S if self.visual == False:
            #reload(libsumo)
        self.start()
        #print state
        #print len(state)

        observation = []
        reward = []
        info = (self.time, len(self.veh_list.keys()))

        for l in self.lane_list:
            self.lane_list[l].reset()
            self.lane_list[l].update_lane_reward()

        #print 'haha', self.tl_id_list
        for tlid in self.tl_id_list:
            tl = self.tl_list[tlid]
            #print actions
            #tl.step(actions[i])
            tl.move_to_next_phase()
            tl.updateRLParameters()
            observation.append(tl.traffic_state)
            reward.append(self.tl_list[tlid].reward)
            #i += 1
        #print 'haha', observation
        return np.array(observation)

    # def get_result(self):
    #     average_waiting_time_list = []
    #     equipped_average_waiting_time_list = []
    #     nonequipped_average_waiting_time_list = []
    #
    #     for hour in range(0,24):
    #         average_waiting_time,equipped_average_waiting_time,nonequipped_average_waiting_time = self.average_hourly_waiting_time(hour)
    #         average_waiting_time_list.append(average_waiting_time)
    #         equipped_average_waiting_time_list.append(equipped_average_waiting_time)
    #         nonequipped_average_waiting_time_list.append(nonequipped_average_waiting_time)
    #
    #     average_waiting_time, equipped_average_waiting_time, nonequipped_average_waiting_time = self.average_waiting_time()
    #     #print n_equipped, equipped_waiting
    #     return average_waiting_time_list,equipped_average_waiting_time_list,nonequipped_average_waiting_time_list,average_waiting_time, equipped_average_waiting_time, nonequipped_average_waiting_time
    def get_result(self):
        return self.veh_list

    def print_status(self):
        #print self.veh_list
        tl = self.tl_list[self.tl_id_list[0]]
        print('current time:', self.time, ' total cars:', len(self.veh_list.keys()), 'traffic status', tl.traffic_state, 'reward:', tl.reward)


    def average_waiting_time(self):
        vlist = self.veh_list.keys()
        n_total = 0.
        total_waiting = 0.
        equipped_waiting = 0.
        non_equipped_waiting = 0.
        n_equipped = 0.
        n_non_equipped = 0.
        for vid in vlist:
            v = self.veh_list[vid]

            n_total += 1
            total_waiting += v.waiting_time
            if v.equipped:
                n_equipped +=1
                equipped_waiting += v.waiting_time
            else:
                n_non_equipped += 1
                non_equipped_waiting += v.waiting_time

        average_waiting_time = total_waiting/n_total if n_total>0 else 0
        equipped_average_waiting_time = equipped_waiting/n_equipped if n_equipped>0 else 0
        nonequipped_average_waiting_time = non_equipped_waiting/n_non_equipped if n_non_equipped>0 else 0
        return average_waiting_time, equipped_average_waiting_time, nonequipped_average_waiting_time

    def average_hourly_waiting_time(self,hour):
        vlist = self.timely_veh_list[hour]
        n_total = 0.
        total_waiting = 0.
        equipped_waiting = 0.
        non_equipped_waiting = 0.
        n_equipped = 0.
        n_non_equipped = 0.
        for vid in vlist:
            v = self.veh_list[vid]

            n_total += 1
            total_waiting += v.waiting_time
            if v.equipped:
                n_equipped +=1
                equipped_waiting += v.waiting_time
            else:
                n_non_equipped += 1
                non_equipped_waiting += v.waiting_time

        average_waiting_time = total_waiting/n_total if n_total>0 else 0
        equipped_average_waiting_time = equipped_waiting/n_equipped if n_equipped>0 else 0
        nonequipped_average_waiting_time = non_equipped_waiting/n_non_equipped if n_non_equipped>0 else 0
        return average_waiting_time, equipped_average_waiting_time, nonequipped_average_waiting_time


    def record_result(self):
        f = open(self.record_file, 'w')
        vl = list(self.veh_list.values())
        vl.sort(key=lambda x:x.depart_time)
        for v in vl:
            f.write('{}\t{}\t{}\t{}\n'.format(v.id, v.depart_time, v.waiting_time, v.equipped))
        f.close()
   # def _update_vehicles(self):
   #     if self.visual == False:
   #         self.current_veh_list = pysumo.vehicle_list()
   #         return


class Lane():
    def __init__(self,lane_id,simulator,length = 251,penetration_rate = 1):
        self.id = lane_id
        self.simulator = simulator
        self.vehicle_list = []
        self.length = length
        self.car_number = 0
        self.detected_car_number = 0
        self.lane_reward = 0
        self.penetration_rate = penetration_rate
        self.car_normalizing_number = self.length/6

    def update_lane_reward(self):
        self.lane_reward = 0
        for vid in self.vehicle_list:
            if self.simulator.veh_list[vid].lane_position< TRUNCATE_DISTANCE:
                self.lane_reward+=(Vehicle.max_speed - self.simulator.veh_list[vid].speed)/Vehicle.max_speed
        #self.lane_reward = - self.lane_reward
        self.lane_reward = max(min(self.lane_reward, 20), 0) # reward should be possitive, trunccate with 20

    def _get_vehicles(self):
        if self.simulator.visual == False:
            return list(libsumo.lane_getLastStepVehicleIDs(self.id))

        else:
            return traci.lane.getLastStepVehicleIDs(self.id)

    def step(self):
        vidlist = self._get_vehicles()

        self.vehicle_list = vidlist
        self.car_number = len(vidlist)
        self.detected_car_number = 0
        for vid in vidlist:
            if not vid in self.simulator.veh_list.keys():
                self.simulator.veh_list[vid]= Vehicle(vid,self.simulator, equipped = random.random()<self.penetration_rate)
                self.simulator.timely_veh_list[self.simulator.current_day_time].append(vid)
            self.simulator.veh_list[vid].lane = self
            self.simulator.veh_list[vid].step()
            if self.simulator.veh_list[vid].equipped == True and self.simulator.veh_list[vid].lane_position< TRUNCATE_DISTANCE:
                self.detected_car_number += 1


        self.update_lane_reward()

    def reset(self):
        self.vehicle_list = []
        self.car_number = 0
        self.detected_car_number = 0
        self.lane_reward = 0


class ActionSpaces:
    def __init__(self, num_TL, num_actions):
        self.num_TL = num_TL
        self.n = num_actions

    def sample(self):
        return np.random.randint(self.n, size=self.num_TL)


if __name__ == '__main__':

    num_episode = 100
    episode_time = 3000

    sim = Simulator(episode_time = episode_time,
                    visual=True,
                    penetration_rate = 1.,
                    map_file = 'map/whole-day-training-flow-LuST-12408/traffic.net.xml',
                    route_file = 'map/whole-day-training-flow-LuST-12408/traffic.rou.xml',
                    whole_day = True,
                    num_traffic_state = 27,
                    state_representation = 'original',
                    flow_manager_file_prefix = 'map/whole-day-training-flow-LuST-12480/traffic',
                    traffic_light_module = TrafficLightLuxembourg)
    #sim = Simulator(visual = True, episode_time=episode_time)
    # # use this commend if you don't have pysumo installed
    sim.start()
    for _ in range(num_episode):
         for i in range(episode_time):
         #while True:
             action = sim.action_space.sample()
             next_state, reward, terminal, info = sim.step(action)
    #         #print reward
             sim.print_status()
    #         #if terminal:
         state = sim.reset()
    #         #    print state
    #         #    array = np.array(state, np.float32)
    #             #sim.print_status()
    #         #    break
    sim.stop()
