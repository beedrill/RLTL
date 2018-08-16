try:
    import libsumo
except ImportError:
    print 'libsumo not installed properly, please use traci only'
# comment this line if you dont have pysumo and set visual = True, it should still run traci

import traci
from time import time
import numpy as np
import random

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
                 record_file = "record.txt",
                 simple=False,
                 aggregated_reward = False):

        self.aggregated_reward = aggregated_reward
        self.simple_inputs = simple
        self.visual = visual
        self.map_file = map_file
        self.end_time = end_time
        self.route_file = route_file
        self.additional_file = additional_file
        self.gui_setting_file = gui_setting_file
        self.veh_list = {}
        self.tl_list = {}
        self.is_started = False
        self.time = 0
        self.reset_to_same_time = False

        self.penetration_rate = penetration_rate

        self._init_sumo_info()

        ##############
        self.episode_time = episode_time
        self.action_space = ActionSpaces(len(self.tl_list), 2) # action = 1 means move to next phase, otherwise means stay in current phase
        self.current_day_time = 0 # this is a value from 0 to 24



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
        self.record_file = record_file

    def step_(self):
        #use step() for standard operation, this is only for normal traffic light
        self._simulation_step()
        for l in self.lane_list:
            self.lane_list[l].step()

    def _init_sumo_info(self):
        cmd = ['sumo',
                  '--net-file', self.map_file,
                  '--route-files', self.route_file,
                  '--end', str(self.end_time)]
        traci.start(cmd)
        #time.sleep(1)
        tl_list = traci.trafficlights.getIDList()

        self.tl_id_list = tl_list
        max_number_of_lanes = 0
        for tlid in tl_list:
    	    lanes_ordered_by_signal_order = set(traci.trafficlights.getControlledLanes(tlid))
            if len(lanes_ordered_by_signal_order) > max_number_of_lanes:
                max_number_of_lanes = len(lanes_ordered_by_signal_order)

        if self.simple_inputs:
            self.fixed_number_of_tl_states = 12
            self.num_traffic_state = self.fixed_number_of_tl_states
        else:
            # 2 states per lane (number of cars + distance of closest car)
            # 1 state for traffic light id
            # 1 for orange light indicator
            # 1 for current phase time
            self.fixed_number_of_tl_states = 4
            self.num_traffic_state = max_number_of_lanes * 2 + self.fixed_number_of_tl_states + 5

        number_of_tl = len(tl_list)
        print("there are " + str(number_of_tl) + " controlled traffic lights")
        tl_counter = 0
        for tlid in tl_list:
    	    lanes_ordered_by_signal_order = list(traci.trafficlights.getControlledLanes(tlid))
    	    definition = traci.trafficlights.getCompleteRedYellowGreenDefinition(tlid)
    	    definition = str(traci.trafficlights.getCompleteRedYellowGreenDefinition(tlid))
            signal_groups = [phase_definition.split('\n')[0] for phase_definition in definition.split('phaseDef: ')[1:]]
            normalized_id = tl_counter / float(number_of_tl)
            tl_counter += 1
            number_of_phases = definition.count('Phase:')
            self.tl_list[tlid] = SimpleTrafficLight(tlid, self, num_traffic_state = self.num_traffic_state, number_of_phases=number_of_phases, lane_list = lanes_ordered_by_signal_order, signal_groups= signal_groups, fixed_number_of_tl_states= self.fixed_number_of_tl_states, normalized_id = normalized_id, simple_inputs=self.simple_inputs)

        lane_list = traci.lane.getIDList()
        self.lane_list = {}
        for l in lane_list:
            if l.startswith(':'):
                continue
            self.lane_list[l] = Lane(l,self,penetration_rate=self.penetration_rate, length = traci.lane.getLength(l))

        traci.close()
    def _simulation_start(self):
        if self.visual == False:
            libsumo.start(self.cmd)
            return
        else:
            traci.start(self.cmd)
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
            if not self.aggregated_reward:
                reward.append(tl.reward)
            i += 1

        if self.aggregated_reward:
            total_reward = sum([self.tl_list[tlid].reward for tlid in self.tl_id_list])
            for tlid in self.tl_id_list:
                reward.append(total_reward)

        #print reward
        observation = np.array(observation)
        reward = np.array(reward)
        info = (self.time, len(self.veh_list.keys()))
        #if not type( observation[0][0]) in ['int',np.float64]:
        #    print 'something wrong', observation[0][0], type(observation[0][0])
        #print reward

        return observation, reward, self.time == self.episode_time, info

    def start(self):
        self._simulation_start()
        self.is_started = True

    def stop(self):
        if self.is_started == False:
            print 'not started yet'
            return
        self._simulation_end()
        self.is_started = False

    def reset(self):
        return self._reset()

    def _reset(self):
        if self.is_started == True:
            self.stop()
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

    def get_result(self):
        total_waiting = 0.
        equipped_waiting = 0.
        non_equipped_waiting = 0.

        n_total = 0.
        n_equipped = 0.
        n_non_equipped = 0.

        for vid in self.veh_list:
            v = self.veh_list[vid]

            n_total += 1
            total_waiting += v.waiting_time
            if v.equipped:
                n_equipped +=1
                equipped_waiting += v.waiting_time
            else:
                n_non_equipped += 1
                non_equipped_waiting += v.waiting_time


        self.average_waiting_time = total_waiting/n_total if n_total>0 else 0
        self.equipped_average_waiting_time = equipped_waiting/n_equipped if n_equipped>0 else 0
        self.nonequipped_average_waiting_time = non_equipped_waiting/n_non_equipped if n_non_equipped>0 else 0
        #print n_equipped, equipped_waiting
        return self.average_waiting_time,self.equipped_average_waiting_time, self.nonequipped_average_waiting_time
    def print_status(self):
        #print self.veh_list
        tl = self.tl_list[self.tl_id_list[0]]
        print 'current time:', self.time, ' total cars:', len(self.veh_list.keys()), 'traffic status', tl.traffic_state, 'reward:', tl.reward




    def record_result(self):
        f = open(self.record_file, 'a')
        f.write('{}\t{}\t{}\n'.format(*(self.get_result())))
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

    def update_lane_reward(self):
        self.lane_reward = 0
        for vid in self.vehicle_list:
            self.lane_reward+=(Vehicle.max_speed - self.simulator.veh_list[vid].speed)/Vehicle.max_speed
        #self.lane_reward = - self.lane_reward

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
            if self.simulator.veh_list[vid].equipped == True:
                self.detected_car_number += 1
            self.simulator.veh_list[vid].lane = self
            self.simulator.veh_list[vid].step()
        self.update_lane_reward()

    def reset(self):
        self.vehicle_list = []
        self.car_number = 0
        self.detected_car_number = 0
        self.lane_reward = 0


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
        print 'specify this method in subclass before use'
        pass


class SimpleTrafficLight(TrafficLight):
    def __init__(self, tlid, simulator, max_phase_time= 40., number_of_phases=4, min_phase_time = 5, yellow_time = 3, num_traffic_state = 11, lane_list = [], signal_groups=[], fixed_number_of_tl_states = 4, normalized_id=0, simple_inputs=False):

        TrafficLight.__init__(self, tlid, simulator)
        self.signal_groups = signal_groups
        self.current_phase = 0
        self.current_phase_time = 0
        self.max_time = max_phase_time
        self.min_phase_time = min_phase_time
        self.yellow_time = yellow_time
        self.number_of_lanes = len(set(lane_list))
        self.number_of_phases = number_of_phases
        # Traffic State 1
        # (car num, .. , dist to TL, .., current phase time)
        self.num_traffic_state = num_traffic_state
        self.traffic_state = [0 for i in range(0, self.num_traffic_state)]
        self.lane_list = lane_list
        self.reward = None
        self.fixed_number_of_tl_states = fixed_number_of_tl_states
        self.normalized_id = normalized_id
        self.simple_inputs = simple_inputs
        if self.simple_inputs:
            self.updateRLParameters = self.updateRLParameters_simple
        else:
            self.updateRLParameters = self.updateRLParameters_normal



    def updateRLParameters_normal(self):
        lane_list = self.lane_list
        sim = self.simulator
        self.reward = 0

        current_phase = str(self.signal_groups[self.current_phase]).lower()
    	# Traffic State 1
    	for i in range(0, len(self.traffic_state)):
    	    self.traffic_state[i] = 0

        self.traffic_state[0] = self.normalized_id
        # 1 if there is an orange light else -1
        self.traffic_state[1] = 1 if 'g' not in current_phase else -1
        self.traffic_state[2] = self.current_phase_time/float(self.max_time)
        self.traffic_state[3] = 1 if self.is_left_turn() else 0
        lane_counter = 0
        for lane, light_color in set(zip(lane_list,current_phase)):
                sign_phase = 1.0
                if light_color == 'r' or light_color == 'y':
                    sign_phase = -1.0

                car_normalizing_number = sim.lane_list[lane].length
                self.traffic_state[self.fixed_number_of_tl_states + lane_counter * 2] = sign_phase * sim.lane_list[lane].detected_car_number/ float(car_normalizing_number)
                temp = car_normalizing_number
                for vid in sim.lane_list[lane].vehicle_list:
                    v = sim.veh_list[vid]
                    if v.equipped == False:
                        continue
                    if v.lane_position < temp and v.equipped:
                        temp = sim.veh_list[vid].lane_position
                self.traffic_state[self.fixed_number_of_tl_states + lane_counter * 2 + 1] = sign_phase * (1 - temp / 125.)

                self.reward += sim.lane_list[lane].lane_reward
                lane_counter += 1


    def updateRLParameters_simple(self):
        lane_list = self.lane_list
        sim = self.simulator
        self.reward = 0
        current_phase = str(self.signal_groups[self.current_phase]).lower()
        distributions = [[], []]
        # Traffic State 1
        for i in range(0, len(self.traffic_state)):
            self.traffic_state[i] = 0
        for lane, light_color in set(zip(lane_list,str(current_phase))):
            i = 0
            if light_color == 'r' or light_color == 'y':
                i = 1
            car_normalizing_number = sim.lane_list[lane].length
            self.traffic_state[i] += sim.lane_list[lane].detected_car_number/car_normalizing_number
            distributions[i].append(sim.lane_list[lane].detected_car_number/car_normalizing_number)
            for vid in sim.lane_list[lane].vehicle_list:
                v = sim.veh_list[vid]
                if v.equipped == False:
                    continue
                if v.lane_position < self.traffic_state[i+2] or self.traffic_state[i+2] == 0:
                    self.traffic_state[i+2] = sim.veh_list[vid].lane_position / car_normalizing_number
            self.reward += sim.lane_list[lane].lane_reward
        self.traffic_state[2] = 1 - self.traffic_state[2]
        self.traffic_state[3] = 1 - self.traffic_state[3]
        self.traffic_state[4] = np.std(distributions[0]) if distributions[0] else 0
        self.traffic_state[5] = np.std(distributions[1]) if distributions[1] else 0
        self.traffic_state[6] = self.number_of_phases / 10.0 # generalize to max number of phases
        self.traffic_state[7] = current_phase.count('g')/float(len(lane_list))
        self.traffic_state[8] = self.current_phase_time/float(self.max_time)
        self.traffic_state[9] = 1 if 'g' not in current_phase else -1
        self.traffic_state[10] = self.normalized_id
        self.traffic_state[11] = 1 if self.is_left_turn() else 0


    def step(self, action):
        self.current_phase_time += 1
        # make sure this phrase remain to keep track on current phase time

        if self.is_left_turn() and action == 1:
            self.move_to_next_phase()
        elif self.is_orange_light():
            # yellow phase, action doesn't affect
            if self.current_phase_time > self.yellow_time:
                self.move_to_next_phase()
        elif self.min_duration_has_passed():
            if action == 1 or self.current_phase_time > self.max_time:
                self.move_to_next_phase()
        self.updateRLParameters()

    def min_duration_has_passed(self):
        return self.current_phase_time > self.min_phase_time

    def is_orange_light(self):
        return 'y' in self.signal_groups[self.current_phase]

    def is_left_turn(self):
        #is_orange_before_left_turn =  'y' in self.signal_groups[self.current_phase] and 'g' in self.signal_groups[self.current_phase]
        is_left_turn_only_green =  'y' in self.signal_groups[self.last_phase()] and 'g' in self.signal_groups[self.last_phase()].lower()
        return is_left_turn_only_green

    def next_phase(self):
        return (self.current_phase + 1) % len(self.signal_groups)

    def last_phase(self):
        return (self.current_phase - 1) % len(self.signal_groups)

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
