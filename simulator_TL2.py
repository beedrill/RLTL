# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:22:48 2017

@author: rusheng
"""

import pysumo
# comment this line if you dont have pysumo and set visual = True, it should still run traci
# Todo: another class for another kind of traffic state formation
import traci
from time import time
import random
import numpy as np
from simulator import Simulator


class DSRCATLAgent():
    def __init__(self, gap_allowed = 70):
        self.gap_allowed = gap_allowed
    def fit(self, env):
        actions = [0]
        #lane_list = {'N':'25638852#8_0', 'E':'26750073_0','S':'-26657759#0_0','W':'-26657746#4_0'} 
        tl_list = ['0']
        env.start()
        step = 0
        while True:
            step+=1
            if step > 6000:
                break
            o,_,_,_ = env.step(actions)
            for tlid in tl_list:
                #print env.get_result()
                tl = env.tl_list[tlid]
                #nNS  = env.lane_list[lane_list['N']].detected_car_number + env.lane_list[lane_list['S']].detected_car_number
                #nEW =  env.lane_list[lane_list['E']].detected_car_number + env.lane_list[lane_list['W']].detected_car_number
                dNS = min(abs(o[0][4]),abs(o[0][6]))
                dEW = min(abs(o[0][5]),abs(o[0][7]))
                #print dNS, dEW
                actions = [0]
                if tl.current_phase == 2:
                    if dNS >= self.gap_allowed and dEW <= self.gap_allowed:
                        actions = [1]
                elif tl.current_phase == 0:
                    if dNS <= self.gap_allowed and dEW >= self.gap_allowed:
                        actions = [1]
                        
                #if env.time>500:
                #    env.record_result()
                #    env.stop()
                #    return
                #print dNS, dEW, actions, tl.current_phase 
            #env.print_status()
            #print env.get_result()
        env.record_result()
                        
        env.stop()
                    
class NormalAgent():
    
    def fit(self,env):
        step = 0
        
        env.start()
        while True:
            step+=1
            if step>6000:
                break
            env.step_()
            
        env.record_result()
        env.stop()
  
if __name__ == '__main__':
    
    #for pr in [0.01,0.2,0.4,0.6,0.8,0.99]:
    
        #sim = Simulator(penetration_rate = pr)
        #sim.cmd[0] = 'sumo'
        
    #sim.start()
        #agent = DSRCATLAgent()
        #agent.fit(sim)
        
    sim = Simulator(visual=False,penetration_rate = 0,map_file='map/5-intersections/traffic.net.xml',\
                       route_file='map/5-intersections/traffic.rou.xml')

    agent = NormalAgent()
    agent.fit(sim)
###########################for traffic light in a whole day:        
#    sim = Simulator(visual=False,penetration_rate = 0,map_file='map/whole-day-flow/traffic.net.xml',\
#                        route_file='map/whole-day-flow/traffic-0.rou.xml', whole_day = True)
#    
#    sim.reset_to_same_time = True
#    agent = NormalAgent()
#    for t in range(0,24):
#        sim = Simulator(visual=False,penetration_rate = 0,map_file='map/whole-day-flow/traffic.net.xml',\
#                        route_file='map/whole-day-flow/traffic-0.rou.xml', whole_day = True)
#    
#        sim.reset_to_same_time = True
#        sim.flow_manager.travel_to_time(t)
#        
#        agent.fit(sim)
        ##################################################