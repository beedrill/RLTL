# -*- coding: utf-8 -*-
import libsumo

map_file = 'map/1-intersection/traffic.net.xml'
route_file = 'map/1-intersection/traffic.rou.xml'
cmd = ['sumo', '--net-file', map_file, '--route-files', route_file]

libsumo.start(cmd)
print vars(libsumo.simulation)
libsumo.close()

libsumo.start(cmd)
 
