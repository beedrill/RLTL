#!/bin/bash
python "/home/rusheng/pysumo/sumo/sumo/tools/randomTrips.py" -n osm.net.xml --seed 42 --fringe-factor 1 -p 2.955860 -r osm.passenger.rou.xml -o osm.passenger.trips.xml -e 3600 --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --trip-attributes 'speedDev="0.1" departLane="best"' --validate
