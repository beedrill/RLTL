# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET

def create_route(col,row,name, end = 3600, ns_p = 0.03, sn_p = 0.03, ew_p = 0.1, we_p = 0.05):
    ns_routes = []
    sn_routes = []
    ew_routes = []
    we_routes = []
    routes = ET.Element('routes')
    vtype = ET.SubElement(routes,'vType', decel="4.5",accel="2", id="Car", maxSpeed="100.0", sigma="0.5", length="5.0")
    for c in range(2,col):
        route_string = ''
        for r in range(1,row):
            route_string += 'e_{}_{}_{}_{} '.format(c,r,c,r+1)

        route_string = route_string.strip()
        vars()['routesn{}'.format(c)] = ET.SubElement(routes,'route', id="route_sn_{}".format(c), edges = route_string)
        sn_routes.append("route_sn_{}".format(c))
        
        route_string = ''
        for r in range(row,1,-1):
            route_string += 'e_{}_{}_{}_{} '.format(c,r,c,r-1)
        route_string = route_string.strip()
        vars()['routens{}'.format(c)] = ET.SubElement(routes,'route', id="route_ns_{}".format(c), edges = route_string)
        ns_routes.append("route_ns_{}".format(c))
        
    for r in range(2,row):
        route_string = ''
        for c in range(1,col):
            route_string += 'e_{}_{}_{}_{} '.format(c,r,c+1,r)
        route_string = route_string.strip()
        vars()['routewe{}'.format(r)] = ET.SubElement(routes,'route', id="route_we_{}".format(r), edges = route_string)
        we_routes.append("route_we_{}".format(r))
        
        route_string = ''
        for c in range(col,1,-1):
            route_string += 'e_{}_{}_{}_{} '.format(c,r,c-1,r)
        route_string = route_string.strip()
        vars()['routeew{}'.format(r)] = ET.SubElement(routes,'route', id="route_ew_{}".format(r), edges = route_string)
        ew_routes.append("route_ew_{}".format(r))
    
    for rid in ns_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(ns_p))
        
    for rid in sn_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(sn_p))
        
    for rid in ew_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(ew_p))
        
    for rid in we_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(we_p))
    
    tree = ET.ElementTree(routes)
    tree.write("{}.rou.xml".format(name))
    
if __name__=="__main__":
    #this will create for 5x1 mahattan grid
    col = 3
    row = 3
    name = 'traffic'
    ns_flow = [0.02,0.021,0.025,0.031,0.067,0.15,0.22,0.41,0.49,0.42,0.33,0.25,0.24,0.2,0.18,0.16,0.19,0.22,0.23,0.21,0.19,0.16,0.1,0.05]
    ew_flow = [0.019,0.02,0.03,0.029,0.061,0.12,0.15,0.19,0.21,0.18,0.18,0.16,0.15, 0.16,0.16,0.15,0.18,0.21,0.22,0.19,0.16,0.13,0.08,0.03]
    sn_flow = [0.021,0.023,0.033,0.04,0.066,0.13,0.14,0.25,0.24,0.19,0.17,0.17,0.16,0.18,0.22,0.21,0.31,0.43,0.55,0.42,0.32,0.19,0.11,0.04]
    we_flow = [0.02,0.021,0.03,0.038,0.063,0.09,0.13,0.22,0.23,0.18,0.17,0.17,0.15, 0.17, 0.15,0.14,0.17,0.19,0.21,0.2,0.18,0.15,0.09,0.03]
    #print len(ns_flow), len(ew_flow), len(sn_flow), len(we_flow)
    for i in range(0,24):
        create_route(col,row,name+'-{}'.format(i), ns_p = ns_flow[i], sn_p = sn_flow[i], ew_p = ew_flow[i], we_p = we_flow[i])

