# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET

def generate_continued_edge_string(col_start, row_start, col_end, row_end, direction):
    #direction[0], col direction direction[1]: row direction
    if col_start != col_end and row_start != row_end:
        print 'begin point and end point has to be on the same line'
        return
    if direction[0] == 0:
        beg = row_start
        end = row_end
        d = direction[1]
    elif direction[1] == 0:
        beg = col_start
        end = col_end
        d = direction[0]
    else:
        print "one of the direction must be 0"
        return
        
    route_string = ""    
    c = col_start
    r = row_start
    for i in range(beg,end,d):
        route_string += 'e_{}_{}_{}_{} '.format(c,r,c+direction[0],r+direction[1])
        c = c+direction[0]
        r = r + direction[1]
    return route_string.strip()
            
        

def create_straight_route(col,row,name, end = 3600, ns_p = 0.03, sn_p = 0.03, ew_p = 0.1, we_p = 0.05):
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
    
def create_one_turn_route(col,row,name, end = 3600, ns_p = 0.01, sn_p = 0.01, ew_p = 0.1, we_p = 0.05, 
                          rate_ns_e = 0.01, rate_ns_w = 0.01, rate_sn_e = 0.01, rate_sn_w = 0.01, 
                          rate_ew_n = 0.01, rate_ew_s=0.01, rate_we_n = 0.01, rate_we_s = 0.01):
# rate_**_* specify flow rate of turning
#**_p indicate the rate of straight flow
    ns_straight_routes = []
    sn_straight_routes = []
    ew_straight_routes = []
    we_straight_routes = []

    ns_turned_east_routes = []
    ns_turned_west_routes = []
    sn_turned_east_routes = []
    sn_turned_west_routes = []
    ew_turned_south_routes = []
    ew_turned_north_routes = []
    we_turned_south_routes = []
    we_turned_north_routes = []
    routes = ET.Element('routes')
    vtype = ET.SubElement(routes,'vType', decel="4.5",accel="2", id="Car", maxSpeed="100.0", sigma="0.5", length="5.0")

    for c in range(2,col):
        for r in range(2,row):
            #route_string = generate_continued_edge_string(c,1,c,r)
            turned_string = generate_continued_edge_string(c,1,c,r,[0,1]) +' '+ generate_continued_edge_string(c,r,col,r,[1,0])
            vars()['routesn_e_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_sn_e_{}_{}".format(c,r), edges = turned_string)
            sn_turned_east_routes.append('route_sn_e_{}_{}'.format(c,r))            
            
            turned_string = generate_continued_edge_string(c,1,c,r,[0,1]) +' '+ generate_continued_edge_string(c,r,1,r,[-1,0])
            vars()['routesn_w_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_sn_w_{}_{}".format(c,r), edges = turned_string)
            sn_turned_west_routes.append('route_sn_w_{}_{}'.format(c,r))     
                

        route_string = generate_continued_edge_string(c,1,c,row,[0,1])
        vars()['routesn{}'.format(c)] = ET.SubElement(routes,'route', id="route_sn_{}".format(c), edges = route_string)
        sn_straight_routes.append("route_sn_{}".format(c))
        
        for r in range(2,row):
            #route_string = generate_continued_edge_string(c,1,c,r)
            turned_string = generate_continued_edge_string(c,row,c,r,[0,-1]) +' '+ generate_continued_edge_string(c,r,col,r,[1,0])
            vars()['routens_e_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_ns_e_{}_{}".format(c,r), edges = turned_string)
            ns_turned_east_routes.append('route_ns_e_{}_{}'.format(c,r))            
            
            turned_string = generate_continued_edge_string(c,row,c,r,[0,-1]) +' '+ generate_continued_edge_string(c,r,1,r,[-1,0])
            vars()['routens_w_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_ns_w_{}_{}".format(c,r), edges = turned_string)
            ns_turned_west_routes.append('route_ns_w_{}_{}'.format(c,r))     
                

        route_string = generate_continued_edge_string(c,row,c,1,[0,-1])
        vars()['routens{}'.format(c)] = ET.SubElement(routes,'route', id="route_ns_{}".format(c), edges = route_string)
        sn_straight_routes.append("route_ns_{}".format(c))
        

        
    for r in range(2,row):
        for c in range(2,col):
            #route_string = generate_continued_edge_string(c,1,c,r)
            turned_string = generate_continued_edge_string(1,r,c,r,[1,0]) +' '+ generate_continued_edge_string(c,r,c,row,[0,1])
            vars()['routewe_n_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_we_n_{}_{}".format(c,r), edges = turned_string)
            we_turned_north_routes.append('route_we_n_{}_{}'.format(c,r))            
            
            turned_string = generate_continued_edge_string(1,r,c,r,[1,0]) +' '+ generate_continued_edge_string(c,r,c,1,[0,-1])
            vars()['routewe_s_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_we_s_{}_{}".format(c,r), edges = turned_string)
            we_turned_south_routes.append('route_we_s_{}_{}'.format(c,r))     
                

        route_string = generate_continued_edge_string(1,r,col,r,[1,0])
        vars()['routewe{}'.format(c)] = ET.SubElement(routes,'route', id="route_we_{}".format(r), edges = route_string)
        we_straight_routes.append("route_we_{}".format(r))
        
        for c in range(2,col):
            #route_string = generate_continued_edge_string(c,1,c,r)
            turned_string = generate_continued_edge_string(col,r,c,r,[-1,0]) +' '+ generate_continued_edge_string(c,r,c,row,[0,1])
            vars()['routeew_n_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_ew_n_{}_{}".format(c,r), edges = turned_string)
            ew_turned_north_routes.append('route_ew_n_{}_{}'.format(c,r))            
            
            turned_string = generate_continued_edge_string(col,r,c,r,[-1,0]) +' '+ generate_continued_edge_string(c,r,c,1,[0,-1])
            vars()['routeew_s_{}_{}'.format(c,r)] = ET.SubElement(routes,'route', id="route_ew_s_{}_{}".format(c,r), edges = turned_string)
            ew_turned_south_routes.append('route_ew_s_{}_{}'.format(c,r))     
                

        route_string = generate_continued_edge_string(col,r,1,r,[-1,0])
        vars()['routeew{}'.format(c)] = ET.SubElement(routes,'route', id="route_ew_{}".format(r), edges = route_string)
        ew_straight_routes.append("route_ew_{}".format(r))
    
    for rid in ns_straight_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(ns_p))
        
    for rid in sn_straight_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(sn_p))
        
    for rid in ew_straight_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(ew_p))
        
    for rid in we_straight_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(we_p))
        
    for rid in ns_turned_east_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_ns_e))
        
    for rid in ns_turned_west_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_ns_w))
        
    for rid in sn_turned_east_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_sn_e))
        
    for rid in sn_turned_west_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_sn_w))
        
    for rid in we_turned_north_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_we_n))
    
    for rid in we_turned_south_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_we_s))    
    
    for rid in ew_turned_north_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_ew_n))
        
    for rid in ew_turned_south_routes:
        fname = rid.replace('route','flow')
        vars()[fname] = ET.SubElement(routes,'flow', id = fname, depart = "1",begin = '0', end = str(end), type = "Car", route = rid, probability = str(rate_ew_s))
    tree = ET.ElementTree(routes)
    tree.write("{}.rou.xml".format(name))
    
if __name__=="__main__":
    #this will create for 5x1 mahattan grid
    col = 6
    row = 6
    name = 'traffic'
    #create_one_turn_route(col,row,name)
    create_straight_route(col,row,name)
    
    #print generate_continued_edge_string(12,5,6,5,[-1,0])