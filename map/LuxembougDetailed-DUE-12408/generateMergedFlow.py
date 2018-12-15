import xml.etree.ElementTree as ET

filename = 'traffic-2.rou.xml'
in_edges = ['west_in','north_in','east_in','south_in']
out_edges = ['west_out','north_out','east_out','south_out']

tree = ET.parse(filename)
root = tree.getroot()
carflow = {}
for idx, in_edge in enumerate(in_edges):
    for idx2, out_edge in enumerate(out_edges):
        if idx!=idx2:
            edges = in_edge+' '+out_edge
            flow = 0
            print('------------------edges are: {}'.format(edges))
            for c in root.findall('flow'):
                r = c.attrib['route']
                temp_edges = None
                for c1 in root.findall('route'):
                    if r==c1.attrib['id']:
                        temp_edges = c1.attrib['edges']
                        break
                if temp_edges == None:
                    print('something wrong')

                if edges in temp_edges:
                    print('detect flow: {}'.format(temp_edges))
                    flow+=float(c.attrib['probability'])
            print('flow of {} is {}'.format(edges, flow))
            carflow[edges] = flow

    routes = ET.Element('routes')
    ET.SubElement(routes, 'vType',
        vClass="passenger",
        id="passenger1",
        color=".8,.2,.2",
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5.0",
        minGap="1.5",
        maxSpeed="70",
        guiShape="passenger/sedan")

    for edges in carflow.keys():
        p = carflow[edges]
        route_name = edges.replace(' ', '_')
        ET.SubElement(routes, 'route', id=route_name, edges = edges)

    for edges in carflow.keys():
        p = carflow[edges]
        route_name = edges.replace(' ', '_')
        ET.SubElement(routes, 'flow',
                    id="flow_{}".format(route_name),
                    type='passenger1',
                    route=route_name,
                    probability=str(p),
                    depart="1",
                    departLane = 'best',
                    begin = "0",
                    end = "3600")
tree = ET.ElementTree(routes)
tree.write('a.rou.xml')
