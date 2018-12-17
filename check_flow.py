import xml.etree.ElementTree as ET

#filename = 'map/OneIntersectionLuST-12408-stationary/8/traffic.rou.xml'
#filename = 'map/LuxembougDetailed-DUE-12408/traffic-8.rou.xml'
filename = 'map/LuxembougDetailed-DUE-12408/traffic-8.rou.xml'
edge_interested = 'south_in'

tree = ET.parse(filename)
root = tree.getroot()
flow = 0
for c in root.findall('flow'):
    r = c.attrib['route']
    edges = None
    for c1 in root.findall('route'):
        if r==c1.attrib['id']:
            edges = c1.attrib['edges']
            break
    if edges == None:
        print('something wrong')
    if edge_interested in edges.split():
        flow+=float(c.attrib['probability'])
print('flow is {}'.format(flow))
