import matplotlib.pyplot as plt
import time
import random as rd
import networkx as nx
import statistics


def locInit(file):
    global agents, pop, idToZipCode
    agents, idToZipCode = {}, {}
    agents = {}
    agentIdx = 0
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        loc = loc.split(",")
        agents[agentIdx] = (float(loc[2]), float(loc[1]))
        idToZipCode[agentIdx] = int(loc[0])
        agentIdx+=1
    pop = agentIdx

def findScalingFactor(net):
    distList = []
    for node in net:
        coord1 = agents[node]
        for nextNode in net:
            if node != nextNode:
                coord2 = agents[nextNode]
                if coord1[0] == coord2[0] and coord1[1] == coord2[1]:
                    continue
                dist = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
                distList.append(dist)
    distList.sort()
    minD = distList[100] #change the index for a different scaling factor
    return 1/minD

def getNodeOrder():
    global nodeOrderList
    minX = float("inf")
    minXAgent = 0
    for node in agents:
        if agents[node][0] < minX:
            minX = agents[node][0]
            minXAgent = node
    nodeOrderList = [minXAgent]
    remainingNodes = {n for n in agents if n != minXAgent}
    while remainingNodes:
        coord1 = (sum([agents[n][0] for n in nodeOrderList])/len(nodeOrderList), sum([agents[n][1] for n in nodeOrderList])/len(nodeOrderList))
        minD = float("inf")
        minDAgent = 0
        for node in remainingNodes:
            coord2 = agents[node]
            dist = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
            if dist < minD:
                minD = dist
                minDAgent = node
        nodeOrderList.append(minDAgent)
        remainingNodes.remove(minDAgent)
    
def normalizeDistances(net, deg, new, alpha, scalingFactor):
    total = 0
    for node in net:
        if node !=new:
            coord1 = agents[node]
            coord2 = agents[new]
            d = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
            d_alpha = (d*scalingFactor)**(-1*alpha)
            ConnPr = d_alpha * deg[node]
            total += ConnPr
    return total



def Select(net,new,m,alpha,scalingFactor):
    deg = {node:val for (node, val) in net.degree()}
    if new == len(agents):
        return
    new_coord = agents[new]
    nodeJ_coord = list()
    targets = set()
    exponent = alpha * (-1)
    while len(targets) < m:
        nodeJ = rd.choice(list(deg.keys()))
        while nodeJ in targets or nodeJ == new:
            nodeJ = rd.choice(list(deg.keys()))
        nodeJ_coord = agents[nodeJ]
        d = (float(nodeJ_coord[0] - new_coord[0])**2 + (float(nodeJ_coord[1] - new_coord[1])**2))**0.5
        d_alpha = (d*scalingFactor)**(exponent)
        ConnPr = d_alpha * deg[nodeJ]
        chance = rd.random()
        if (ConnPr <= 1 and ConnPr > chance):
            targets.add(nodeJ)
    return targets

def SelectNodeOrder(net,new,m,alpha,scalingFactor):
    deg = {node:val for (node, val) in net.degree()}
    if new == len(agents):
        return
    new_coord = agents[new]
    nodeJ_coord = list()
    targets = set()
    exponent = alpha * (-1)
    total = normalizeDistances(net, deg, new, alpha, scalingFactor)
    while len(targets) < m:
        nodeJ = rd.choice(list(deg.keys()))
        while nodeJ in targets or nodeJ == new:
            nodeJ = rd.choice(list(deg.keys()))
        nodeJ_coord = agents[nodeJ]
        d = (float(nodeJ_coord[0] - new_coord[0])**2 + (float(nodeJ_coord[1] - new_coord[1])**2))**0.5
        d_alpha = (d*scalingFactor)**(exponent)
        ConnPr = (d_alpha * deg[nodeJ])/total
        chance = rd.random()
        if (ConnPr <= 1 and ConnPr > chance):
            targets.add(nodeJ)
    return targets


def SpatialNetSFNodeOrder(m,alpha,scalingFactor):
    targets = nodeOrderList[:m]
    network = nx.empty_graph(m)
    source = m
    network.add_edges_from(zip([source]*m,targets))
    for source in nodeOrderList[m:]:
        targets = SelectNodeOrder(network,source,m,alpha,scalingFactor)
        network.add_edges_from(zip([source]*m,targets))
    nx.set_node_attributes(network, agents, name="pos")
    return network

def SpatialNetSF(m,alpha,scalingFactor):
    network = nx.empty_graph(m)
    targets = list(range(m))
    source = m
    while source < pop-1:
        network.add_edges_from(zip([source]*m,targets))
        source += 1
        targets = Select(network,source,m,alpha,scalingFactor)
    nx.set_node_attributes(network, agents, name="pos")
    return network


start_time = time.time()

a = 3

#locInit("FairfaxCensusTractLatLong.csv")
locInit("VAZipcodesWithLinks.csv")
scalingFactor = findScalingFactor(agents)
getNodeOrder()

numEdges = []
averageDegree = []
stddevDegree = []
radiusGyration = []
stddevRadiusGyration = []
avgLengthEdges = []
stdDevEdgeLength = []
numTriangles = []
avgPathLength = []
maximumDegree = []
totalTime = 0

count = 1

for i in range(count):
    start_time = time.time()
    print("Iteration " + str(i))
    g = SpatialNetSFNodeOrder(10,a,scalingFactor)


    degrees = g.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree.append(sum_of_edges/pop)
    stddevDegree.append(statistics.stdev(dict(degrees).values()))
    numEdges.append(g.number_of_edges())

    avgPathLength.append(nx.average_shortest_path_length(g))

    degList = [i for i in degrees]
    maxDeg = max(degList, key= lambda x: x[1])
    maximumDegree.append(maxDeg[1])

    averageDistances = []
    radiusOfGyration = []

    for node in g:
        maxDistance = 0
        for n in g.neighbors(node):
            coord1, coord2 = agents[node], agents[n]
            dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
            averageDistances.append(dist)
            if dist > maxDistance:
                maxDistance = dist
        radiusOfGyration.append(maxDistance)

    radiusGyration.append(sum(radiusOfGyration)/pop)
    stddevRadiusGyration.append(statistics.stdev(radiusOfGyration))
    avgLengthEdges.append(sum(averageDistances)/len(averageDistances))
    stdDevEdgeLength.append(statistics.stdev(averageDistances))

    numTriangles.append(sum(nx.triangles(g).values()) / 3)


print("Number of Edges:")
print(sum(numEdges)/count)
print("Average Degree:")
print(sum(averageDegree)/count)
print("Standad Deviation of the Degree:")
print(sum(stddevDegree)/count)
print("Radius of Gyration:")
print(sum(radiusGyration)/count)
print("Standard Deviation of Radius of Gyration:")
print(sum(stddevRadiusGyration)/count)
print("Average Distance Between Connected Nodes:")
print(sum(avgLengthEdges)/count)
print("Standard Deviation of Distance Between Connected Nodes:")
print(sum(stdDevEdgeLength)/count)
print("Number of Triangles:")
print(sum(numTriangles)/count)
print("Average Shortest Path Length:")
print(sum(avgPathLength)/count)
print("Maximum Degree:")
print(sum(maximumDegree)/count)


nx.draw(g, nx.get_node_attributes(g, 'pos'), node_size=1, width=0.25)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
