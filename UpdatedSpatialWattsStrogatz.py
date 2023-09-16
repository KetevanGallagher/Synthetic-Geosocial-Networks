import matplotlib.pyplot as plt
import random as rd
import networkx as nx
import time
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

def getNearestNodes(k):
    global nearestNeighbors
    nearestNeighbors = {}
    for node in agents:
        nodeNeighbors = []
        for n in agents:
            if n == node:
                continue
            if n in nearestNeighbors and node in nearestNeighbors[n]:
                continue
            coord1, coord2 = agents[node], agents[n]
            dist = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
            if not nodeNeighbors:
                nodeNeighbors.append((dist, n))
            elif nodeNeighbors[-1][0] < dist:
                if len(nodeNeighbors) < k:
                    nodeNeighbors.append((dist, n))
            else:
                i = len(nodeNeighbors)-1
                while nodeNeighbors[i][0] > dist and i > -1:
                    i-=1
                nodeNeighbors.insert(i+1, (dist, n))
                if len(nodeNeighbors) > k:
                    nodeNeighbors.pop()
        nearestNeighbors[node] = {i[1] for i in nodeNeighbors}


def SpatialSW2(p,scalingFactor,a):
    global u,v,w,G,nodes
    G = nx.Graph()
    for node in nearestNeighbors:
        for n in nearestNeighbors[node]:
            G.add_edge(node, n)
    nodes = list(range(pop)) # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for u in nearestNeighbors:
        for v in nearestNeighbors[u]: 
            if rd.random() < p:
                chosen = False
                while chosen == False:
                    w = rd.choice(nodes)
                    while w == u or G.has_edge(u, w): # Enforce no self-loops or multiple edges
                        w = rd.choice(nodes)
                    u_coord = agents[u]
                    w_coord = agents[w]
                    d = (float(u_coord[0] - w_coord[0])**2 + (float(u_coord[1] - w_coord[1])**2))**0.5
                    if d < 1/scalingFactor:
                        d = 1/scalingFactor
                    q = (d*scalingFactor)**(-a)
                    if rd.random() < q:
                        G.remove_edge(u,v)
                        G.add_edge(u,w)
                        chosen = True
    nx.set_node_attributes(G, agents, name="pos")
    return G


def SpSW(scalingFactor, a):
    global u,v,w,G,nodes,percentRewired
    rewire_count = 0
    G = nx.Graph()
    for node in nearestNeighbors:
        for n in nearestNeighbors[node]:
            G.add_edge(node, n)
    totalEdges = G.number_of_edges()
    nodes = list(range(pop)) # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for u in nearestNeighbors:
        for v in nearestNeighbors[u]: 
            u_coord = agents[u]
            v_coord = agents[v]
            d = (float(u_coord[0] - v_coord[0])**2 + (float(u_coord[1] - v_coord[1])**2))**0.5
            if d < 1/scalingFactor:
                d = 1/scalingFactor
            p = (d*scalingFactor)**(-a)
            if rd.random() < p:
                w = rd.choice(nodes)
                while w == u or G.has_edge(u, w): # Enforce no self-loops or multiple edges
                    w = rd.choice(nodes)
                G.remove_edge(u,v)  
                G.add_edge(u,w)
                rewire_count += 1
    nx.set_node_attributes(G, agents, name="pos")
    print("Rewire Count:")
    print(str(rewire_count) + " edges out of " + str(totalEdges))
    percentRewired = (rewire_count/totalEdges)
    print(f"{percentRewired*100}% of edges rewired")
    return G



start_time = time.time()
a = 3
k = 20
p = 1/10

locInit("VAZipcodesWithLinks.csv")
#locInit("FairfaxCensusTractLatLong.csv")
scalingFactor = findScalingFactor(agents)

getNearestNodes(k//2)

numEdges1 = []
averageDegree1 = []
stddevDegree1 = []
radiusGyration1 = []
stddevRadiusGyration1 = []
avgLengthEdges1 = []
stdDevEdgeLength1 = []
numTriangles1 = []
avgPathLength1 = []
maximumDegree1 = []

numEdges2 = []
averageDegree2 = []
stddevDegree2 = []
radiusGyration2 = []
stddevRadiusGyration2 = []
avgLengthEdges2 = []
stdDevEdgeLength2 = []
numTriangles2 = []
avgPathLength2 = []
maximumDegree2 = []

count = 1

for i in range(count):
    print("Iteration " + str(i))
    g1 = SpSW(scalingFactor, a)

    degrees = g1.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree1.append(sum_of_edges/pop)
    stddevDegree1.append(statistics.stdev(dict(degrees).values()))
    numEdges1.append(g1.number_of_edges())
    
    degList1 = [i for i in degrees]
    maxDeg = max(degList1, key= lambda x: x[1])
    maximumDegree1.append(maxDeg[1])
    
    avgPathLength1.append(nx.average_shortest_path_length(g1))

    averageDistances = []
    radiusOfGyration = []

    for node in g1:
        maxDistance = 0
        for n in g1.neighbors(node):
            coord1, coord2 = agents[node], agents[n]
            dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
            averageDistances.append(dist)
            if dist > maxDistance:
                maxDistance = dist
        radiusOfGyration.append(maxDistance)

    radiusGyration1.append(sum(radiusOfGyration)/pop)
    stddevRadiusGyration1.append(statistics.stdev(radiusOfGyration))
    avgLengthEdges1.append(sum(averageDistances)/len(averageDistances))
    stdDevEdgeLength1.append(statistics.stdev(averageDistances))

    numTriangles1.append(sum(nx.triangles(g1).values()) / 3)



    g2 = SpatialSW2(p, scalingFactor, a)
    
    degrees = g2.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree2.append(sum_of_edges/pop)
    stddevDegree2.append(statistics.stdev(dict(degrees).values()))
    numEdges2.append(g2.number_of_edges())

    avgPathLength2.append(nx.average_shortest_path_length(g2))

    degList2 = [i for i in degrees]
    maxDeg = max(degList2, key= lambda x: x[1])
    maximumDegree2.append(maxDeg[1])

    averageDistances = []
    radiusOfGyration = []

    for node in g2:
        maxDistance = 0
        for n in g2.neighbors(node):
            coord1, coord2 = agents[node], agents[n]
            dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
            averageDistances.append(dist)
            if dist > maxDistance:
                maxDistance = dist
        radiusOfGyration.append(maxDistance)

    radiusGyration2.append(sum(radiusOfGyration)/pop)
    stddevRadiusGyration2.append(statistics.stdev(radiusOfGyration))
    avgLengthEdges2.append(sum(averageDistances)/len(averageDistances))
    stdDevEdgeLength2.append(statistics.stdev(averageDistances))

    numTriangles2.append(sum(nx.triangles(g2).values()) / 3)


print("Number of edges, Graph1:")
print(sum(numEdges1)/count)
print("Average Degree, Graph1:")
print(sum(averageDegree1)/count)
print("Standard Deviation of the Degree, Graph1:")
print(sum(stddevDegree1)/count)
print("Radius of Gyration, Graph1:")
print(sum(radiusGyration1)/count)
print("Standard Deviation of Radius Gyration, Graph1:")
print(sum(stddevRadiusGyration1)/count)
print("Average Distance Between Connected Edges, Graph1:")
print(sum(avgLengthEdges1)/count)
print("Standard Deviation of Distance Between Connected Edges, Graph1:")
print(sum(stdDevEdgeLength1)/count)
print("Number of Triangles, Graph1:")
print(sum(numTriangles1)/count)

print("Number of edges, Graph2:")
print(sum(numEdges2)/count)
print("Average Degree, Graph2:")
print(sum(averageDegree2)/count)
print("Standard Deviation of the Degree, Graph2:")
print(sum(stddevDegree2)/count)
print("Radius of Gyration, Graph2:")
print(sum(radiusGyration2)/count)
print("Standard Deviation of Radius Gyration, Graph2:")
print(sum(stddevRadiusGyration2)/count)
print("Average Distance Between Connected Edges, Graph2:")
print(sum(avgLengthEdges2)/count)
print("Standard Deviation of Distance Between Connected Edges, Graph2:")
print(sum(stdDevEdgeLength2)/count)
print("Number of Triangles, Graph2:")
print(sum(numTriangles2)/count)
print("Average Shortest Path Length, Graph2:")
print(sum(avgPathLength2)/count)
print("Maximum Degree, Graph2:")
print(sum(maximumDegree2)/count)


nx.draw(g1, nx.get_node_attributes(g1, 'pos'), node_size=1, width=0.25)
plt.show()

nx.draw(g2, nx.get_node_attributes(g2, 'pos'), node_size=1, width=0.25)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))