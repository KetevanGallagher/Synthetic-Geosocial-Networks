import matplotlib.pyplot as plt
import networkx as nx
import time
import statistics

VAFile = "VA_zipcode_centroids.csv"
FBFile = "VirginiaZipCodeLinks.csv"

def countyInit(locFile, linkFile):
    seenSet = set()
    linkSet = set()
    global agents, pop
    agents = {}
    agentIdx = 0
    with open (linkFile) as file2:
        linkList = [line.strip() for line in file2]
    for link in linkList:
        link = link.split(",")
        if link[0] not in linkSet:
            linkSet.add(link[0])
    with open (locFile) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        loc = loc.split(",")
        tract = loc[1]
        if tract in linkSet:
            agents[int(tract)] = (float(loc[10]), float(loc[9]))
            agentIdx+=1
            seenSet.add(tract)
    pop = agentIdx

def getNearestNodes(k, locFile):
    global nearestNeighbors
    nearestNeighbors = {}
    with open (locFile) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        fromLoc, toLoc, sci = list(map(int, loc.split(",")))
        if fromLoc == toLoc:
            continue
        if fromLoc not in nearestNeighbors:
            nearestNeighbors[fromLoc] = [(sci, toLoc)]
        elif nearestNeighbors[fromLoc][-1][0] > sci:
            if len(nearestNeighbors[fromLoc]) < k:
                nearestNeighbors[fromLoc].append((sci, toLoc))
        else:
            i = len(nearestNeighbors[fromLoc])-1
            while nearestNeighbors[fromLoc][i][0] < sci and i > -1:
                i-=1
            nearestNeighbors[fromLoc].insert(i+1, (sci, toLoc))
            if len(nearestNeighbors[fromLoc]) > k:
                nearestNeighbors[fromLoc].pop()

def createGraph():
    network = nx.Graph()
    for agent in agents:
        coord = agents[agent]
        network.add_node(agent, pos=(coord[0], coord[1]))
    for fromLoc in nearestNeighbors:
        for toLoc in nearestNeighbors[fromLoc]:
            network.add_edge(fromLoc,toLoc[1])
            network.add_edge(toLoc[1],fromLoc)
    return network


start_time = time.time()
k = 16

countyInit(VAFile, FBFile)
getNearestNodes(k, FBFile)
FBg = createGraph()


degrees = FBg.degree()
sum_of_edges = sum(dict(degrees).values())
print("Average Degree:")
print(sum_of_edges/(pop))
print("Standard Deviation of the Degree:")
print(statistics.stdev(dict(degrees).values()))

avgPathLength = (nx.average_shortest_path_length(FBg))
degList = [i for i in degrees]
maxDeg = max(degList, key= lambda x: x[1])
maximumDegree = maxDeg[1]

averageDistances = []
radiusOfGyration = []

for node in FBg:
    maxDistance = 0
    for n in FBg.neighbors(node):
        coord1, coord2 = agents[node], agents[n]
        dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
        averageDistances.append(dist)
        if dist > maxDistance:
            maxDistance = dist
    radiusOfGyration.append(maxDistance)

print("Radius of Gyration:")
print(sum(radiusOfGyration)/(len(agents)))
print("Standard Deviation of Radius of Gyration:")
print(statistics.stdev(radiusOfGyration))
print("Average Distance Between Connected Edges:")
print(sum(averageDistances)/len(averageDistances))
print("Standard Deviation of Distance Between Connected Edges:")
print(statistics.stdev(averageDistances))
print("Number of Triangles:")
print(sum(nx.triangles(FBg).values()) / 3)
print("Average Shortest Path Length:")
print(avgPathLength)
print("Maximum Degree:")
print(maximumDegree)

nx.draw(FBg, nx.get_node_attributes(FBg, 'pos'), node_size=1, width =0.25)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

otherGraphFile = "" #file name of the graph you want to find the jaccard index of

totalJI = 0
trials = 20

for i in range(trials):
    with open(otherGraphFile) as f:
        exec(f.read())
    fbEdges = {*FBg.edges(data=False)}
    erIDEdges = {*g.edges(data=False)}
    erEdges = set()
    for edgeTuple in erIDEdges:
        newTuple = (idToZipCode[edgeTuple[0]], idToZipCode[edgeTuple[1]])
        erEdges.add(newTuple)
    ji = len(fbEdges.intersection(erEdges))/len(fbEdges | erEdges)
    totalJI += ji

print("Jaccard Index:")
avgJI = totalJI/trials
print(avgJI)