
# coding: utf-8

import matplotlib.pyplot as plt
import pylab as pl
import random as rd
import numpy as np
import networkx as nx
import scipy as sp
from scipy import stats
from collections import Counter
import time

# Code from Generating and analyzing spatial social networks. Function "locInit" was added so
# real world data could be used.

# n = population size
# density = density of the grid cell
def init(n,density):
    global agents, pop
    width = int(round((n/density)**0.5))
    height = int(round((n/density)**0.5))
    config = np.zeros([height, width])
    pop = 0 #actual population size
    for x in range(height):
        for y in range(width):
            if rd.random() < density:
                config[x,y] = 1
                pop += 1

    # Populating the agents matrix with a random sequence of agents
    seq = [x for x in range(pop)]
    rd.shuffle(seq)
    agents = np.zeros([height, width])
    for r in range(height):
        for t in range(width):
            if agents[r,t] == 0:
                agents[r,t] = -1
    z = 0
    for i in range(height):
        for j in range(width):
            if config[i,j] == 1:
                agents[i,j]=seq[z]
                z += 1

def spatialInit(n,density):
    global agents, pop
    width = int(round((n/density)**0.5))
    height = int(round((n/density)**0.5))
    config = np.zeros([height, width])
    pop = 0 #actual population size
    for x in range(height):
        for y in range(width):
            if rd.random() < density:
                config[x,y] = 1
                pop += 1

    seq = [x for x in range(pop)]
    rd.shuffle(seq)
    agents = {}
    z = 0
    for i in range(height):
        for j in range(width):
            if config[i,j] == 1:
                agents[seq[z]]=(i, j)
                z += 1

def locInit(file):
    global agents, pop, idToZipCode
    agents, idToZipCode = {}, {}
    agents = {}
    agentIdx = 0
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    locList = locList[1:]
    pop = len(locList)
    seq = [x for x in range(pop)]
    rd.shuffle(seq)
    agents = {}
    z = 0
    for agentIdx in seq:
        loc = locList[z]
        loc = loc.split(",")
        agents[agentIdx] = (float(loc[2]), float(loc[1]))
        idToZipCode[agentIdx] = int(loc[0])
        z+=1
    pop = len(agents)

def Select(net,new,m,alpha):
    deg = {node:val for (node, val) in net.degree()}
    new_coord = agents[new]
    nodeJ_coord = list()
    targets = list()
    exponent = alpha * (-1)
    while len(targets) < m:
        nodeJ = rd.choice(list(deg.keys()))
        while nodeJ in targets:
            nodeJ = rd.choice(list(deg.keys()))
        nodeJ_coord = agents[nodeJ]
        d = (float(nodeJ_coord[0] - new_coord[0])**2 + (float(nodeJ_coord[1] - new_coord[1])**2))**0.5
        d_alpha = (d**(exponent))
        ConnPr = d_alpha * deg[nodeJ]
        chance = rd.random()
        if (ConnPr <= 1 and ConnPr > chance):
            targets.append(nodeJ)
    return targets


def SpatialNetSF(m,alpha):
    network = nx.empty_graph(m)
    targets = list(range(m))
    source = m
    while source < pop-1:
        network.add_edges_from(zip([source]*m,targets))
        source += 1
        targets = Select(network,source,m,alpha)
    nx.set_node_attributes(network, agents, name="pos")
    return network

start_time = time.time()
m = 7
a = 3
locInit("FairfaxCensusTractLatLong.csv")
g = SpatialNetSF(m, a)
n=len(agents)

nx.draw(g, nx.get_node_attributes(g, 'pos'), node_size=1, width=0.25)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))