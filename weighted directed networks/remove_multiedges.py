import numpy as np
import sys

network = sys.argc([1]) # cit-HepTh, CoRA, Industry, lfr, Openflights, Pokemon, Sociopatterns
S = int(sys.argv[2]) # set random generator seed
np.random.seed(S)

f_network = open("data/" + network + ".network", "r")
source, target = [], []
for line in f_network:
    node1, node2, weight = np.array(line.split()).astype(int)
    for i in range(weight):
        source.append(node1)
        target.append(node2)
f_network.close()

M = len(source) # number of edges
num_edges = M
for f in np.arange(100, 0, -1):
    Mf = M * (f / 100.0) # number of edges to keep in the network    
    while (num_edges > Mf):
        x = np.random.randint(num_edges)
        source.pop(x)
        target.pop(x)
        num_edges -= 1
    fout = open("input_network/" + network + "_network_seed" + str(S) + "_" + str(f) + ".dat", "w")
    for node1, node2 in zip(source, target):
        fout.write(str(node1) + "\t" + str(node2) + " 1\n")   
    fout.close()     