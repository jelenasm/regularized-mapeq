import numpy as np
import sys
import networkx as nx

network_size = {'cit-HepTh' : 4378, 'CoRA' : 3385, 'Industry' : 1778, 'lfr' : 1000, 'Openflights' : 964, 'Pokemon' : 743, 'Sociopatterns' : 318}

network = sys.argv[1] # cit-HepTh, CoRA, Industry, lfr, Openflights, Pokemon, Sociopatterns
S = int(sys.argv[2]) # sample
N = network_size[network]

for f in range(100, 0, -1):
    
    f_network = open("input_network/" + network + "_network_seed" + str(S) + "_" + str(f) + ".dat", "r")
    if network in ['cit-HepTh', 'CoRA', 'lfr', 'Openflights']:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    source, target = [], []
    for line in f_network:
        node1, node2, w = line.split()
        if not(G.has_edge(node1, node2)):
            G.add_edge(node1, node2)
            G[node1][node2]["weight"] = 0
            source.append(node1)
            target.append(node2)
        G[node1][node2]["weight"] += 1
    f_network.close()    
    
    fout = open("infomap_network/" + network + "_network_seed" + str(S) + "_" + str(f) + ".net", "w")   
    fout.write("*Vertices " + str(N) + "\n")
    for node in range(1, N + 1):
        fout.write(str(node) + ' ' + '"' + str(node) + '"\n')
    fout.write("*Edges " + str(G.number_of_edges()) + "\n")
    for node1, node2 in zip(source, target):
        fout.write(node1 + " " + node2 + " " + str(G[node1][node2]['weight']) + "\n")  
    fout.close() 