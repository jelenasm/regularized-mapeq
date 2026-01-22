import numpy as np
import sys
import networkx as nx

S = int(sys.argv[1]) # network sample: S in {1..100}

network = "sociopatterns"
N, nA, nB = 318, 143, 175 # total number of nodes, number of nodes in the left partition and number of nodes in the right partition
prior_uniform = np.log(N) / nA # bipartite prior link probability

#read community assignment
nodes, labels = [], []
f_community = open(network + "/community.dat", "r")
for line in f_community:
    node, label = line.split()
    nodes.append(int(node) - 1)
    labels.append(int(label) - 1)
f_community.close()        
nodes = np.array(nodes)
labels = np.array(labels)   


for f in range(1, 101): #f represents fraction of links in the network
    
    #read edgelist and create adjacency matrix       
    f_network = open("input_network/" + network + "_network_seed" + str(S) + "_" + str(f) + ".dat", "r")

    A = np.zeros(N * N).reshape(N, N) # adjacency matrix
    W = np.zeros(N * N).reshape(N, N) # weighted network
    for line in f_network:
        node_i, node_j, w = np.array(line.split()).astype(int)
        A[int(node_i)-1][int(node_j)-1] = 1
        A[int(node_j)-1][int(node_i)-1] = 1
        W[int(node_i)-1][int(node_j)-1] += w
        W[int(node_j)-1][int(node_i)-1] += w
    f_network.close()    
    
    #create prior network based on metadata
    s, k, mass = np.zeros(N), np.zeros(N), np.ones(N)
    for node in range(N):
        s[node] = np.sum(W[node, :])    
        k[node] = np.sum(A[node, :])
        if k[node] > 0:
            mass[node] = np.float(s[node]) / k[node]
    norm = (np.sum(k)) / (np.sum(s))        

    Aprior = np.zeros(N * N).reshape(N, N) # prior adjacency matrix
    for m in np.unique(labels):
        nodes_in_m = nodes[labels == m]
        nodes_in_A = nodes_in_m[nodes_in_m <= nA]
        nodes_in_B = nodes_in_m[nodes_in_m > nA]
        Nmin = np.min([len(nodes_in_A), len(nodes_in_B)])
        if Nmin == 0:
            continue
        alpha = np.log(len(nodes_in_m)) / Nmin # link probability between two nodes with metadata label "m"
        for node_i in nodes_in_A:
            for node_j in nodes_in_B:
                Aprior[node_i][node_j] = alpha * norm * mass[node_i] * mass[node_j]    
    
    # print posterior network
    fout = open("infomap_network_posterior_metadata/" + network + "_network_seed" + str(S) + "_" + str(f) + ".net", "w")   
    fout.write("*Vertices " + str(N) + "\n")
    for node in range(1, N + 1):
        fout.write(str(node) + ' ' + '"' + str(node) + '"\n')
    fout.write("*Edges " + str(nA * nB) + "\n")
    for node_i in range(nA):
        for node_j in range(nA, N):
            weight = W[node_i][node_j] + Aprior[node_i][node_j] + prior_uniform * norm * mass[node_i] * mass[node_j]    
            fout.write(str(node_i + 1) + " " + str(node_j + 1) + " " + str(weight) + "\n")
    fout.close() 