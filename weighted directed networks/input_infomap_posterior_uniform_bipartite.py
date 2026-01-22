import numpy as np
import sys
import networkx as nx

S = int(sys.argv[1]) # network sample: S in {1..100}

network = "sociopatterns"
N, nA, nB = 318, 143, 175
alpha = np.log(N) / nA # bipartite prior link probability

for f in range(1, 101): # f represents fraction of links in the network
    
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
    
    #create prior network
    s, k, mass = np.zeros(N), np.zeros(N), np.ones(N)
    for node in range(N):
        s[node] = np.sum(W[node, :])    
        k[node] = np.sum(A[node, :])
        if k[node] > 0:
            mass[node] = np.float(s[node]) / k[node]
    norm = (np.sum(k)) / (np.sum(s))        

    Aprior = np.zeros(N * N).reshape(N, N) # prior adjacency matrix
    for node_i in range(nA):
        for node_j in range(nA, N):
            Aprior[node_i][node_j] = alpha * norm * mass[node_i] * mass[node_j]    
    
    #print posterior network
    fout = open("infomap_network_posterior_uniform/" + network + "_network_seed" + str(S) + "_" + str(f) + ".net", "w")   
    fout.write("*Vertices " + str(N) + "\n")
    for node in range(1, N + 1):
        fout.write(str(node) + ' ' + '"' + str(node) + '"\n')
    fout.write("*Edges " + str(nA * nB) + "\n")
    for node_i in range(nA):
        for node_j in range(nA, N):
            weight = W[node_i][node_j] + Aprior[node_i][node_j]
            fout.write(str(node_i + 1) + " " + str(node_j + 1) + " " + str(weight) + "\n")
    fout.close() 