import numpy as np
import sys
import networkx as nx

network_size = {'cit-HepTh' : 4378, 'CoRA' : 3385, 'lfr' : 1000, 'Openflights' : 964}

network = sys.argv[1] # cit-HepTh, CoRA, lfr, Openflights

S = int(sys.argv[2]) # network sample: S in {1..100}
f = int(sys.argv[3]) # fraction of links in the network: f in {1..100}

N = network_size[network]

f_network = open("input_network/" + network + "_network_seed" + str(S) + "_" + str(f) + ".dat", "r")

# read edgelist and create adjacency matrix
A = np.zeros(N * N).reshape(N, N) # adjacency matrix
W = np.zeros(N * N).reshape(N, N) # weighted network
for line in f_network:
    node_i, node_j, w = np.array(line.split()).astype(int)
    A[int(node_i)-1][int(node_j)-1] = 1
    W[int(node_i)-1][int(node_j)-1] += w
f_network.close()    

#create prior network
s_in, k_in, mass_in, s_out, k_out, mass_out = np.zeros(N), np.zeros(N), np.ones(N), np.zeros(N), np.zeros(N), np.ones(N)
for node in range(N):
    s_out[node] = np.sum(W[node, :])    
    k_out[node] = np.sum(A[node, :])
    if k_out[node] > 0:
        mass_out[node] = np.float(s_out[node]) / k_out[node]
    s_in[node] = np.sum(W[:, node])
    k_in[node] = np.sum(A[:, node])
    if k_in[node] > 0:
        mass_in[node] = np.float(s_in[node]) / k_in[node]
norm = (np.log(N) / N) * (np.sum(k_out) + np.sum(k_in)) / (np.sum(s_out) + np.sum(s_in))        

# print posterior network    
fout = open("infomap_network_posterior_uniform/" + network + "_network_seed" + str(S) + "_" + str(f) + ".net", "w")   
fout.write("*Vertices " + str(N) + "\n")
for node in range(1, N + 1):
    fout.write(str(node) + ' ' + '"' + str(node) + '"\n')
fout.write("*Edges " + str(N * (N - 1)) + "\n")
for node_i in range(N):
    for node_j in range(N):
        if node_i == node_j:
            continue
        weight = W[node_i][node_j] + norm * mass_out[node_i] * mass_in[node_j]
        fout.write(str(node_i + 1) + " " + str(node_j + 1) + " " + str(weight) + "\n")
fout.close() 