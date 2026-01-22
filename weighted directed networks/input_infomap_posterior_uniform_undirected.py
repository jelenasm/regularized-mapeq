import numpy as np
import sys
import networkx as nx

N = int(sys.argv[1]) # network size
input_network = sys.argv[2] # link list: "node_i node_j weight\n" (node indices start from 1)

alpha = np.log(N) / N

# read network  
f_network = open(input_network, "r") 
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
norm = alpha * (np.sum(k)) / (np.sum(s))        

Aprior = np.zeros(N * N).reshape(N, N) # prior adjacency matrix
for node_i in range(N):
    for node_j in range(N):
        if node_i != node_j:
            Aprior[node_i][node_j] = norm * mass[node_i] * mass[node_j]    


fout = open("network-posterior.net", "w")   
fout.write("*Vertices " + str(N) + "\n")
for node in range(1, N + 1):
    fout.write(str(node) + ' ' + '"' + str(node) + '"\n')
fout.write("*Edges " + str(N * (N - 1) / 2.0) + "\n")
for node_i in range(N - 1):
    for node_j in range(node_i + 1, N):
        weight = W[node_i][node_j] + Aprior[node_i][node_j]
        fout.write(str(node_i + 1) + " " + str(node_j + 1) + " " + str(weight) + "\n")
fout.close() 
