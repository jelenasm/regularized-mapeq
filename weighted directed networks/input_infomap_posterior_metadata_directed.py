import numpy as np
import sys

network_size = {'cit-HepTh' : 4378, 'CoRA' : 3385, 'lfr' : 1000, 'Openflights' : 964}

network = sys.argv[1] # cit-HepTh, CoRA, lfr, Openflights

S = int(sys.argv[2]) # network sample: S in {1..100}
f = int(sys.argv[3]) # fraction of links in the network: f in {1..100}

p = np.float(sys.argv[4]) # fraction of nodes with random community assignment (for lfr network only)

N = network_size[network]
prior_uniform = np.log(N) / N

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
M = np.max(labels) # number of labels

Np = int(p * N)
noise_id = []
while len(noise_id) < Np:
    node = np.random.randint(N)
    if node in noise_id:
        continue
    noise_id.append(node)
    
for node in noise_id:
    m = labels[node]
    m_rand = np.random.randint(M)        
    while(m == m_rand):
        m_rand = np.random.randint(M)                
    labels[node] = m_rand
    
#read edgelist and create adjacency matrix   
f_network = open("input_network/" + network + "_network_seed" + str(S) + "_" + str(f) + ".dat", "r")

A = np.zeros(N * N).reshape(N, N) # adjacency matrix
W = np.zeros(N * N).reshape(N, N) # weighted network
for line in f_network:
    node_i, node_j, w = np.array(line.split()).astype(int)
    A[int(node_i)-1][int(node_j)-1] = 1
    W[int(node_i)-1][int(node_j)-1] += w
f_network.close()    

#create prior network based on metadata

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
norm = (np.sum(k_out) + np.sum(k_in)) / (np.sum(s_out) + np.sum(s_in))        

Aprior = np.zeros(N * N).reshape(N, N) # prior adjacency matrix
for m in np.unique(labels):
    nodes_in_m = nodes[labels == m]
    Nm = len(nodes_in_m)
    alpha = np.log(Nm) / Nm # link probability between two nodes with the same label "m"
    for node_i in nodes_in_m:
        for node_j in nodes_in_m:
            if node_i == node_j:
                continue
            Aprior[node_i][node_j] = alpha * norm * mass_out[node_i] * mass_in[node_j]  


# print posterior network
fout = open("infomap_network_posterior_metadata/" + network + "_p" + str(p) + "_seed" + str(S) + "_" + str(f) + ".net", "w")
fout.write("*Vertices " + str(N) + "\n")
for node in range(1, N + 1):
    fout.write(str(node) + ' ' + '"' + str(node) + '"\n')
fout.write("*Edges " + str(N*(N-1)) + "\n")
for node_i in range(N):
    for node_j in range(N):
        if node_i == node_j:
            continue        
        weight = W[node_i][node_j] + Aprior[node_i][node_j] + prior_uniform * norm * mass_out[node_i] * mass_in[node_j]
        fout.write(str(node_i + 1) + " " + str(node_j + 1) + " " + str(weight) + "\n")
fout.close()