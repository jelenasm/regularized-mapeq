import numpy as np
import pandas as pd
import random
import networkx as nx
import infomap
from sklearn.metrics.cluster import adjusted_mutual_info_score
import json
import re
from pathlib import Path
import math
import csv

import seaborn as sns
import matplotlib.pyplot as plt

'''
posterior_original aB --> bC
posterior_physical aB --> bC
posterior_1b aB --> cD (target state node can be any state node except within the same physical node; self loops are not allowed)
posterior_2a aB --> cB (self loops are not allowed)
posterior_2b aA --> bB (no constraints)
'''

def create_posterior_original(inputfile, outputfile):

    if Path(outputfile).is_file():
        print(f"Skip generate posterior original network {outputfile} from {inputfile}")
        return
    
    vertices, vertices_print, states = [], [], {}
    links, weight = {}, []
    in_degree, in_strength, out_degree, out_strength = {}, {}, {}, {}
    total_degree, total_strength = 0, 0

    with open(inputfile, "r") as fin:
        line = fin.readline()    
        line_vertices = line
        
        line = fin.readline()
        while not (line.startswith("*States")):
            vertices.append(line.split('"')[1])
            vertices_print.append(line)
            line = fin.readline()
        line_states = line
            
        line = fin.readline()
        while not (line.startswith("*Links")):
            state_name = line.split('"')[1]
            states[line.split(' ')[0]] = state_name
            in_degree[state_name], in_strength[state_name] = 0, 0
            out_degree[state_name], out_strength[state_name] = 0, 0
            line = fin.readline()
        line_links = line
            
        line = fin.readline()
        while line:
            arc = line.split()
            n1, n2, w = arc[0], arc[1], int(arc[2])
            links[states[n1] + "\t" + states[n2]] = w
            weight.append(w)
            out_degree[states[n1]] += 1
            out_strength[states[n1]] += w
            in_degree[states[n2]] += 1
            in_strength[states[n2]] += w   
            total_degree += 2
            total_strength += 2
            line = fin.readline()

    in_mass, out_mass = {}, {}        
    for snode in states.values():
        if in_degree[snode] > 0:
            in_mass[snode] = in_strength[snode] / in_degree[snode]
        if out_degree[snode] > 0:
            out_mass[snode] = out_strength[snode] / out_degree[snode]           
    in_mass_min  = min(in_mass.values())
    out_mass_min = min(out_mass.values()) 

    num_nodes = len(vertices)
    prior_norm = np.log(num_nodes) / (2 * num_nodes * num_nodes) * total_degree / total_strength   
            
    states_prior = {}
    with open(outputfile, "w") as fout:
        
        fout.write(line_vertices)
        for line in vertices_print:
            fout.write(line)

        fout.write("*States\n")
        state_ID = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                source = vertices[i]
                target = vertices[j]
                node_ID = j + 1
                state_ID += 1
                state_label = "{" + str(source) + "}_" + str(target)
                states_prior[state_label] = state_ID
                if not (state_label in in_mass):
                    in_mass[state_label] = in_mass_min
                if not (state_label in out_mass):
                    out_mass[state_label] = out_mass_min                
                fout.write(str(state_ID) + ' ' + str(node_ID) + ' "' + state_label + '"\n')
        
        fout.write("*Links\n")
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n2 == n1:
                    continue
                for n3 in range(num_nodes):
                    if n3 == n2:
                        continue
                    
                    state_source = "{" + str(vertices[n1]) + "}_" + str(vertices[n2])
                    state_target = "{" + str(vertices[n2]) + "}_" + str(vertices[n3])
                    prior_link = state_source + '\t' + state_target
                    
                    if prior_link in links:
                        w = links[prior_link] + prior_norm * out_mass[state_source] * in_mass[state_target]
                    else:
                        w = prior_norm * out_mass[state_source] * in_mass[state_target]            
                                                    
                    fout.write(str(states_prior[state_source]) + " " + str(states_prior[state_target]) + " " + str(w) + "\n")   



def create_posterior_physical(inputfile, outputfile):
    """
    Physical posterior is equivalent to the original posterior.
    """    

    if Path(outputfile).is_file():
        print(f"Skip generate posterior physical network {outputfile} from {inputfile}")
        return
    # Read a state network:

    vertices, vertices_print, states = [], [], {}
    links, weight = {}, []
    in_degree, in_strength, out_degree, out_strength = {}, {}, {}, {}
    total_degree, total_strength = 0, 0

    with open(inputfile, "r") as fin:
        line = fin.readline()    
        line_vertices = line
        
        line = fin.readline()
        while not (line.startswith("*States")):
            vertices.append(line.split('"')[1])
            vertices_print.append(line)
            line = fin.readline()
        line_states = line
            
        line = fin.readline()
        while not (line.startswith("*Links")):
            state_name = line.split('"')[1]
            states[line.split(' ')[0]] = state_name
            in_degree[state_name], in_strength[state_name] = 0, 0
            out_degree[state_name], out_strength[state_name] = 0, 0
            line = fin.readline()
        line_links = line
            
        line = fin.readline()
        while line:
            arc = line.split()
            n1, n2, w = arc[0], arc[1], int(arc[2])
            links[states[n1] + "\t" + states[n2]] = w
            weight.append(w)
            out_degree[states[n1]] += 1
            out_strength[states[n1]] += w
            in_degree[states[n2]] += 1
            in_strength[states[n2]] += w   
            total_degree += 2
            total_strength += 2
            line = fin.readline()

    in_mass, out_mass = {}, {}        
    for snode in states.values():
        if in_degree[snode] > 0:
            in_mass[snode] = in_strength[snode] / in_degree[snode]
        if out_degree[snode] > 0:
            out_mass[snode] = out_strength[snode] / out_degree[snode]           
    in_mass_min  = min(in_mass.values())
    out_mass_min = min(out_mass.values()) 

    num_nodes = len(vertices)
    prior_norm = np.log(num_nodes) / (2 * num_nodes * num_nodes) * total_degree / total_strength   
            
    states_prior = {}
    with open(outputfile, "w") as fout:
        
        fout.write(line_vertices)
        for line in vertices_print:
            fout.write(line)

        fout.write("*States\n")
        state_ID = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                source = vertices[i]
                target = vertices[j]
                node_ID = j + 1
                state_ID += 1
                state_label = "{" + str(source) + "}_" + str(target)
                states_prior[state_label] = state_ID
                if not (state_label in in_mass):
                    in_mass[state_label] = in_mass_min
                if not (state_label in out_mass):
                    out_mass[state_label] = out_mass_min                
                fout.write(str(state_ID) + ' ' + str(node_ID) + ' "' + state_label + '"\n')
        
        fout.write("*Links\n")
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n2 == n1:
                    continue
                for n3 in range(num_nodes):
                    if n3 != n2:
                        continue
                    for n4 in range(num_nodes):
                        if (n4 == n3) | (n4==n2):
                            continue
                    
                        state_source = "{" + str(vertices[n1]) + "}_" + str(vertices[n2])
                        state_target = "{" + str(vertices[n3]) + "}_" + str(vertices[n4])
                        prior_link = state_source + '\t' + state_target
                        
                        if prior_link in links:
                            w = links[prior_link] + prior_norm * out_mass[state_source] * in_mass[state_target]
                        else:
                            w = prior_norm * out_mass[state_source] * in_mass[state_target]            
                                                        
                        fout.write(str(states_prior[state_source]) + " " + str(states_prior[state_target]) + " " + str(w) + "\n")

def create_posterior_1b(inputfile, outputfile):
    """
    Like physical but allow n2 and n3 to be different. N times more state-nodes
    """

    if Path(outputfile).is_file():
        print(f"Skip generate posterior 1b network {outputfile} from {inputfile}")
        return
    
    # Read a state network:

    vertices, vertices_print, states = [], [], {}
    links, weight = {}, []
    in_degree, in_strength, out_degree, out_strength = {}, {}, {}, {}
    total_degree, total_strength = 0, 0

    with open(inputfile, "r") as fin:
        line = fin.readline()    
        line_vertices = line
        
        line = fin.readline()
        while not (line.startswith("*States")):
            vertices.append(line.split('"')[1])
            vertices_print.append(line)
            line = fin.readline()
        line_states = line
            
        line = fin.readline()
        while not (line.startswith("*Links")):
            state_name = line.split('"')[1]
            states[line.split(' ')[0]] = state_name
            in_degree[state_name], in_strength[state_name] = 0, 0
            out_degree[state_name], out_strength[state_name] = 0, 0
            line = fin.readline()
        line_links = line
            
        line = fin.readline()
        while line:
            arc = line.split()
            n1, n2, w = arc[0], arc[1], int(arc[2])
            links[states[n1] + "\t" + states[n2]] = w
            weight.append(w)
            out_degree[states[n1]] += 1
            out_strength[states[n1]] += w
            in_degree[states[n2]] += 1
            in_strength[states[n2]] += w   
            total_degree += 2
            total_strength += 2
            line = fin.readline()

    in_mass, out_mass = {}, {}        
    for snode in states.values():
        if in_degree[snode] > 0:
            in_mass[snode] = in_strength[snode] / in_degree[snode]
        if out_degree[snode] > 0:
            out_mass[snode] = out_strength[snode] / out_degree[snode]           
    in_mass_min  = min(in_mass.values())
    out_mass_min = min(out_mass.values()) 

    num_nodes = len(vertices)
    prior_norm = np.log(num_nodes) / (2 * num_nodes * num_nodes * num_nodes) * total_degree / total_strength   
            
    states_prior = {}
    with open(outputfile, "w") as fout:
        
        fout.write(line_vertices)
        for line in vertices_print:
            fout.write(line)

        fout.write("*States\n")
        state_ID = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                source = vertices[i]
                target = vertices[j]
                node_ID = j + 1
                state_ID += 1
                state_label = "{" + str(source) + "}_" + str(target)
                states_prior[state_label] = state_ID
                if not (state_label in in_mass):
                    in_mass[state_label] = in_mass_min
                if not (state_label in out_mass):
                    out_mass[state_label] = out_mass_min                
                fout.write(str(state_ID) + ' ' + str(node_ID) + ' "' + state_label + '"\n')
        
        fout.write("*Links\n")
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n2 == n1:
                    continue
                for n3 in range(num_nodes):
                    for n4 in range(num_nodes):
                        if (n4 == n3) | (n4==n2):
                            continue
                    
                        state_source = "{" + str(vertices[n1]) + "}_" + str(vertices[n2])
                        state_target = "{" + str(vertices[n3]) + "}_" + str(vertices[n4])
                        prior_link = state_source + '\t' + state_target
                        
                        if prior_link in links:
                            w = links[prior_link] + prior_norm * out_mass[state_source] * in_mass[state_target]
                        else:
                            w = prior_norm * out_mass[state_source] * in_mass[state_target]            
                                                        
                        fout.write(str(states_prior[state_source]) + " " + str(states_prior[state_target]) + " " + str(w) + "\n")


def create_posterior_2a(inputfile, outputfile):
    """
    Same as 1 but without the constraint of no links within the same physical node, i.e. allow n2 == n4
    (n1 - n2) -> (n3 - n4)
    (n2 is always equal to n3 for real links, but this constraint is removed for the prior)
    Avoid self-loops only
    """
    # Read a state network:

    if Path(outputfile).is_file():
        print(f"Skip generate posterior 2a network {outputfile} from {inputfile}")
        return

    vertices, vertices_print, states = [], [], {}
    links, weight = {}, []
    in_degree, in_strength, out_degree, out_strength = {}, {}, {}, {}
    total_degree, total_strength = 0, 0

    with open(inputfile, "r") as fin:
        line = fin.readline()    
        line_vertices = line
        
        line = fin.readline()
        while not (line.startswith("*States")):
            vertices.append(line.split('"')[1])
            vertices_print.append(line)
            line = fin.readline()
        line_states = line
            
        line = fin.readline()
        while not (line.startswith("*Links")):
            state_name = line.split('"')[1]
            states[line.split(' ')[0]] = state_name
            in_degree[state_name], in_strength[state_name] = 0, 0
            out_degree[state_name], out_strength[state_name] = 0, 0
            line = fin.readline()
        line_links = line
            
        line = fin.readline()
        while line:
            arc = line.split()
            n1, n2, w = arc[0], arc[1], int(arc[2])
            links[states[n1] + "\t" + states[n2]] = w
            weight.append(w)
            out_degree[states[n1]] += 1
            out_strength[states[n1]] += w
            in_degree[states[n2]] += 1
            in_strength[states[n2]] += w   
            total_degree += 2
            total_strength += 2
            line = fin.readline()

    in_mass, out_mass = {}, {}        
    for snode in states.values():
        if in_degree[snode] > 0:
            in_mass[snode] = in_strength[snode] / in_degree[snode]
        if out_degree[snode] > 0:
            out_mass[snode] = out_strength[snode] / out_degree[snode]           
    in_mass_min  = min(in_mass.values())
    out_mass_min = min(out_mass.values()) 

    num_nodes = len(vertices)
    prior_norm = np.log(num_nodes) / (2 * num_nodes * num_nodes * num_nodes) * total_degree / total_strength   
            
    states_prior = {}
    with open(outputfile, "w") as fout:
        
        fout.write(line_vertices)
        for line in vertices_print:
            fout.write(line)

        fout.write("*States\n")
        state_ID = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                source = vertices[i]
                target = vertices[j]
                node_ID = j + 1
                state_ID += 1
                state_label = "{" + str(source) + "}_" + str(target)
                states_prior[state_label] = state_ID
                if not (state_label in in_mass):
                    in_mass[state_label] = in_mass_min
                if not (state_label in out_mass):
                    out_mass[state_label] = out_mass_min                
                fout.write(str(state_ID) + ' ' + str(node_ID) + ' "' + state_label + '"\n')
        
        fout.write("*Links\n")
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                if n2 == n1:
                    continue
                for n3 in range(num_nodes):
                    for n4 in range(num_nodes):
                        if (n4 == n3):
                            continue
                    
                        state_source = "{" + str(vertices[n1]) + "}_" + str(vertices[n2])
                        state_target = "{" + str(vertices[n3]) + "}_" + str(vertices[n4])
                        prior_link = state_source + '\t' + state_target
                        
                        if prior_link in links:
                            w = links[prior_link] + prior_norm * out_mass[state_source] * in_mass[state_target]
                        else:
                            w = prior_norm * out_mass[state_source] * in_mass[state_target]            
                                                        
                        fout.write(str(states_prior[state_source]) + " " + str(states_prior[state_target]) + " " + str(w) + "\n")

def create_posterior_2b(inputfile, outputfile):
    """
    Same as 1 but without the constraint of no links within the same physical node, i.e. allow n2 == n4
    (n1 - n2) -> (n3 - n4)
    (n2 is always equal to n3 for real links, but this constraint is removed for the prior)
    Include self-loops

    This should correspond to regularized Infomap for memory networks
    """
    # Read a state network:

    if Path(outputfile).is_file():
        print(f"Skip generate posterior 2b network {outputfile} from {inputfile}")
        return

    vertices, vertices_print, states = [], [], {}
    links, weight = {}, []
    in_degree, in_strength, out_degree, out_strength = {}, {}, {}, {}
    total_degree, total_strength = 0, 0

    with open(inputfile, "r") as fin:
        line = fin.readline()    
        line_vertices = line
        
        line = fin.readline()
        while not (line.startswith("*States")):
            vertices.append(line.split('"')[1])
            vertices_print.append(line)
            line = fin.readline()
        line_states = line
            
        line = fin.readline()
        while not (line.startswith("*Links")):
            state_name = line.split('"')[1]
            states[line.split(' ')[0]] = state_name
            in_degree[state_name], in_strength[state_name] = 0, 0
            out_degree[state_name], out_strength[state_name] = 0, 0
            line = fin.readline()
        line_links = line
            
        line = fin.readline()
        while line:
            arc = line.split()
            n1, n2, w = arc[0], arc[1], int(arc[2])
            links[states[n1] + "\t" + states[n2]] = w
            weight.append(w)
            out_degree[states[n1]] += 1
            out_strength[states[n1]] += w
            in_degree[states[n2]] += 1
            in_strength[states[n2]] += w   
            total_degree += 2
            total_strength += 2
            line = fin.readline()

    in_mass, out_mass = {}, {}        
    for snode in states.values():
        if in_degree[snode] > 0:
            in_mass[snode] = in_strength[snode] / in_degree[snode]
        if out_degree[snode] > 0:
            out_mass[snode] = out_strength[snode] / out_degree[snode]           
    in_mass_min  = min(in_mass.values())
    out_mass_min = min(out_mass.values()) 

    num_nodes = len(vertices)
    prior_norm = np.log(num_nodes) / (2 * num_nodes * num_nodes * num_nodes) * total_degree / total_strength   
            
    states_prior = {}
    with open(outputfile, "w") as fout:
        
        fout.write(line_vertices)
        for line in vertices_print:
            fout.write(line)

        fout.write("*States\n")
        state_ID = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                source = vertices[i]
                target = vertices[j]
                node_ID = j + 1
                state_ID += 1
                state_label = "{" + str(source) + "}_" + str(target)
                states_prior[state_label] = state_ID
                if not (state_label in in_mass):
                    in_mass[state_label] = in_mass_min
                if not (state_label in out_mass):
                    out_mass[state_label] = out_mass_min                
                fout.write(str(state_ID) + ' ' + str(node_ID) + ' "' + state_label + '"\n')
        
        fout.write("*Links\n")
        for n1 in range(num_nodes):
            for n2 in range(num_nodes):
                for n3 in range(num_nodes):
                    for n4 in range(num_nodes):
                    
                        state_source = "{" + str(vertices[n1]) + "}_" + str(vertices[n2])
                        state_target = "{" + str(vertices[n3]) + "}_" + str(vertices[n4])
                        prior_link = state_source + '\t' + state_target
                        
                        if prior_link in links:
                            w = links[prior_link] + prior_norm * out_mass[state_source] * in_mass[state_target]
                        else:
                            w = prior_norm * out_mass[state_source] * in_mass[state_target]            
                                                        
                        fout.write(str(states_prior[state_source]) + " " + str(states_prior[state_target]) + " " + str(w) + "\n")




# ########################################################################
# # Andrea's code                                                        #
# ########################################################################

'''
    select trigrams out
    of all possible trigrams
    with two probabilities
    pin
    pout
'''

def get_communities(mems):
    
    '''
        returns { 0:[node1, node2, ...], 1:[node3, node4, ...], ...}
    '''
    communities={}
    for n in range(len(mems)):
        for c in mems[n]:
            if c not in communities:
                communities[c]=[]
            communities[c].append(n)
    return communities


def check_pair_is_ok(pair_one, rc, pairs_coms):

    return (pair_one not in pairs_coms or pairs_coms[pair_one]==rc)
    


def select_random_trigram_incommunities(communities, n_sample):

    '''
        -assuming all communities are equally sized-
        first select a random community
        then select a random trigram
    '''
    # print(f"select_random_trigram_incommunities: {communities=}, {n_sample=}")
    
    #print 'in', n_sample
    
    # {(t1, t2): com}
    pairs_coms={}
    trigrams=[]
    random_communities=[]
    community_ids = list(communities.keys())
    for i in range(n_sample):
        try:
            random_communities.append(random.sample(community_ids, 1)[0])
        except:
            print(f"Error sampling communities at sample {i} ")
            raise
    
    
    rejections=0
    for rc in random_communities:
        
        while True:
            try:
                trial_tri=random.sample(communities[rc], 3)
            except:
                print(f"Error sampling trigram from community {rc} with members {communities[rc]} ")
                raise
            pair_one=(trial_tri[0], trial_tri[1])
            pair_two=(trial_tri[1], trial_tri[2])
            if  check_pair_is_ok(pair_one, rc, pairs_coms) and \
                check_pair_is_ok(pair_two, rc, pairs_coms):
                trigrams.append(trial_tri)
                pairs_coms[pair_one]=rc
                pairs_coms[pair_two]=rc
                break
            else:
                rejections+=1
    
    #print 'rejections in sampling', rejections
    return trigrams, pairs_coms


def outcommunities(l1, l2, l3):
    '''
        returns true if the intersection between l1 and l2 is zero
        or l2 l3 is zero 
        or l1 l3 is zero
    '''
    
    if len(set(l1) & set(l2))==0:
        return True
    if len(set(l2) & set(l3))==0:
        return True
    if len(set(l1) & set(l3))==0:
        return True
    
    return False




def select_random_trigram_outcommunities(mems, n_sample):
    '''
        select a random trigram and accepts it 
        only if 
        not all the three nodes in it belong to the same communities
    '''
    
    #print 'out', n_sample
    trigrams=[]
    while len(trigrams)<n_sample:
        trigram=random.sample(range(len(mems)), 3)
        if outcommunities(mems[trigram[0]], mems[trigram[1]], mems[trigram[2]]):
            trigrams.append(trigram)
    return trigrams


def compute_mems(N, M, nc):
    '''
        we have N nodes and M communities
        of nc nodes each
        we write all the memberships in random order
        and we assign them to random nodes.
    '''
    all_mems=[]
    for i in range(M):
        for n in range(nc):
            all_mems.append(i)
    
    # numbers to be assigned to nodes
    all_mems=random.sample(all_mems, len(all_mems))
    
    # nodes in random order
    all_nodes=random.sample(range(N), N)
    
    # mems[node] -> list of memberships
    mems=[]
    for i in range(N):
        mems.append([])
    
    try_counter=0
    while True:
        discarded=[]
        for counter, m in enumerate(all_mems):
            node= counter%len(all_nodes)
            if m not in mems[all_nodes[node]]:
                mems[all_nodes[node]].append(m)
            else:
                discarded.append(m)
        #print 'discarded', discarded
        all_mems=list(discarded)
        all_nodes=random.sample(range(N), N)
        if try_counter>100:
            break
        if len(discarded)==0:
            break
    
    
    communities=get_communities(mems)
    
    return mems, communities

def select_random_trigram_outcommunities_without_creating_more_state_nodes(pairs_coms, n_sample):
    '''
        Create a list of random trigrams outside planted communities

        1. Select a random pair p1 from pairs_coms, and the assigned module m1 = pairs_coms[p1]
        2. Select a random pair p2 from pairs_coms, where the assigned module m2 = pairs_coms[p2] is different from m1 and
              the second node of p1 is the first node of p2
        3. Create the trigram (p1[0], p1[1], p2[1])

        New algorithm to avoid sampling to find a valid trigram:
        1. Build a mapping from each node to the list of pairs starting with that node
        2. Select a random pair p1 from pairs_coms, and the assigned module m1 = pairs_coms[p1]
        3. From the mapping, get the list of candidate pairs p2 where the assigned module m2 = pairs_coms[p2] is different from m1 and
              the second node of p1 is the first node of p2
        4. Create the trigram (p1[0], p1[1], p2[1])
    '''

    node_pairs_map = {}
    for pair in pairs_coms.keys():
        if pair[0] not in node_pairs_map:
            node_pairs_map[pair[0]] = []
        node_pairs_map[pair[0]].append(pair)
    
    
    num_trials = 0
    max_num_trials = n_sample * 10
    num_invalid_trigrams = 0
    trigrams=[]
    # print(f"select_random_trigram_outcommunities_without_creating_more_state_nodes: {n_sample=}, {pairs_coms=}")
    while len(trigrams)<n_sample:
        num_trials += 1
        if num_trials > max_num_trials:
            print(f"Warning: Reached maximum number of trials ({max_num_trials}) while selecting only {len(trigrams)}/{n_sample} out-community trigrams with rejection rate {num_invalid_trigrams / num_trials:%}.")
            break

        pair_one = random.sample(list(pairs_coms.keys()), 1)[0]
        module_one = pairs_coms[pair_one]
        # print(f"Iteration {num_trials}: Selected pair one {pair_one} with module {module_one}")
        
        # select pair two among valid trigrams (second node of pair one == first node of pair two)
        candidate_pairs_two = node_pairs_map.get(pair_one[1], [])
        # filter to keep only those pairs with different module
        candidate_pairs_two = [p for p in candidate_pairs_two if (pairs_coms[p] != module_one)]
        # print(f"Candidate pairs two: {candidate_pairs_two}")
        
        if len(candidate_pairs_two) == 0:
            num_invalid_trigrams += 1
            continue
        
        pair_two = random.sample(candidate_pairs_two, 1)[0]
        
        trigram = (pair_one[0], pair_one[1], pair_two[1])
        trigrams.append(trigram)
        
    # print(f"Success: Created {n_sample} out-community trigrams on {num_trials} trials with rejection rate {num_invalid_trigrams / num_trials:%}.")
    return trigrams

def build_syn_network(N, om, nc, E, mu, r, force_regenerate=False):

    # Skip if files already exists
    if not force_regenerate and Path(get_planted_partition_filename(N, om, nc, E, mu, r)).is_file():
        print(f"Skip generate synthetic data for {get_suffix(N, om, nc, E, mu, r)}")
        return
    
    M = om * 4

    #print 'M >>', M
    # computing the memberships
    mems, communities=compute_mems(N, M, nc)

    # selects the trigram inside the communities
    trigrams_all, pairs_coms=select_random_trigram_incommunities(communities, int((1-mu)*E))
    # selects the trigram outside the communities
    #trigrams_all+=select_random_trigram_outcommunities(mems, int(mu*E))
    trigrams_all+=select_random_trigram_outcommunities_without_creating_more_state_nodes(pairs_coms, int(mu*E))

    
    #print len(trigrams_all), 'trigrams'
    
    suffix = get_suffix(N, om, nc, E, mu, r)
    
    # computing weights
    trigrams_all_hist={}
    digrams_all_hist={}
    for t in trigrams_all:
        trigrams_all_hist[tuple(t)]=trigrams_all_hist.get(tuple(t),0)+1
        digrams_all_hist[(t[0], t[1])]=digrams_all_hist.get((t[0], t[1]),0)+1
        digrams_all_hist[(t[1], t[2])]=digrams_all_hist.get((t[1], t[2]),0)+1
    
    # writing trigrams
    with open("output/trigram" + suffix + ".txt", "w") as outfile_trigrams:
        for t in trigrams_all_hist:
            plus_one=(t[0]+1, t[1]+1, t[2]+1)
            outfile_trigrams.write('%d %d %d '%(plus_one)+str(trigrams_all_hist[t])+'\n')

    # writing diagrams
    memery_nodes=[]
    
    # writing mems of memory nodes
    with open(get_planted_partition_filename(N, om, nc, E, mu, r), "w") as outfile_mem_coms:
        state_ID = 0
        states = np.zeros(N * N).astype(int).reshape(N, N)
        for t, rc in pairs_coms.items():
            state_ID += 1
            outfile_mem_coms.write('{' + str(t[0] + 1) + '}_' + str(t[1] + 1) + ' ' + str(state_ID) + ' ' + str(rc) + '\n')
            states[t[0]][t[1]] = state_ID
    
    for t in trigrams_all_hist:
        if states[t[0]][t[1]] == 0:
            state_ID += 1
            states[t[0]][t[1]] = state_ID
        if states[t[1]][t[2]] == 0:
            state_ID += 1
            states[t[1]][t[2]] = state_ID
    
    # write the network to statefile
    with open("output/network" + suffix + ".net", "w") as statefile:

        # first, write the physical nodes
        statefile.write(f"*Vertices {N}\n")

        for node_ID in range(1, N + 1):
            statefile.write(f"""{node_ID} "{node_ID}"\n""")
        
        # then write the state nodes
        statefile.write("*States\n")
        
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                node_ID = j
                state_ID = states[i - 1][j - 1]
                if state_ID > 0:
                    state_label = f"{{{i}}}_{j}"
                    statefile.write(f"""{state_ID} {node_ID} "{state_label}"\n""")
        
        # finally, write the edges
        statefile.write("*Links\n")
        
        for t in trigrams_all_hist:
            statefile.write(str(states[t[0]][t[1]]) + ' ' + str(states[t[1]][t[2]]) + ' ' + str(trigrams_all_hist[t]) + '\n')
    
    # memory links, memory nodes
    return len(trigrams_all_hist), len(set(memery_nodes))

# ########################################################################


# def get_state_id(n1: int, n2: int, state_id_offset: int):
#     return n1 * state_id_offset + n2


# def read_trigram(filename: str, verbose: bool = True, state_id_offset=None):
#     context = "3grams"
#     G = nx.DiGraph()
#     if verbose:
#         print(f"Read 3gram network from file '{filename}'...")
#     physical_names = {}
#     with open(filename, "r") as fp:
#         for line in fp:
#             if line.startswith("#"):
#                 continue
#             if line.startswith("*"):
#                 ll = line.lower()
#                 if ll.startswith("*vertices"):
#                     context = "Vertices"
#                     continue
#                 if ll.startswith("*3grams"):
#                     context = "3grams"
#                     if state_id_offset is None:
#                         n_max = max(physical_names)  # max key
#                         state_id_offset = 10 ** math.ceil(np.log10(n_max + 1))
#                     continue
#                 else:
#                     raise Exception(f"Unknown section type: '{ll}'")
#             if context == "Vertices":
#                 m = re.match(r"(\d+) \"(.+)\"", line)
#                 if m:
#                     [node_id, name] = m.groups()
#                     physical_names[int(node_id)] = name
#                 continue
#             if context == "3grams":
#                 e = line.split()
#                 if len(e) < 3:
#                     raise Exception(f"Can't parse 3gram from line '{line}'")
#                 n1 = int(e[0])
#                 n2 = int(e[1])
#                 n3 = int(e[2])
#                 weight = float(e[3]) if len(e) >= 4 else 1.0

#                 s1 = get_state_id(n1, n2, state_id_offset)
#                 s2 = get_state_id(n2, n3, state_id_offset)

#                 G.add_node(s1, node_id=n2)
#                 G.add_node(s2, node_id=n3)

#                 G.add_edge(s1, s2, weight=weight)

#     if verbose:
#         print(f"Parsed network: {G}")

#     return G

        
def get_suffix(N, om, nc, E, mu, r):
    '''
        N: No. of Nodes
        om: Number of communities per physical node
        nc: Size of each community
        E: Number of trigrams
        mu: mixing
        r: sample id
    '''

    return f"_N{N}_om{om}_nc{nc}_E{E}_mu{int(mu*100)}_sample{r}"
        
def get_result_suffix(N, nc, E, mu):
    '''
        Same as get_suffix but without om and sample id.
    '''

    return f"_N{N}_nc{nc}_E{E}_mu{int(mu*100)}"
        
def get_network_filename(N, om, nc, E, mu, r):
    return "output/network" + get_suffix(N, om, nc, E, mu, r) + ".net"
        
def get_planted_partition_filename(N, om, nc, E, mu, r):
    return "output/planted_partition" + get_suffix(N, om, nc, E, mu, r) + ".txt"

def read_state_network(filename, verbose=True):
    context = None
    G = nx.DiGraph()
    if verbose:
        print(f"Read state network from file '{filename}'...")
    physical_names = {}
    with open(filename, "r") as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            if line.startswith("*"):
                ll = line.lower()
                if ll.startswith("*vertices"):
                    context = "Vertices"
                    continue
                if ll.startswith("*states"):
                    context = "States"
                    continue
                if ll.startswith("*links"):
                    context = "Links"
                    continue
                else:
                    raise Exception(f"Unknown section type: '{ll}'")
            if context == "Vertices":
                # TODO: Implement to set physical names
                m = re.match(r"(\d+) \"(.+)\"", line)
                if m:
                    [node_id, name] = m.groups()
                    physical_names[int(node_id)] = name
                continue
            if context == "States":
                m = re.match(r"(\d+) (\d+)(?: \"(.+)\")?", line)
                if m:
                    [state_id, node_id, name] = m.groups()
                    if name:
                        node_attributes = dict(name=name)
                    else:
                        node_attributes = {}
                    physical_name = physical_names.get(int(node_id), str(node_id))
                    G.add_node(
                        int(state_id),
                        node_id=int(node_id),
                        physical_name=physical_name,
                        **node_attributes,
                    )
                continue
            if context == "Links":
                parts = line.split()
                if len(parts) < 2:
                    continue
                source = int(parts[0])
                target = int(parts[1])
                weight = float(parts[2]) if len(parts) >= 3 else 1.
                if G.has_edge(source, target):
                    G[source][target]["weight"] += weight
                else:
                    G.add_edge(source, target, weight=weight)
    if verbose:
        print(f"Parsed network: {G}")
    return G

def write_state_network(G, path: str, physical_node_id_attribute: str = "node_id"):
    """
    Write a state network to a Pajek-like file.
    """
    p = Path(path)
    p.parent.mkdir(exist_ok=True)
    node_ids = set(nx.get_node_attributes(G, physical_node_id_attribute).values())

    with open(p, "w") as fp:
        fp.write(f"*Vertices {len(node_ids)}\n")
        for node_id in node_ids:
            fp.write(f'{node_id} "{node_id}"\n')
        fp.write(f"*States {G.number_of_nodes()}\n#state_id node_id\n")
        for state_id, node_id in G.nodes.data(physical_node_id_attribute):
            fp.write(f"{state_id} {node_id}\n")
        fp.write(f"*Links {G.number_of_edges()}\n")
        for u, v, w in G.edges.data("weight", default=1):
            fp.write(f"{u} {v} {w}\n")
    print(f"State network written to {p}")

def read_state_names_from_net(net_file):
    """
    Parse the *States section of a .net file and return
    a dict: state_id -> state_name
    """
    state_names = {}
    in_states_section = False

    with open(net_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("*States"):
                in_states_section = True
                continue

            if line.startswith("*") and in_states_section:
                # End of *States section
                break

            if in_states_section:
                # Expected format:
                # state_id node_id "{state}_node"
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    state_id = int(parts[0])
                    state_name = parts[2].strip('"')
                    state_names[state_id] = state_name

    return state_names


def read_tree_file(tree_file):
    """
    Parse a .tree file and return a list of rows with:
    path, module_id, flow, name, state_id, node_id
    """
    rows = []

    with open(tree_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            path_str = parts[0]
            flow = float(parts[1])
            name = parts[2].strip('"')
            state_id = int(parts[3])
            node_id = int(parts[4])

            # Parse path like "1:3" → (1, 3)
            path = tuple(int(x) for x in path_str.split(":"))
            module_id = path[0]

            rows.append({
                "state_id": state_id,
                "node_id": node_id,
                "name": name,
                "module_id": module_id,
                "path": path,
                "flow": flow
            })

    return rows


def write_tsv(output_file, tree_rows, state_names):
    """
    Write final TSV file with merged information
    """
    header = [
        "state_id",
        "node_id",
        "name",
        "module_id",
        "path",
        "flow",
        "state_name"
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)

        for row in tree_rows:
            state_id = row["state_id"]
            writer.writerow([
                state_id,
                row["node_id"],
                row["name"],
                row["module_id"],
                row["path"],
                row["flow"],
                state_names.get(state_id, "")
            ])


def tree_net_to_tsv(N=128, nc=32, E=25000, mu=0.1, r=1):
    for om in range(2,9):
        suffix = get_suffix(N, om, nc, E, mu, r)
        tree_file = f"output/posterior_unconstrained_network{suffix}_states.tree"
        net_file = f"output/posterior_unconstrained_network{suffix}.net"
        output_tsv = f"output/infomap_partition_{suffix}_posterior_unconstrained.tsv"
        state_names = read_state_names_from_net(net_file)
        tree_rows = read_tree_file(tree_file)
        write_tsv(output_tsv, tree_rows, state_names)


def find_communities_simple(
    G: nx.Graph | str,
    silent=True,
    initial_partition=None,
    phys_id="node_id",
    store_json_output=False,
    print_result=True,
    **infomap_args
):
    im = infomap.Infomap(silent=silent, **infomap_args)

    have_file_input = isinstance(G, str)
    if have_file_input:
        im.read_file(G)
    else:
        im.add_networkx_graph(G, phys_id=phys_id)

    im.run(initial_partition=initial_partition)

    if print_result:
        if im.num_levels > 2:
            print(
                f"Found {im.num_levels} levels with {im.num_top_modules} top modules and codelength {im.codelength}\n"
            )
        else:
            print(
                f"Found {im.num_top_modules} modules with codelength {im.codelength}\n"
            )

    if have_file_input:
        is_state_network = im.have_memory
    else:
        phys_ids = dict(G.nodes.data(phys_id)) if phys_id is not None else dict()
        is_state_network = None not in phys_ids.values()

        if store_json_output:
            # with tempfile.TemporaryDirectory() as tmp:
            #     json_filename = Path(tmp).joinpath('infomap_output.json')
            json_filename = "output/temp/infomap_output.json"
            im.write_json(json_filename)
            with open(json_filename, "r") as fp:
                G.graph["output"] = json.load(fp)

        G.graph["N"] = im.num_nodes
        G.graph["E"] = im.num_links
        G.graph["num_top_modules"] = im.num_top_modules
        G.graph["effective_num_top_modules"] = im.get_effective_num_modules(
            depth_level=1
        )
        G.graph["L"] = im.codelength
        G.graph["L_ind"] = im.index_codelength
        G.graph["L_mod"] = im.module_codelength
        G.graph["L_0"] = im.one_level_codelength
        G.graph["savings"] = im.relative_codelength_savings
        G.graph["max_depth"] = im.max_depth
        G.graph["modules"] = pd.DataFrame(
            im.get_multilevel_modules(states=is_state_network)
        ).T
        G.graph["effective_num_modules_per_level"] = [
            im.get_effective_num_modules(lvl) for lvl in range(1, im.max_depth)
        ]
        # G.graph["effective_num_nodes"] = calc_effective_number_of_nodes(G)
        # G.graph['entropy_rate'] = calc_entropy_rate(G)

    # return im.get_dataframe(
    #     ["state_id", "node_id", "name", "module_id", "path", "flow"]
    # ).set_index("state_id")

    # if output_file is not None:
        # im.write_state_network(output_file, states=True)

    df = im.get_dataframe(
        ["state_id", "node_id", "name", "module_id", "path", "flow"]
    )
    # return df.set_index("state_id")

    # Add state name
    df_name = pd.DataFrame([[state_id, d["name"]] for state_id, d in G.nodes.data()], columns=["state_id", "state_name"])
    df_merged = pd.merge(df, df_name, left_on="state_id", right_on="state_id")
    return df_merged.set_index("state_id")
    

def get_reference_partition():
    return pd.read_csv("output/planted_communities.txt", sep=" ", names=["state_name", "state_id", "module_id"])

def compute_ami(df_ref, df_test):
    """
    Assume both has state_name and module_id columns, merge on state_name
    """
    df = pd.merge(df_ref, df_test, left_on="state_name", right_on="state_name", suffixes=("_ref", "_test"), how="inner")
    partition1 = df["module_id_ref"]
    partition2 = df["module_id_test"]
    ami = adjusted_mutual_info_score(partition1, partition2)
    return ami


def create_synthetic_network(N=32, om=2, nc=8, E=3000, mu=0.1, r=1):
    # No. of Nodes
    # N=32
    ## No. of communities
    #M=8
    # Size of each community
    # nc=8
    # Number of trigrams
    # E=25000
    # E=3000
    ## mixing
    #mu=0.1
    # for om in range(1, 9):
    # for mu in np.arange(0.1, 0.4, 0.1):
    # for r in range(1, 11):
    # om = 2
    # mu = 0.1
    # r = 1
    build_syn_network(N, om, nc, E, mu, r)

    
def run_posterior_physical_infomap(N=32, om=2, nc=8, E=3000, mu=0.1, num_samples=1, num_infomap_samples=1):
    """
    
    """
    if type(om) is list:
        om_list = om
    else:
        om_list = [om]

    for om in om_list:
        for r in range(1, num_samples+1):
            suffix = get_suffix(N, om, nc, E, mu, r)

            # 1. Create synthetic network (also creates a reference partition with planted communities)
            build_syn_network(N, om, nc, E, mu, r)

            network_filename = get_network_filename(N, om, nc, E, mu, r)

            posterior_network_filename = f"output/posterior_physical_network{suffix}.net"
            # 2. Create posterior network
            create_posterior_original(network_filename, posterior_network_filename)

            # 3. Run Infomap
            G_states = read_state_network(posterior_network_filename)
            
            # Get the number of physical nodes
            N = len(set([G_states.nodes[n]["node_id"] for n in G_states.nodes]))

            # Correct the regularization strength for the state network
            regularization_strength = 1/N

            for seed in range(1, num_infomap_samples+1):
                result = find_communities_simple(G_states, regularized=False, two_level=True, regularization_strength=regularization_strength, seed=seed)
                result.to_csv(f"output/infomap_partition_{suffix}_posterior_physical.tsv", sep="\t")
                M = result["module_id"].nunique()
                effective_num_modules = G_states.graph["effective_num_top_modules"]
                codelength = G_states.graph["L"]
                savings = G_states.graph["savings"]
                
                # 4. Calculate AMI
                ref = pd.read_csv(get_planted_partition_filename(N, om, nc, E, mu, r), sep=" ", names=["state_name", "state_id", "module_id"])
                ami = compute_ami(ref, result)
                
                data = {
                    "N": N,
                    "om": om,
                    "nc": nc,
                    "E": E,
                    "mu": mu,
                    "r": r,
                    "seed": seed,
                    "num_modules": M,
                    "effective_num_modules": effective_num_modules,
                    "codelength": codelength,
                    "one_level_codelength": G_states.graph["L_0"],
                    "savings": savings,
                    "ami": ami,
                }

                # 5. Append result to file
                result_filename = f"output/result_posterior_physical_{get_result_suffix(N, nc, E, mu)}.tsv"
                with open(result_filename, "a") as fp:
                    if fp.tell() == 0:
                        # Write header if the file is empty
                        fp.write("\t".join(data.keys()) + "\n")
                    fp.write("\t".join(map(str, data.values())) + "\n")
                print(f"Found {M} modules with {codelength=:.3f} and {ami=:.3f}")
    
def run_posterior_unconstrained_infomap(N=32, om=2, nc=8, E=3000, mu=0.1, num_samples=1, num_infomap_samples=1):
    """
    
    """
    if type(om) is list:
        om_list = om
    else:
        om_list = [om]

    for om in om_list:
        for r in range(1, num_samples+1):
            suffix = get_suffix(N, om, nc, E, mu, r)

            # 1. Create synthetic network (also creates a reference partition with planted communities)
            build_syn_network(N, om, nc, E, mu, r)

            network_filename = get_network_filename(N, om, nc, E, mu, r)

            posterior_network_filename = f"output/posterior_unconstrained_network{suffix}.net"
            # 2. Create posterior network
            create_posterior_2b(network_filename, posterior_network_filename)

            # 3. Run Infomap
            G_states = read_state_network(posterior_network_filename)
            
            # Get the number of physical nodes
            N = len(set([G_states.nodes[n]["node_id"] for n in G_states.nodes]))

            # Correct the regularization strength for the state network
            regularization_strength = 1/N

            for seed in range(1, num_infomap_samples+1):
                result = find_communities_simple(G_states, regularized=False, two_level=True, regularization_strength=regularization_strength, seed=seed)
                result.to_csv(f"output/infomap_partition_{suffix}_posterior_unconstrained.tsv", sep="\t")
                M = result["module_id"].nunique()
                effective_num_modules = G_states.graph["effective_num_top_modules"]
                codelength = G_states.graph["L"]
                savings = G_states.graph["savings"]
                
                # 4. Calculate AMI
                ref = pd.read_csv(get_planted_partition_filename(N, om, nc, E, mu, r), sep=" ", names=["state_name", "state_id", "module_id"])
                ami = compute_ami(ref, result)
                
                data = {
                    "N": N,
                    "om": om,
                    "nc": nc,
                    "E": E,
                    "mu": mu,
                    "r": r,
                    "seed": seed,
                    "num_modules": M,
                    "effective_num_modules": effective_num_modules,
                    "codelength": codelength,
                    "one_level_codelength": G_states.graph["L_0"],
                    "savings": savings,
                    "ami": ami,
                }

                # 5. Append result to file
                result_filename = f"output/result_posterior_unconstrained_{get_result_suffix(N, nc, E, mu)}.tsv"
                with open(result_filename, "a") as fp:
                    if fp.tell() == 0:
                        # Write header if the file is empty
                        fp.write("\t".join(data.keys()) + "\n")
                    fp.write("\t".join(map(str, data.values())) + "\n")
                print(f"Found {M} modules with {codelength=:.3f} and {ami=:.3f}")

def run_regularized_infomap(N=32, om=2, nc=8, E=3000, mu=0.1, num_samples=1, num_infomap_samples=1, regularization_strength_multiplier=1):
    if type(om) is list:
        om_list = om
    else:
        om_list = [om]

    for om in om_list:
        for r in range(1, num_samples+1):
            suffix = get_suffix(N, om, nc, E, mu, r)

            # 1. Create synthetic network (also creates a reference partition with planted communities)
            build_syn_network(N, om, nc, E, mu, r)

            network_filename = get_network_filename(N, om, nc, E, mu, r)

            # 3. Run Infomap
            G_states = read_state_network(network_filename)
            
            # Get the number of physical nodes
            N = len(set([G_states.nodes[n]["node_id"] for n in G_states.nodes]))

            # Correct the regularization strength for the state network
            regularization_strength = 1/N * regularization_strength_multiplier

            for seed in range(1, num_infomap_samples+1):
                result = find_communities_simple(G_states, regularized=True, two_level=True, regularization_strength=regularization_strength, seed=seed)
                result.to_csv(f"output/infomap_partition_{suffix}_regularized.tsv", sep="\t")
                M = result["module_id"].nunique()
                effective_num_modules = G_states.graph["effective_num_top_modules"]
                codelength = G_states.graph["L"]
                savings = G_states.graph["savings"]
                
                # 4. Calculate AMI
                ref = pd.read_csv(get_planted_partition_filename(N, om, nc, E, mu, r), sep=" ", names=["state_name", "state_id", "module_id"])
                ami = compute_ami(ref, result)

                data = {
                    "N": N,
                    "om": om,
                    "nc": nc,
                    "E": E,
                    "mu": mu,
                    "r": r,
                    "seed": seed,
                    "regularization_strength_multiplier": regularization_strength_multiplier,
                    "num_modules": M,
                    "effective_num_modules": effective_num_modules,
                    "codelength": codelength,
                    "one_level_codelength": G_states.graph["L_0"],
                    "savings": savings,
                    "ami": ami,
                }
                
                # 5. Append result to file
                result_filename = f"output/result_regularized_{get_result_suffix(N, nc, E, mu)}.tsv"
                with open(result_filename, "a") as fp:
                    if fp.tell() == 0:
                        # Write header if the file is empty
                        fp.write("\t".join(data.keys()) + "\n")
                    fp.write("\t".join(map(str, data.values())) + "\n")
                print(f"Found {M} modules with {codelength=:.3f} and {ami=:.3f}")

def run_standard_infomap(N=32, om=2, nc=8, E=3000, mu=0.1, num_samples=1, num_infomap_samples=1):
    if type(om) is list:
        om_list = om
    else:
        om_list = [om]

    for om in om_list:
        for r in range(1, num_samples+1):
            suffix = get_suffix(N, om, nc, E, mu, r)

            # 1. Create synthetic network (also creates a reference partition with planted communities)
            build_syn_network(N, om, nc, E, mu, r)

            network_filename = get_network_filename(N, om, nc, E, mu, r)

            # 3. Run Infomap
            G_states = read_state_network(network_filename)
            
            # Get the number of physical nodes
            N = len(set([G_states.nodes[n]["node_id"] for n in G_states.nodes]))

            for seed in range(1, num_infomap_samples+1):
                result = find_communities_simple(G_states, regularized=False, two_level=True, seed=seed)
                result.to_csv(f"output/infomap_partition_{suffix}_standard.tsv", sep="\t")
                M = result["module_id"].nunique()
                effective_num_modules = G_states.graph["effective_num_top_modules"]
                codelength = G_states.graph["L"]
                savings = G_states.graph["savings"]
                
                # 4. Calculate AMI
                ref = pd.read_csv(get_planted_partition_filename(N, om, nc, E, mu, r), sep=" ", names=["state_name", "state_id", "module_id"])
                ami = compute_ami(ref, result)

                data = {
                    "N": N,
                    "om": om,
                    "nc": nc,
                    "E": E,
                    "mu": mu,
                    "r": r,
                    "seed": seed,
                    "num_modules": M,
                    "effective_num_modules": effective_num_modules,
                    "codelength": codelength,
                    "one_level_codelength": G_states.graph["L_0"],
                    "savings": savings,
                    "ami": ami,
                }
                
                # 5. Append result to file
                result_filename = f"output/result_standard_{get_result_suffix(N, nc, E, mu)}.tsv"
                with open(result_filename, "a") as fp:
                    if fp.tell() == 0:
                        # Write header if the file is empty
                        fp.write("\t".join(data.keys()) + "\n")
                    fp.write("\t".join(map(str, data.values())) + "\n")
                print(f"Found {M} modules with {codelength=:.3f} and {ami=:.3f}")

def run_reference_infomap(N=32, om=2, nc=8, E=3000, mu=0.1, num_samples=1, num_infomap_samples=1):
    if type(om) is list:
        om_list = om
    else:
        om_list = [om]

    for om in om_list:
        for r in range(1, num_samples+1):
            suffix = get_suffix(N, om, nc, E, mu, r)

            # 1. Create synthetic network (also creates a reference partition with planted communities)
            build_syn_network(N, om, nc, E, mu, r)

            network_filename = get_network_filename(N, om, nc, E, mu, r)
            ref = pd.read_csv(get_planted_partition_filename(N, om, nc, E, mu, r), sep=" ", names=["state_name", "state_id", "module_id"])
            planted_partition = ref.set_index("state_id")["module_id"].to_dict()

            # 3. Run Infomap
            G_states = read_state_network(network_filename)
            
            # Get the number of physical nodes
            N = len(set([G_states.nodes[n]["node_id"] for n in G_states.nodes]))

            for seed in range(1, 2):
                result = find_communities_simple(G_states, regularized=False, two_level=True, seed=seed, initial_partition=planted_partition, no_infomap=True)
                result.to_csv(f"output/infomap_partition_{suffix}_reference.tsv", sep="\t")
                M = result["module_id"].nunique()
                effective_num_modules = G_states.graph["effective_num_top_modules"]
                codelength = G_states.graph["L"]
                savings = G_states.graph["savings"]
                
                # 4. Calculate AMI
                ami = compute_ami(ref, result)

                data = {
                    "N": N,
                    "om": om,
                    "nc": nc,
                    "E": E,
                    "mu": mu,
                    "r": r,
                    "seed": seed,
                    "num_modules": M,
                    "effective_num_modules": effective_num_modules,
                    "codelength": codelength,
                    "one_level_codelength": G_states.graph["L_0"],
                    "savings": savings,
                    "ami": ami,
                }
                
                # 5. Append result to file
                result_filename = f"output/result_reference_{get_result_suffix(N, nc, E, mu)}.tsv"
                with open(result_filename, "a") as fp:
                    if fp.tell() == 0:
                        # Write header if the file is empty
                        fp.write("\t".join(data.keys()) + "\n")
                    fp.write("\t".join(map(str, data.values())) + "\n")
                print(f"Found {M} modules with {codelength=:.3f} and {ami=:.3f}")

def run_reference_regularized_infomap(N=32, om=2, nc=8, E=3000, mu=0.1, num_samples=1, num_infomap_samples=1, regularization_strength_multiplier=1):
    if type(om) is list:
        om_list = om
    else:
        om_list = [om]

    for om in om_list:
        for r in range(1, num_samples+1):
            suffix = get_suffix(N, om, nc, E, mu, r)

            # 1. Create synthetic network (also creates a reference partition with planted communities)
            build_syn_network(N, om, nc, E, mu, r)

            network_filename = get_network_filename(N, om, nc, E, mu, r)
            ref = pd.read_csv(get_planted_partition_filename(N, om, nc, E, mu, r), sep=" ", names=["state_name", "state_id", "module_id"])
            planted_partition = ref.set_index("state_id")["module_id"].to_dict()

            # 3. Run Infomap
            G_states = read_state_network(network_filename)
            
            # Get the number of physical nodes
            N = len(set([G_states.nodes[n]["node_id"] for n in G_states.nodes]))

            # Correct the regularization strength for the state network
            regularization_strength = 1/N * regularization_strength_multiplier

            for seed in range(1, 2):
                result = find_communities_simple(G_states, regularized=True, two_level=True, regularization_strength=regularization_strength, seed=seed, initial_partition=planted_partition, no_infomap=True)
                result.to_csv(f"output/infomap_partition_{suffix}_reference.tsv", sep="\t")
                M = result["module_id"].nunique()
                effective_num_modules = G_states.graph["effective_num_top_modules"]
                codelength = G_states.graph["L"]
                savings = G_states.graph["savings"]
                
                # 4. Calculate AMI
                ami = compute_ami(ref, result)

                data = {
                    "N": N,
                    "om": om,
                    "nc": nc,
                    "E": E,
                    "mu": mu,
                    "r": r,
                    "seed": seed,
                    "num_modules": M,
                    "effective_num_modules": effective_num_modules,
                    "codelength": codelength,
                    "one_level_codelength": G_states.graph["L_0"],
                    "savings": savings,
                    "ami": ami,
                }
                
                # 5. Append result to file
                result_filename = f"output/result_reference_regularized_{get_result_suffix(N, nc, E, mu)}.tsv"
                with open(result_filename, "a") as fp:
                    if fp.tell() == 0:
                        # Write header if the file is empty
                        fp.write("\t".join(data.keys()) + "\n")
                    fp.write("\t".join(map(str, data.values())) + "\n")
                print(f"Found {M} modules with {codelength=:.3f} and {ami=:.3f}")




def run_regularized_infomap_test_32():
    N=32
    om=list(range(1,9))
    nc=8
    E=3000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    regularization_strength_multiplier=1

    run_regularized_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples, regularization_strength_multiplier)

    
def run_posterior_physical_infomap_test_32():
    N=32
    om=list(range(1,9))
    nc=8
    E=3000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_posterior_physical_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)
    
def run_posterior_unconstrained_infomap_test_32():
    N=32
    om=list(range(1,9))
    nc=8
    E=3000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_posterior_unconstrained_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)
    
def run_standard_infomap_test_32():
    N=32
    om=list(range(1,9))
    nc=8
    E=3000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_standard_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)
    
def run_reference_infomap_test_32():
    N=32
    om=list(range(1,9))
    nc=8
    E=3000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_reference_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)
    
def run_reference_regularized_infomap_test_32():
    N=32
    om=list(range(1,9))
    nc=8
    E=3000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_reference_regularized_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)


def run_regularized_infomap_test_128():
    N=128
    om=list(range(1,9))
    nc=32
    E=25000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    regularization_strength_multiplier=1

    run_regularized_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples, regularization_strength_multiplier)

    
def run_posterior_physical_infomap_test_128():
    N=128
    om=list(range(1,9))
    nc=32
    E=25000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_posterior_physical_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)
    
def run_posterior_unconstrained_infomap_test_128():
    N=128
    om=list(range(1,9))
    nc=32
    E=25000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_posterior_unconstrained_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)
    
def run_standard_infomap_test_128():
    N=128
    om=list(range(1,9))
    nc=32
    E=25000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_standard_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)
    
def run_reference_infomap_test_128():
    N=128
    om=list(range(1,9))
    nc=32
    E=25000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_reference_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)

def run_reference_regularized_infomap_test_128():
    N=128
    om=list(range(1,9))
    nc=32
    E=25000
    mu=0.1
    num_samples=1
    num_infomap_samples=1

    run_reference_regularized_infomap(N, om, nc, E, mu, num_samples, num_infomap_samples)



def run_all_test_32():
    print("run_regularized_infomap_test_32()...")
    run_regularized_infomap_test_32()
    print("run_posterior_physical_infomap_test_32()...")
    run_posterior_physical_infomap_test_32()
    print("run_posterior_unconstrained_infomap_test_32()...")
    run_posterior_unconstrained_infomap_test_32()
    print("run_standard_infomap_test_32()...")
    run_standard_infomap_test_32()
    print("run_reference_infomap_test_32()...")
    run_reference_infomap_test_32()
    print("run_reference_regularized_infomap_test_32()...")
    run_reference_regularized_infomap_test_32()


def run_all_test_128():
    run_regularized_infomap_test_128()
    run_posterior_physical_infomap_test_128()
    run_posterior_unconstrained_infomap_test_128()
    run_standard_infomap_test_128()
    run_reference_infomap_test_128()
    run_reference_regularized_infomap_test_128()
    


def plot_result(N=32, nc=8, E=3000, mu=0.1):
    suffix = get_result_suffix(N=N, nc=nc, E=E, mu=mu)
    df_standard = pd.read_csv(f"output/result_standard_{suffix}.tsv", sep="\t")
    df_standard["method"] = "standard"

    df_regularized = pd.read_csv(f"output/result_regularized_{suffix}.tsv", sep="\t")
    df_regularized["method"] = "regularized"
    
    df_posterior_physical = pd.read_csv(f"output/result_posterior_physical_{suffix}.tsv", sep="\t")
    df_posterior_physical["method"] = "posterior_physical"
    
    df_posterior_unconstrained = pd.read_csv(f"output/result_posterior_unconstrained_{suffix}.tsv", sep="\t")
    df_posterior_unconstrained["method"] = "posterior_unconstrained"
    
    df_reference = pd.read_csv(f"output/result_reference_{suffix}.tsv", sep="\t")
    df_reference["method"] = "reference"
    
    df_reference_regularized = pd.read_csv(f"output/result_reference_regularized_{suffix}.tsv", sep="\t")
    df_reference_regularized["method"] = "reference_regularized"

    df = pd.concat([df_standard, df_regularized, df_posterior_physical, df_posterior_unconstrained, df_reference, df_reference_regularized])

    df = df[(df["N"] == N) | (df["nc"] == nc) | (df["E"] == E)]

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

    # y_targets = ["num_modules", "effective_num_modules", "ami", "one_level_codelength"]
    
    # for i, ax in enumerate(axes.flat):
    #     sns.lineplot(df, y=y_targets[i], x="om", hue="method", ax=ax)

    y_targets = ["num_modules", "effective_num_modules", "ami", "one_level_codelength", "savings", "codelength"]
    num_targets = len(y_targets)

    ncols = 2
    nrows = (num_targets + ncols - 1) // ncols
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    axes = axes.flat
    
    for ax, y in zip(axes, y_targets):
        sns.lineplot(df, y=y, x="om", hue="method", style="method", dashes=True, markers=True, ax=ax)
    
    # Remove unused axes (if any)
    for ax in axes[len(y_targets):]:
        ax.remove()



if __name__=='__main__':


    # No. of Nodes
    N=32
    ## No. of communities
    #M=8
    # Size of each community
    nc=8
    # Number of trigrams
    # E=25000
    E=3000
    ## mixing
    #mu=0.1
    # for om in range(1, 9):
    # for mu in np.arange(0.1, 0.4, 0.1):
    # for r in range(1, 11):
    om = 2
    mu = 0.1
    r = 1
    build_syn_network(N, om, nc, E, mu, r)
    


