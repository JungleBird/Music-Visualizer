import numpy as np
from numpy.linalg import norm
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.animation
from rbm import RBM
from audio_parser import Audio_Parser
from graph_node import Graph_Node

import networkx as nx
from functools import partial
import random
import math
import time
    

def random_initialization(A,rank):
    number_of_documents = A.shape[0]
    number_of_terms = A.shape[1]
    W = np.random.uniform(1,2,(number_of_documents,rank))
    H = np.random.uniform(1,2,(rank,number_of_terms))
    return W,H
                          

def nndsvd_initialization(A,rank):
    #np.nan_to_num(A,posinf=0,neginf=0)

    u,s,v=np.linalg.svd(A,full_matrices=False)
    v=v.T
    w=np.zeros((A.shape[0],rank))
    h=np.zeros((rank,A.shape[1]))

    w[:,0]=np.sqrt(s[0])*np.abs(u[:,0])
    h[0,:]=np.sqrt(s[0])*np.abs(v[:,0].T)

    for i in range(1,rank):
        
        ui=u[:,i]
        vi=v[:,i]
        ui_pos=(ui>=0)*ui
        ui_neg=(ui<0)*-ui
        vi_pos=(vi>=0)*vi
        vi_neg=(vi<0)*-vi
        
        ui_pos_norm=np.linalg.norm(ui_pos,2)
        ui_neg_norm=np.linalg.norm(ui_neg,2)
        vi_pos_norm=np.linalg.norm(vi_pos,2)
        vi_neg_norm=np.linalg.norm(vi_neg,2)
        
        norm_pos=ui_pos_norm*vi_pos_norm
        norm_neg=ui_neg_norm*vi_neg_norm
        
        if norm_pos>=norm_neg:
            w[:,i]=np.sqrt(s[i]*norm_pos)/ui_pos_norm*ui_pos
            h[i,:]=np.sqrt(s[i]*norm_pos)/vi_pos_norm*vi_pos.T
        else:
            w[:,i]=np.sqrt(s[i]*norm_neg)/ui_neg_norm*ui_neg
            h[i,:]=np.sqrt(s[i]*norm_neg)/vi_neg_norm*vi_neg.T

    return w,h
    
def mu_method(A,k,max_iter,init_mode='random'):
    
    if init_mode == 'random':
        W ,H = random_initialization(A,k)
    elif init_mode == 'nndsvd':
        W ,H = nndsvd_initialization(A,k) 
    norms = []
    e = 1.0e-10
    for n in range(max_iter):
        # Update H
        W_TA = W.T@A
        W_TWH = W.T@W@H+e
        for i in range(np.size(H, 0)):
            for j in range(np.size(H, 1)):
                H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]
        # Update W
        AH_T = A@H.T
        WHH_T =  W@H@H.T+ e

        for i in range(np.size(W, 0)):
            for j in range(np.size(W, 1)):
                W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]

        norm = np.linalg.norm(A - W@H, 'fro')
        norms.append(norm)
    return np.matrix(W), np.matrix(H), norms

def proportion_matrix(data, array_size, downsample = None):
    dataSize = array_size//downsample if downsample and downsample > 1 else array_size
    proportionMatrixData = []
    gaussianFilterData = []
    edgeLineData = []
    sectionSteps = []
    
    #data = gaussian_filter1d(data, 1.5)
    
    for i in range(len(data)):
        feature_data = ap.downsample(data[i], downsample) if downsample and downsample > 1 else data[i]
        
        maxRowIndex = np.argmax(feature_data)
        proportionMatrix = np.zeros((dataSize,dataSize))

        maxVal = 0
        for n in range(dataSize):
            proportionH = feature_data[maxRowIndex]/(feature_data[n]+1)
            maxVal = max(maxVal, proportionH)
            proportionMatrix[maxRowIndex][n] = proportionH

        #peaks, _ = signal.find_peaks(proportionMatrix[maxRowIndex], distance=16)
        peak = np.argmax(proportionMatrix[maxRowIndex])

        #for p in peaks:
        for m in range(dataSize):
            proportionV = feature_data[m]/(feature_data[peak]+1)
            maxVal = max(maxVal, proportionV)
            proportionMatrix[m][peak] = proportionV
        
        proportionMatrix /= maxVal
        proportionMatrixData.append(proportionMatrix)


        edgeLine = proportionMatrix.T[peak]
        edgeLine = gaussian_filter1d(edgeLine, 1) #reduces amount of nodes created by 1/3
        edgeLineData.append(edgeLine)
        #edgeLineData.append(edgeLine)

        gaussianData = proportionMatrix[maxRowIndex]
        gaussianData = gaussian_filter1d(gaussianData, 1) #reduces amount of nodes created by 1/3
        gaussianFilterData.append(gaussianData)

    return proportionMatrixData, gaussianFilterData, edgeLineData

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def recursive_remove_duplicate(node_array):
    if len(node_array) == 0:
        return []

    if len(node_array) == 1:
        return [node_array[0]]

    nodes = [node_array[0]]
    node_list = node_array[1:]
    next_list = []
    
    for n in node_list:
        if nodes[0].node_distance(n) > 0.0085:
            next_list.append(n)

    return nodes + recursive_remove_duplicate(next_list)
    
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''

    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        '''
        if pos is None:

            pos = {root:(vert_loc*math.cos(xcenter),vert_loc*math.sin(xcenter))}
            #pos = {root:(xcenter,vert_loc)}

        else:

            pos[root] = (vert_loc*math.cos(xcenter),vert_loc*math.sin(xcenter))
            #pos[root] = (xcenter, vert_loc)

        children = list(G.neighbors(root))
        
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  

        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2

            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


filename = 'C:/Users/Escobar/Documents/Audacity/Canon in D - Pachelbel.wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Madonna - Like A Prayer (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Desire - Under Your Spell (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Nightcore - Careless (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Nightcore - Sick Of It All (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Darker Than Black Ending 1.wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Blink 182 - All Of This (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Billie Eilish - Bad Guy (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Green Day - Whatsername (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Boogiepop - Boogiepop and Others (Mono).wav'

modifier = 2
chunk = 1024*modifier
ap = Audio_Parser(filename, chunk)

arr = None
testVis = None
testFeatures = None

zero_vector = np.zeros(128)
zero_matrix = np.zeros((128,128))

zero_vector[-1] = 1
zero_matrix[-1][-1] = 1

za0, zb0, zc0 = proportion_matrix([zero_vector], 128)
zero_node = Graph_Node([za0[0], zb0[0], zc0[0]], 'Zero')

reserve_nodes = [zero_node]

net_nodes = 0
num_features = 4
basis_samples = 16

animfig, animax = plt.subplots(1,1)
plt.axis('off')
animfig.patch.set_facecolor('black')
graph=nx.Graph()
color_map = []
size_map = []
color_dict = {}
nextindex = 1
pos = None

def init_radial_tree(Graph):
    global pos
    Graph.add_node(zero_node.label)
    color_map.append('green')
    color_dict[zero_node.label] = 0
    size_map.append(50)

    pos = hierarchy_pos(Graph, zero_node.label, width = 2*math.pi, xcenter=0)

#for i in range(ap.num_frames//ap.chunk_size):
def play_sound(i):

    global arr, graph, color_map, size_map, color_dict, animax, net_nodes, nextindex, basis_samples, pos

    #start = time.time()
    data = ap.play_chunk()   

    if data is None or len(data) < ap.chunk_size:
        return None

    fft, fftx = ap.audio_fft(data, ap.sample_rate, ap.chunk_size)
    fft_reduced_0 = ap.downsample(fft[:148], 2)
    fft_reduced_1 = ap.downsample(fft[148:292], 4)
    fft_reduced_2 = ap.downsample(fft[292:436], 8)

    fft_reduced = np.append(fft_reduced_0, fft_reduced_1)
    fft_reduced = np.append(fft_reduced, fft_reduced_2)

    fft_reduced[fft_reduced < ap.chunk_size] = 0
    fft_reduced = fft_reduced//ap.chunk_size
    
    rbm_fft_reduced = np.log1p(fft_reduced)

    if arr is None or len(arr) == 0:
        arr = np.array([fft_reduced])
    else:
        if arr.shape[0] > basis_samples-1:

            arr[:-1] = arr[1:]
            arr[-1] = fft_reduced
        else:
            arr = np.append(arr, [fft_reduced], axis=0)

        if arr.shape[0] > basis_samples-1:

            w, h, n = mu_method(arr,num_features,20,'random')

            features = np.where(h > 0, h, 0)
            basis = np.where(w > 0, w, 0)

            net_nodes += num_features
            a, b, c = proportion_matrix(features, 128)
        
            #TODO: add node size param based on basis/temporal values
            feature_nodes = [
                                Graph_Node([a[0], b[0], c[0], basis[0]], f'Bl{i}'), 
                                Graph_Node([a[1], b[1], c[1], basis[1]], f'Gr{i}'), 
                                Graph_Node([a[2], b[2], c[2], basis[2]], f'Or{i}'), 
                                Graph_Node([a[3], b[3], c[3], basis[3]], f'Re{i}')
                            ]

            color_labels = ['Bl', 'Gr', 'Or', 'Re']

            pruned_nodes = recursive_remove_duplicate(feature_nodes)

            collect_nodes = []
            new_nodes = []

            for free_node in pruned_nodes:
                fixed_node = reserve_nodes[0]
                visited_set = set()
                fixed_node, distance, steps, similarity, num_neighbors, visited_nodes, travel_path, visit_path = fixed_node.compare_neighbors(free_node, visited_set)

                sim_score = distance*similarity
                animax.cla()

                energy = 0
                if len(fixed_node.data) > 3:
                    energy = free_node.data[3][-1]

                #if np.isnan(distance) or (distance < 0.0125) or (sim_score < 0.0165):
                if np.isnan(distance) or (distance < 0.0085) or (sim_score < 0.0135):

                    collect_nodes.append([fixed_node.label, energy])
                    #collect all free nodes and do this in batches at end of the for loop
                    continue

                fixed_node.add_neighbor(free_node, distance)
                free_node.add_neighbor(fixed_node, distance)
                reserve_nodes.append(free_node)

                print(free_node.label, ':', fixed_node.label, '\tdist: ', f'{distance:0.2f}', '\tsteps: ', steps, ' \tsim: ', f'{similarity:0.2f}', f'{sim_score:0.4f}')#, energyfft, energymf)

                new_nodes.append(free_node.label)
                graph.add_node(free_node.label)
                graph.add_edge(free_node.label, fixed_node.label)
                color_map.append('red')
                size_map.append(50 + energy*30)
                color_dict[free_node.label] = nextindex
                nextindex += 1

            pos = hierarchy_pos(graph, zero_node.label, width = 2*math.pi, xcenter=0)

            for cnodes, energy in collect_nodes:
                ind = color_dict[cnodes]

                if ind > 0:
                    color_map[ind] = 'orange'

                size_map[ind] = 50 + energy*30
            
            animax.cla()
            
            nx.draw(graph, pos=pos, node_size=size_map, ax=animax, node_color=color_map, edge_color='white')
        
            for cnodes, energy in collect_nodes:
                ind = color_dict[cnodes]
                
                if ind > 0:
                    color_map[ind] = 'blue'
                
            
            size_map = list(map(lambda x: max(50, x-20), size_map))
            arr = arr[8:]

    return [animax]

#Sampling Rate: 44.1 Mhz * 512 Samples = 11.61 ms 
upd = partial(play_sound)
ani = matplotlib.animation.FuncAnimation(animfig,upd,init_func=init_radial_tree(graph), repeat=False,interval=11.6,frames=ap.num_frames//ap.chunk_size, blit=True)
plt.show()
