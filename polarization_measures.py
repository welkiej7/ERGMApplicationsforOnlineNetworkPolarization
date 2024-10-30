import numpy as np 
import time
import psutil
import warnings
import igraph as ig
from tqdm import tqdm, trange
from scipy.stats import gaussian_kde, entropy
from math import e
from copy import deepcopy
from math import log
## For benchmark

import seaborn as sns 
import matplotlib.pyplot as plt 
from community_detection import partition_with_metis
import matplotlib.colors as mcolors

def _run_info(function, *args, **kwargs):
    process = psutil.Process()
    start_time = time.perf_counter()
    start_memory = process.memory_info().rss / 1024**2

    result = function(*args,**kwargs)


    end_time = time.perf_counter()
    end_memory = process.memory_info().rss / 1024**2


    print('\x1b[6;30;42m' + f'Finished the Process with {end_memory - start_memory} MB Ram Usage in {end_time - start_time} seconds' + '\x1b[0m')

    return result
def random_walk_controversy_2comms(network:ig.Graph, 
                            cutoff_for_influential_nodes:int, 
                            membership:list, 
                            network_type:str = "retweet",
                            n_iteration:int = 10000,
                            walk_len:int = "default",
                            reverse:bool = True,
                            influential_node_mode = "out",
                            walk_mode = "out"):
    '''
    Directly samples random walks from a network and measures the random walk controversy. If it reaches to an influential node, the random walk will be cut
    this process is repeated n_iterations time. 
    Important Notes: Membership must be a list filled with integers 0 and 1.


    Influential Node Mode: According to which type of (in or out) degree that the influential nodes
    will be selected. 

    Walk Mode: What is the direction of the node while making the node. Remember that there is no
    random placement for the walker so if a walker gets stuck that walk will end there. 


    Suggestions: Influential node mode out and walk mode out is the default version for a directed
    retweet network. Since the number of retweets from a node decides on it being influential. 
    While doing the walks, it must lead to same direction to simulate the affect of diffusion.
    '''

    membership = np.array(membership)
    # Walk's can only happen in the reverse direction of retweets. Hence we are reversing the network.
    if reverse:
        rw_network = ig.Graph(n = len(membership), edges = [(j,i) for (i,j) in network.get_edgelist()], directed = True)
    else:
        rw_network = network
    # Pick a random walker for 500 times from one community and 500 times from another community.
    left_membership = np.argwhere(membership == 0).flatten()
    right_membership = np.argwhere(membership == 1).flatten()
    
    left_influential = np.argwhere(np.array(rw_network.degree(left_membership, mode = influential_node_mode)) >= cutoff_for_influential_nodes).flatten()
    left_influential = left_membership[left_influential]
    right_influential = np.argwhere(np.array(rw_network.degree(right_membership, mode = walk_mode)) >= cutoff_for_influential_nodes).flatten()
    right_influential = right_membership[right_influential]
    LR = 0
    RL = 0
    LL = 0
    RR = 0
    TW = 0

    left_start_nodes = np.random.choice(left_membership, size = int(n_iteration* 0.5), replace= True)
    right_start_nodes = np.random.choice(right_membership, size = int(n_iteration* 0.5), replace= True)

    #Left Members
    for node in tqdm(left_start_nodes, desc = 'Walking from Left'):
        for __ in range(int(round(rw_network.vcount() ** 0.5))):
            possible_walks = rw_network.neighborhood(node, mode = walk_mode)
            possible_walks.pop(0) # Prevent self walks. 
            if len(possible_walks) == 0:
                break
            walk = np.random.choice(possible_walks, size = 1).item()
            if walk in left_influential:
                LL += 1
                TW += 1
                break
            if walk in right_influential:
                LR += 1
                TW += 1
                break 
            
            node = walk
    
    #Right Members
    for node in tqdm(right_start_nodes,'Walking from Right'):
        for __ in range(int(round(rw_network.vcount() ** 0.5))):
            possible_walks = rw_network.neighborhood(node, mode = walk_mode)
            possible_walks.pop(0) # Prevent self walks. 
            if len(possible_walks) == 0:
                break

            walk = np.random.choice(possible_walks, size = 1).item()
            if walk in left_influential:
                RL += 1
                TW += 1
                break
            if walk in right_influential:
                RR += 1
                TW += 1
                break 
            
            node = walk

    return ((LL / (LR + LL)) * (RR / (RL + RR))) - ((RL / (RR + RL)) * (LR / (LL + LR)))
def betweenness_controversy_2comms(network:ig.Graph, membership:list, kernel:str = "Gaussian", pdf_plot = True):

    '''
    Calculates the Kullback Leibler Divergence of the Centralities between edges in across communities and within communities.
    The PDFs of edges are calculated with kernel density estimation.
    

    This one is a bit harsh and computationally expensive since you need to calculate the edge betweenness for each edge and 
    also need to find the edges running in between communities. The most efficient way to do this is to generate an array 
     with binary values. It gets 1 if it is running across communities, 0 otherwise. 
    '''
    #Get all the edges.

    tmp_edge_list = ig.Graph.get_edgelist(network)
    between_edges_info = [False if membership[i] == membership[j] else True for (i,j) in tmp_edge_list]
    edge_btws = network.edge_betweenness() #Calculate edge btwns
    
    ig.plot(network, target = "graph.png", vertex_color = ["white" if i == 1 else "black" for i in membership],edge_color = ["blue" if i else "green" for i in between_edges_info])

    between_edges = [edge_btws[i] for i in range(len(between_edges_info)) if between_edges_info[i]]
    inside_edges = [edge_btws[i] for i in range(len(between_edges_info)) if not between_edges_info[i]]
    
    if kernel == "Gaussian":
        btw_pdf_kernel = gaussian_kde(between_edges, bw_method= "silverman")
        ins_pdf_kernel = gaussian_kde(inside_edges, bw_method= "silverman")
        
        x_values = np.linspace(min(min(between_edges), min(inside_edges)), 
                        max(max(between_edges), max(inside_edges)), 1000)

        pdf_between = btw_pdf_kernel(x_values)
        pdf_inside = ins_pdf_kernel(x_values)
        


        kl_divergence = entropy(pdf_between, pdf_inside)
        score = 1 - (e**-kl_divergence)
        if pdf_plot:

            fig, ax = plt.subplots()
            ax.plot(x_values, pdf_between, label='Between Edges PDF', color = "orange", alpha = 0.6)
            ax.plot(x_values, pdf_inside, label='Inside Edges PDF', color = "blue", alpha = 0.6)
            ax.fill_between(x_values, pdf_between, color='orange', alpha=0.3)
            ax.fill_between(x_values, pdf_inside, color='blue', alpha=0.3)
            ax.set_xlabel('Edge Betweenness')
            ax.set_ylabel('Density')
            ax.set_title(f"PDF Estimation with {kernel} Kernel,\nKL Divergence:{kl_divergence} -- BCC:{score}")
            ax.legend()

            return score, fig
        else:
            return score
    

    # It is also possible to add other kernels in here. Like a scale free kernel
    # for representing the edge betweenness a bit better ?? maybe ?? 
def dipole_moment_2comms(network:ig.Graph, membership:list, convergence_tolerance:float = 1e-5, inf_node_perc:int=5, max_iteration:int = 1e3, include_itself = False, reverse = True, propagation_mode = "in", mode_degree = "out"):
    '''
    Calculates the label propagation and difference between grativity centers in a network. From each community
    inf_node_perc percent influential nodes are selected. They are assigned a value of -1 and 1. Then label propagation
    starts and each node updates their values according to the in_degree edges. If there are no indegree edges
    the node will stay zero.

    @param:ig.Graph network = The network given
    @param:list membership = The membership list of the nodes. 
    @param:float convergence_tolerance = The tolerance value of change to accept an epoch as convergence.
    @param:int inf_node_perc = The percent of the most influential nodes for each community.
    @param:int max_iteration = The number of iterations permitted for the convergence process

    @param:str propagation_mode = How the diffusion will progress, what is the direction of the diffusion? in, out or all.
    @param:str mode_degree = What type of degree should be considered when evaluating the mode_degree.  
    '''
    if reverse:
        network = ig.Graph(n = len(membership), edges = [(j,i) for (i,j) in network.get_edgelist()], directed = True)

    degs = np.array(network.degree(mode = mode_degree))
    left_nodes = [i for i in range(len(membership)) if membership[i] == 1]
    right_nodes = [i for i in range(len(membership)) if membership[i] == 0]

    left_nodes_degs = degs[left_nodes].tolist()
    right_nodes_degs = degs[right_nodes].tolist()

    left_dict = dict([(key,value) for i,(key,value) in enumerate(zip(left_nodes, left_nodes_degs))])
    right_dict = dict([(key,value) for i,(key,value) in enumerate(zip(right_nodes, right_nodes_degs))])

    sorted_left = list(dict(sorted(left_dict.items(), key=lambda item: item[1], reverse= True)).keys())
    sorted_right = list(dict(sorted(right_dict.items(), key=lambda item: item[1], reverse= True)).keys())

    # Calculate the k-th percent. 

    left_node_cut = round((len(sorted_left) * inf_node_perc) / 100)
    right_node_cut = round((len(sorted_right) * inf_node_perc) / 100)

    #Distribute the Opinions give 1 to the left side and -1 to the right side
    left_influential = sorted_left[0:left_node_cut]
    right_influential = sorted_right[0:right_node_cut]

    opinion_list = [1 if i in left_influential else -1 if i in right_influential else 0 for i in range(len(degs))]
    
    ### Start the Label Propagation with Mean
    Start_Point = -888
    prev_opinion_list = deepcopy(opinion_list)
    trange = tqdm(range(int(max_iteration)), colour= "cyan", ascii = "=🌯", desc = "Propagating, Epoch Delta = 0")
    for epoch in trange:
        for node in range(len(degs)):
            if (node in left_influential) or (node in right_influential):
                continue
            else:
                if include_itself:
                    TMP_neighborhood = network.neighborhood(node, mode = propagation_mode)
                else:
                    TMP_neighborhood = network.neighborhood(node, mode = propagation_mode)
                    TMP_neighborhood.pop(0)
                opinion_list[node] = np.mean(np.array(opinion_list)[TMP_neighborhood]).item()

        Step_Point = np.mean(np.array(opinion_list))
        trange.set_description(desc= f'Propagating, Epoch Delta = {abs(Start_Point - Step_Point)}')
        if np.all(np.abs(np.array(opinion_list) - np.array(prev_opinion_list)) < convergence_tolerance):
            trange.set_description(desc= f'Finished Propagation. Final Delta: {abs(Start_Point - Step_Point)} at step {epoch}')
            break
            
        else:
            Start_Point = Step_Point  
            prev_opinion_list = deepcopy(opinion_list)          
        
    #Calculate the Dipole Moment
    opinion_list = np.array(opinion_list)
    n_nodes = len(degs)
    n_plus = len(opinion_list[opinion_list > 0])
    n_minus = n_nodes - n_plus
    delta_A = np.abs((n_plus - n_minus) * (1 / n_nodes))
    gc_plus = np.mean(opinion_list[opinion_list > 0])
    gc_minus = np.mean(opinion_list[opinion_list < 0])
    
    pole_D = np.abs(gc_plus - gc_minus) * 0.5
    mblb_score = (1 - delta_A) * pole_D
    return mblb_score
def modularity(network:ig.Graph, membership:list, resolution:float=1, directed = True):

    """
    Calculates the modularity for the given graph considering the memberships.
      When resolution parameter is set to one it retrieves the original modularity formula. 
    """
    q_score = network.modularity(membership=membership, weights= None, resolution= resolution, directed= directed)
    return q_score
def krackhardt_ei_ratio(network:ig.Graph, membership:list):
    """
    Cite: https://knapply.github.io/homophily/reference/ei_index.html
    """

    #Calculate the External Ties for Each Community

    unique_comms = np.unique(np.array(membership))
    elist = network.get_edgelist()
    
    IN_COUNT = []
    OUT_COUNT = []
    # For Each Community:
    for comm in unique_comms:
        TMP_IN = 0
        TMP_OUT = 0
        for edge in elist:
            if (membership[edge[0]] == comm) and (membership[edge[1]] == comm):
                TMP_IN += 1
            elif (membership[edge[0]] == comm) or (membership[edge[1]] == comm):
                TMP_OUT += 1
            else:
                continue
        
        IN_COUNT.append(TMP_IN)
        OUT_COUNT.append(TMP_OUT)
    
    return (np.array(IN_COUNT) - np.array(OUT_COUNT)) / (np.array(IN_COUNT) + np.array(OUT_COUNT)) 
def boundary_polarization_2comms(network:ig.Graph, membership:list, mode:str = "all", plot = True):
    # Find the nodes in the boundary
    #Boundary node defined as a node connected to a node from the other community and also connected to a node from its own community
    # where that node is not directly connected to a node from other community.


    vertex_indices = [i for i in range(network.vcount())]
    node_info = []

    for i in vertex_indices:
        tmp_neighborhood = network.neighborhood(i, mode = mode)
        tmp_neighborhood.pop(0)
        if len(tmp_neighborhood) == 0:
            node_info.append('D') # This node is disconnected
        else:
            if len(np.unique(np.array(membership)[tmp_neighborhood])) < 2:
                node_info.append('I')
            else:
                node_info.append('B')
    
    # Refine the Nodes for Outlier Nodes (X)
    for i in vertex_indices:
        if node_info[i] != "B":
            continue
        else:
            tmp_neighborhood = network.neighborhood(i, mode = mode)
            tmp_neighborhood.pop(0)
            if len(np.unique(np.array(node_info)[tmp_neighborhood])) == 1:
                node_info[i] = "X"
            else:
                continue
    
    ## After finding the properties of the nodes, now we compute to boundary polarization.
    ratio_sum = 0
    for node in vertex_indices:
        if node_info[node] == "B": #If node is a boundary node
            #How many of its connections are Internal nodes? 
            tmp_neighborhood = network.neighborhood(node, mode = mode)
            d_iv = np.sum(np.array(node_info)[tmp_neighborhood] == "I")
            d_bv = np.sum(np.array(node_info)[tmp_neighborhood] == "B")
            tmp_ratio = (d_iv / (d_iv + d_bv)) - 0.5
            ratio_sum += tmp_ratio
        else:
            continue
    
    boundary_score = ratio_sum / (np.sum(np.array(node_info) == "B"))
    if plot:
        ig.plot(network, vertex_color = ["yellow" if i == 1 else "green" for i in membership],
                vertex_label = node_info, target= 'graph.png')
        
    return boundary_score

    pass
def segregation_matrix_index(network:ig.Graph, membership:list):

    """
    Calculates the Freshtman(1997) segregation score for n number of communities.

    Ratio of the in group tie density to the out group tie density for each group
    in the network.

    Given a network mixing matrix m_ijt, and two groups G_1 and G_2:

    d_11 = m_111 / (n_1 * (n_1 - 1))
    d_12 = m_121 / n_1 * n_2

    Then the index R is,

    R = d_11 / d_12 

    @param:network ig.Graph = Network where the score is going to be calculated. Note that 
    the function takes in to consideration whether the network is directed or not.

    @param:membership list = The membership list of the nodes.

    @Cite: Bojanowski, Measuring Segregation in Social Networks
    """    

    directed = network.is_directed()
    membership = np.array(membership)
    unique_comms = np.unique(membership).flatten()
    segregation_scores = []
    

    for comm in unique_comms:
        n_comm = np.sum(membership == comm)

        tmp_in_edge_count = 0
        tmp_out_edge_count = 0
        
        if directed:
            in_edge_possible = (n_comm * (n_comm - 1))
        else:
            in_edge_possible = (n_comm * (n_comm - 1)) / 2

        if directed:
            out_edge_possible = (n_comm * (len(membership) - n_comm))
        else:
            out_edge_possible = (n_comm * (len(membership) - n_comm)) / 2


        for (i,j) in network.get_edgelist():
            if ((membership[i] == comm) and (membership[j] == comm)):
                tmp_in_edge_count += 1
            elif ((membership[i] == comm) or (membership[j] == comm)):
                tmp_out_edge_count += 1
            else:
                continue


        tmp_score = (tmp_in_edge_count / in_edge_possible) / (tmp_out_edge_count / out_edge_possible ) 
        segregation_scores.append(int(tmp_score))


    return segregation_scores
def segregation_matrix_index_normalized(network:ig.Graph, membership:list):
    """
    Normalized version of Freshtman's original publication.
    Ratio of the in group tie density to the out group tie density for each group
    in the network.

    Given a network mixing matrix m_ijt, and two groups G_1 and G_2:

    d_11 = m_111 / (n_1 * (n_1 - 1))
    d_12 = m_121 / n_1 * n_2

    Then the index R is,

    R = d_11 / d_12 

    S_SMI = R(G_1) - 1 / R(G_1) + 1

    @param:network ig.Graph = Network where the score is going to be calculated. Note that 
    the function takes in to consideration whether the network is directed or not.

    @param:membership list = The membership list of the nodes.
    """
    
    directed = network.is_directed()
    membership = np.array(membership)
    unique_comms = np.unique(membership).flatten()
    segregation_scores = []
    

    for comm in unique_comms:
        n_comm = np.sum(membership == comm)

        tmp_in_edge_count = 0
        tmp_out_edge_count = 0
        
        if directed:
            in_edge_possible = (n_comm * (n_comm - 1))
        else:
            in_edge_possible = (n_comm * (n_comm - 1)) / 2

        if directed:
            out_edge_possible = (n_comm * (len(membership) - n_comm))
        else:
            out_edge_possible = (n_comm * (len(membership) - n_comm)) / 2


        for (i,j) in network.get_edgelist():
            if ((membership[i] == comm) and (membership[j] == comm)):
                tmp_in_edge_count += 1
            elif ((membership[i] == comm) or (membership[j] == comm)):
                tmp_out_edge_count += 1
            else:
                continue

        

        tmp_score = (tmp_in_edge_count / in_edge_possible) / (tmp_out_edge_count / out_edge_possible ) 
        tmp_score = float((tmp_score - 1) / (tmp_score + 1))
        segregation_scores.append(tmp_score)

    return segregation_scores
def freemans_segregation_index(network:ig.Graph, membership:list):
    """
    Cite: Freeman 1978b
    1 - (Density of the Cross Group Ties * (1/(Density of the Network)))
    """


    across_group_tie_count = 0
    edge_list = network.get_edgelist()
    directed_info = network.is_directed()

    for (i,j) in edge_list:
        if membership[i] != membership[j]:
            across_group_tie_count += 1
    
    
    
    unique_membership = np.unique(np.array(membership)).flatten()
    possible_across_group_tie = []
    for comm in unique_membership:
        for comm_2 in unique_membership:
            if comm == comm_2:
                possible_across_group_tie.append(0)
            else:
                possible_across_group_tie.append(int(np.sum(membership == comm) * np.sum(membership == comm_2)))


    possible_across_group_tie = np.sum(possible_across_group_tie)
    if not directed_info:
        possible_across_group_tie = possible_across_group_tie / 2

    density_of_the_cross_group_ties = across_group_tie_count / possible_across_group_tie
    

    density_of_the_network = (network.vcount() * (network.vcount() - 1))
    if not directed_info:
        density_of_the_network =  density_of_the_network / 2
    density_of_the_network = network.ecount() / density_of_the_network
    
    freeman_segregation_index = 1 - (density_of_the_cross_group_ties / density_of_the_network)
    return freeman_segregation_index
def moodys_odds_ratio_2comms(network:ig.Graph, membership:list, log_return = True):

    """
    @Cite: Moody - 2001  Charles and Grusky - 1995, Bojanowski
    
    S_Moody = (m111 + m221) * n1*n2 / (n1**2 + n2**2) * m121 
    """
    
    # Calculate the in group and cross group edges
    m111 = 0 
    m221 = 0
    m121 = 0
    n1 = int(np.sum(np.array(membership) == 0))
    n2 = int(np.sum(np.array(membership) == 1))
    print(n1,n2)
    for (i,j) in network.get_edgelist():
        if (membership[i] == membership[j]) and (membership[i] == 0):

            m111 += 1
        elif (membership[i] == membership[j]) and (membership[i] == 1):

            m221 += 1
        else:
            
            m121 += 1
    
    print(m111, m121, m221)
    s_moody = ((m111 + m221) * n1 * n2) / ((n1**2 + n2**2) * m121)
    
    return log(s_moody) if log_return else s_moody


