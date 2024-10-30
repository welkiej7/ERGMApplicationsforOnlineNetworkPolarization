import igraph as ig
import pymetis


def partition_with_metis(network, number_of_partitions):

    '''
    Wrapper around METIS partitioning algorithm to divide network in 'n' communities.
    '''
    ## Take igraph object at turn it to an adjacency list(?) which is a super weird thing to
    ## turn it in to but oh my god, how people are coding these days?
    
    adj_list = ig.Graph.get_adjlist(network)
    ncuts, membership = pymetis.part_graph(nparts= number_of_partitions, adjacency= adj_list)

    return ncuts, membership