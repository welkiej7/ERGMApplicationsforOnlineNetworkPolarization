import igraph as ig
import numpy as np
from tqdm import tqdm
from copy import deepcopy

'''
Generation of the follower and following network and the retweet network as
a half agent based model.
'''

def calculate_similarity(user_space:np.array, normalize = True):

    '''
    Calculates the user similarity with euclidean distance and normalizes it
    between 0 and 1.
    '''

    user_similarity_matrix = np.zeros(shape=(user_space.shape[0],user_space.shape[0]))
    for user in range(user_space.shape[0]):
        for connection in range(user, user_space.shape[0]):
            user_similarity_matrix[user,connection] = np.linalg.norm(user_space[user] - user_space[connection])

    user_similarity_matrix = user_similarity_matrix + user_similarity_matrix.T - np.diag(user_similarity_matrix.diagonal())

    #normalize
    if normalize:
        user_similarity_matrix = np.max(user_similarity_matrix) - user_similarity_matrix

        user_similarity_matrix = (user_similarity_matrix - np.min(user_similarity_matrix))/(np.max(user_similarity_matrix) - np.min(user_similarity_matrix))

        return user_similarity_matrix
    else:
        return user_similarity_matrix



def generate_follower_network_directed_vanilla(deg_dist:list,
                              homophily_level:int,
                              user_similarity_matrix:np.array,
                              prune = True) -> ig.Graph:

    '''
    No self loops sequential vote casting. Multiple edges are permitted.

    '''



    if user_similarity_matrix.shape[0] != len(deg_dist):
        raise ValueError('Length mismatch.')

    edge_list = []
    for node in range(len(deg_dist)):
        similarity = user_similarity_matrix[:,node]
        similarity[node] = 0
        similarity = (similarity ** homophily_level) / np.sum(similarity ** homophily_level)
        tot_vot_right = int(deg_dist[node].item())
        for __ in range(tot_vot_right):
            possible_cast = np.random.choice(range(len(deg_dist)), size = 1, p = similarity).item()
            print('\x1b[0;30;47m' +f'Suggesting {node} ->> {possible_cast}'+ '\x1b[0m')
            print('\x1b[0;30;46m' +f'Accepted {node} ->> {possible_cast}'+ '\x1b[0m')
            edge_list.append((node,possible_cast))
    if prune:
        return ig.Graph(n = user_similarity_matrix.shape[0], edges = edge_list, directed = True).simplify(multiple = True)
    else:
        return ig.Graph(n = user_similarity_matrix.shape[0], edges = edge_list, directed = True)





def generate_follower_network_exhaustive(out_deg_dist:list,
                                        in_deg_dist:list,
                                        homophily_level:int,
                                        user_similarity_matrix:np.array,
                                        generation_tolerance = 1e4):


    '''
    No self loops, sequential vote casting. Multiple edges are not allowed
    and degree distribution is absolute.
    '''

    if (user_similarity_matrix.shape[0] != len(out_deg_dist)) or (user_similarity_matrix.shape[0] != len(out_deg_dist)):
        raise ValueError('Length mismatch')

    edge_list = [] #Initialize the edge list.

            # For each node suggest a cast to a possible node.
            # If possible node connection does not exist and offered node has acceptance for the indegree, accept the offer.

    for NODE in range(len(out_deg_dist)):
        similarity_list = user_similarity_matrix[:,NODE]
        similarity_list[NODE] = 0 # Prevent self loops
        similarity_list = (similarity_list**homophily_level) / (np.sum(similarity_list**homophily_level))
        casting_right = out_deg_dist[NODE]

        for CAST in range(int(casting_right.item())):
            POSSIBLE_CAST = np.random.choice(range(user_similarity_matrix.shape[0]), size = 1, p = similarity_list).item()
            print('\x1b[0;30;47m' +f'Suggesting {NODE} ->> {POSSIBLE_CAST}'+ '\x1b[0m')
            if (NODE,POSSIBLE_CAST) in edge_list:
                print('\x1b[0;37;41m' +f'Rejected: Multiple {NODE} ->> {POSSIBLE_CAST}'+ '\x1b[0m')
                similarity_list[POSSIBLE_CAST] = 0
                similarity_list = (similarity_list) / (np.sum(similarity_list))
                    
            elif in_deg_dist[POSSIBLE_CAST] < 1:
                print('\x1b[0;37;41m' +f'Rejected: No Right {NODE} ->> {POSSIBLE_CAST}'+ '\x1b[0m')
                similarity_list[POSSIBLE_CAST] = 0
                similarity_list = (similarity_list) / (np.sum(similarity_list))

            else:
                print('\x1b[0;30;46m' +f'Accepted {NODE} ->> {POSSIBLE_CAST}'+ '\x1b[0m')
                edge_list.append((NODE,POSSIBLE_CAST))
                in_deg_dist[POSSIBLE_CAST] = in_deg_dist[POSSIBLE_CAST] - 1



    
    return ig.Graph(n = user_similarity_matrix.shape[0], edges = edge_list, directed = True)



def generate_retweet_network(follower_following_network:ig.Graph,
                             number_of_retweets:list,
                             echo_chamber_strength:float,
                             homophily_level:int,
                             user_space:np.array,
                             topic:int,
                             single_retweet:bool = True):
    similarity_matrix = calculate_similarity(user_space[:,topic], normalize= True)



    rt_list = []
    for NODE in tqdm(range(len(number_of_retweets)), desc = 'Retweeting'):
        try:
            RT_RIGHT = number_of_retweets[NODE]
            similarity_list = similarity_matrix[:,NODE]
            similarity_list[NODE] = 0 # No Self Retweets
            similarity_list = (similarity_list ** homophily_level) / (np.sum(similarity_list**homophily_level))

            for __ in range(RT_RIGHT):
                CAST_TYPE = bool(np.random.binomial(1, p = echo_chamber_strength))

                if CAST_TYPE:
                    poss = follower_following_network.neighbors(NODE, mode = 'in')
                    similarity_list_neigh = similarity_list[poss] / np.sum(similarity_list[poss])
                    POSSIBLE_CAST = np.random.choice(poss, size = 1,p = similarity_list_neigh).item()
                    print('\x1b[0;30;45m' + f'Retweet {NODE} ->> {POSSIBLE_CAST}, Echo Chamber' + '\x1b[0m')
                    
                else:
                    POSSIBLE_CAST = np.random.choice(range(similarity_matrix.shape[0]), size = 1, p = similarity_list).item()
                    print('\x1b[0;30;44m' + f'Retweet {NODE} ->> {POSSIBLE_CAST}, Feed Observation' + '\x1b[0m')

                rt_list.append((NODE,POSSIBLE_CAST))
        except Exception as e:
            print(e)
            continue

    
    return ig.Graph(similarity_matrix.shape[0], rt_list, directed = True)


    """
    For each node:

        Get the retweet right
        Cast n numbers of retweet according to their right and similarity matrix in one topic.



    """



if __name__ == "__main__":
    from user_space_models import generate_user_space
    from network_generation import calculate_similarity, generate_follower_network_exhaustive, generate_retweet_network
    import numpy as np

    some_space = generate_user_space(n_users= 100,
                            n_cleavage= 3,
                            n_non_cleavage= 1,
                            n_noise= 1,
                            n_unifying_topics= 1,
                            unification_point= 0.4,
                            cleavage_alignment= 0.95,
                            non_cleavage_alignment= 0.9,
                            bimodal_polarization_strength= 2,
                            non_polarization_attraction=8,
                            clevage_alignment_noise= 0.1,
                            non_cleavage_alignment_noise=0.1,
                            unification_strength=100,
                            unification_alignment_noise=0.01,
                            unification_alignment=1,
                            noise_topic_noise= 0.8)




    similarity_matrix = calculate_similarity(some_space)
    out_deg_dist = np.random.uniform(low = 1, high= 5, size = some_space.shape[0])
    in_deg_dist = np.random.uniform(low= 1, high= 5, size = some_space.shape[0])
    network = generate_follower_network_exhaustive(out_deg_dist,in_deg_dist, 
                                                   homophily_level= 10, 
                                                   user_similarity_matrix= similarity_matrix, 
                                                   generation_tolerance= 1e4)
    retweet_right = [round(i) for i in np.random.uniform(low = 0, high = 5, size = similarity_matrix.shape[0])]
    rt = generate_retweet_network(network, 
                            user_space= some_space,
                            topic= 1,
                            number_of_retweets=retweet_right,
                            echo_chamber_strength= 0.8,
                            homophily_level= 5,
                            single_retweet= True)
    print(rt)