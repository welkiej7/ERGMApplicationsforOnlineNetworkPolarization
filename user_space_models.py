import numpy as np
import warnings



def generate_aligned_array(original_array:np.array,
                           alignment_strength:float = 0.8,
                           noise_level:float = 0.2):

    if alignment_strength > 1 or alignment_strength < 0:
        raise ValueError(f'Alignment strength is outside of the range (0,1)')
    if noise_level > 1 or noise_level < 0:
        raise ValueError(f'Noise level is outside of the range (0,1)')

    generated_noise = np.random.uniform(-1,1,original_array.shape) * noise_level
    aligned_array = alignment_strength * original_array + (1 - alignment_strength) * generated_noise
    aligned_array = np.clip(aligned_array, -1, 1)


    return aligned_array

def generate_user_space(n_users:int,
                        n_cleavage:int,
                        n_non_cleavage:int,
                        n_noise:int,
                        n_unifying_topics:int,
                        unification_point:float,
                        cleavage_alignment:float,
                        non_cleavage_alignment:float,
                        bimodal_polarization_strength:float=2,
                        non_polarization_attraction:float = 8,
                        clevage_alignment_noise:float = 0.2,
                        non_cleavage_alignment_noise:float = 0.1,
                        unification_strength:int = 100,
                        unification_alignment_noise:float = 0.01,
                        unification_alignment:int = 1,
                        noise_topic_noise:float = 0.8):

    '''
    Definition: User space generator for the network generation model. Contains 7 parameters
    to generate a (n_cleavage + n_non_cleavage + n_noise + n_unifying_topics) * n_users matrix.
    Each user has an opinion in the total generated topics between -1 and 1. Alignment parameters will
    set how much these opinions are aligned.


    @param:int n_users = Number of users to be generated.
    @param:int n_cleavage = Number of cleavage topics to be generated.
    @param:int n_non_cleavage = Number of non - cleavage topics to be generated.
    @param:int n_noise = Number of noise, or uniform, or irrelevant topics to be generated.
    @param:int n_unifying_topics = Number of unifying topics to be generated
    @param:float unification_point = The point of unification for the generated unifying topics.
    @param:float cleavage_alignment = The alignment for the clevage topics, 1 is perfect alignment, 0 is no alignment at all.
    @param:float non_cleavage_alignment = The alignment for the non clevage topics, standard deviation for a
    normal distribution for each user's score in a non cleavage topic.
    @param:float bimodal_polarization_strength = The inverse will be passed as the alpha and the beta of the
    beta distribution parameters.
    @param:float non_polarization_attraction = The amount of attractiveness to meet in the middle, lower the better.
    @param:float **_noise = The amount of randomness to be added for the generation process of the selected topic type.
    '''

    #Set the user space where each row is a user and each column is a topic.
    user_space = np.zeros(shape=(n_users,n_cleavage + n_non_cleavage + n_noise + n_unifying_topics))
    #For each topic type we have a different generation strategy. After generating we are going to
    #assign it to a row in the user space matrix. First generate the polarized topics.
    #Check the bimodal_polarization_strength
    if bimodal_polarization_strength < 1:
        warnings.warn('Bimodal Polarization strength is too low to generate a polarized distribution.')
    # Take a beta distribution from the 1 /  of the parameter.
    bimodal_polarization_strength = 1 / bimodal_polarization_strength
    core_polarized = np.random.beta(bimodal_polarization_strength, bimodal_polarization_strength, size = n_users)
    #Apply a linear transformation to the beta distribution to make it between -1 and 1
    core_polarized = (core_polarized * 2) - 1
    #Now sample the other topics according to the alignment strength.
    for COL in range(n_cleavage):
        TMP_aligned_array = generate_aligned_array(core_polarized, alignment_strength= cleavage_alignment, noise_level= clevage_alignment_noise)
        user_space[:,COL] = TMP_aligned_array

    if non_polarization_attraction < 1e-2:
        warnings.warn('Non polarization attraction is too low to generate a non polarized distribution.')
    non_polarization_attraction = 1 / non_polarization_attraction
    core_non_polarized = np.random.normal(loc= 0, scale = non_polarization_attraction, size= n_users)
    for COL in range(n_cleavage, n_cleavage + n_non_cleavage):
        TMP_aligned_array = generate_aligned_array(core_non_polarized, alignment_strength= non_cleavage_alignment, noise_level= non_cleavage_alignment_noise)
        user_space[:,COL] = TMP_aligned_array

    # Generate noise topics.
    noise_topics = np.random.uniform(-1,1,size = n_users)
    for COL in range(n_cleavage + n_non_cleavage, n_cleavage + n_non_cleavage + n_noise):
        user_space[:,COL] = generate_aligned_array(noise_topics, alignment_strength= 0.1, noise_level= noise_topic_noise)

    #Generate unified topics.
    unification_strength = 1 / unification_strength
    unified_topics = np.random.normal(loc = unification_point, scale= unification_strength, size = n_users)
    for COL in range(n_cleavage + n_non_cleavage + n_noise, n_cleavage + n_non_cleavage + n_noise + n_unifying_topics):
        user_space[:,COL] = generate_aligned_array(unified_topics, unification_alignment, unification_alignment_noise)



    return user_space



if __name__ == '__main__':

    generate_user_space(n_users= 100,
                        n_cleavage=10,
                        n_non_cleavage=3,
                        n_noise=2,
                        n_unifying_topics=1,
                        unification_point = -0.4,
                        cleavage_alignment=0.8,
                        non_cleavage_alignment=0.7,
                        bimodal_polarization_strength=2,
                        non_polarization_attraction=10,
                        clevage_alignment_noise=0.1,
                        non_cleavage_alignment_noise=0.2,
                        unification_strength = 100,
                        unification_alignment_noise= 0.01,
                        noise_topic_noise= 0.8)
