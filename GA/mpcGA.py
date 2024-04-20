import numpy as np

population_size = 100  # Npop
num_parameters = 5  # Number of controller parameters
num_checkpoints = 5  # NCP
mutation_rate = 0.4  # P(M)
crossover_rate = 0.5  # P(C)
kT = 4  # s-1
kc = 0.5
lambda_T = 0.3  # Î»T

def initialize_population(population_size, num_parameters):
    return np.random.rand(population_size, num_parameters)


def compute_reward(segment_time, average_time, cones_hit):
    ## to be changed based on how other parts are done
    return np.exp(-kT * (segment_time - average_time) + kc * cones_hit)


def genetic_algorithm(population_size, num_generations, num_parameters):

    population = initialize_population(population_size, num_parameters)
    
    for generation in range(num_generations):
        rewards = np.random.rand(population_size) 
        
        new_population = np.zeros_like(population)

        for i in range(population_size):
            segment_time = np.random.rand() # place holder -- get from previous parts
            average_time = np.random.rand() # place holder -- get from previous parts
            cones_hit = np.random.randint(10) # place holder -- get from previous parts


            reward = compute_reward(segment_time, average_time, cones_hit)
            rewards[i] = reward
            
    
            new_population[i] = np.random.rand(num_parameters)  
            
        population = np.random.rand(population_size, num_parameters)
    
    best_individual_index = np.argmax(rewards)
    best_individual = population[best_individual_index]
    return best_individual
