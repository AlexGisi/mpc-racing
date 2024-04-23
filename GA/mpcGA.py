import numpy as np
import splines
import splines.ParameterizedCenterline
import splines.TrackSegments

population_size = 100  # Npop
num_parameters = 5  # Number of controller parameters
num_generations = 10
num_checkpoints = 5  # NCP
mutation_rate = 0.4  # P(M)
crossover_rate = 0.5  # P(C)
kT = 4  # s-1
kc = 0.5
lambda_T = 0.3  # Î»T

def initialize_population(population_size, num_parameters):
    return np.random.rand(population_size, num_parameters)

def compute_reward(segment_time, average_time):
    ## to be changed based on how other parts are done
    return np.exp(-kT * (segment_time - average_time))

def compute_avg_time(segment_time, average_time):
    return lambda_T * segment_time + (1 - lambda_T) * average_time

def genetic_algorithm(population_size, num_generations, num_parameters):

    line = splines.ParameterizedCenterline.ParameterizedCenterline()
    line.from_file = ("waypoints/shanghai_intl_circuit")

    segments = splines.TrackSegments.TrackSegments(line, num_checkpoints, 30, 1500, 1500, 250)

    avg_times = np.full((num_checkpoints), segments.lap_time / num_checkpoints)

    population = initialize_population(population_size, num_parameters)

    segment = 0
    
    for generation in range(num_generations):
        rewards = np.zeros(population_size)
        
        for i in range(population_size):
            segment_time = np.random.rand() # place holder -- get from previous parts
            average_time = avg_times[segment]
            
            avg_times[segment] = compute_avg_time(segment_time, average_time)

            segment += 1
            if segment >= num_checkpoints:
                segment = 0

            reward = compute_reward(segment_time, average_time)
            rewards[i] += reward

        total_reward = np.sum(rewards)
        fitness = rewards / total_reward

        population = np.random.rand(population_size, num_parameters)
    
        best_individual_index = np.argmax(rewards)
        best_individual = population[best_individual_index]
        print(best_individual)

if __name__ == '__main__':
    result = genetic_algorithm(population_size, num_generations, num_parameters)
    print(result)