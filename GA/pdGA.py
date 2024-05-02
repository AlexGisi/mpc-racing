import numpy as np
from splines.ParameterizedCenterline import ParameterizedCenterline
from splines.TrackSegments import TrackSegments

population_size = 10  # Npop
num_parameters = 4  # Number of controller parameters -- [kP_steer, kD_steer, kP_throttle, kD_throttle]
num_generations = 10
num_checkpoints = 20  # NCP
mutation_rate = 0.4  # P(M)
crossover_rate = 0.5  # P(C)
kT = 4  # s-1
kc = 0.5
lambda_T = 0.3

class GeneticAlgorithm:
    def __init__(self):
        self.line = ParameterizedCenterline()
        self.line.from_file("/home/alex/Projects/graic/autobots-race/waypoints/shanghai_intl_circuit")

        self.segments = TrackSegments(self.line, num_checkpoints, 30, 1500, 1500, 250)
        self.avg_times = np.full((num_checkpoints), self.segments.lap_time / num_checkpoints)
        
        self.current_population = np.random.rand(population_size, num_parameters)
        self.segment_timings = np.zeros(population_size)


    def initialize_population(self):
        return np.random.rand(population_size, num_parameters)

    def compute_reward(self, segment_time, average_time):
        return np.exp(-kT * (segment_time - average_time))
    
    def getPopSize():
        return population_size

    def compute_avg_time(self, segment_time, average_time):
        return lambda_T * segment_time + (1 - lambda_T) * average_time
    
    def getPop(self, index):
        return self.current_population[index]
    
    def getSegmentNumber(self, x, y):
        return self.segments.get_seg_num(x, y)
    
    def updateSegmentTimings(self, index, timing):
        self.segment_timings[index] = timing

    def evaluate_population(self):
        rewards = np.zeros(population_size)
        for i in range(population_size):
            segment_time = self.segment_timings[i]
            average_time = self.avg_times[i]
            self.avg_times[i % num_checkpoints] = self.compute_avg_time(segment_time, average_time) ## if this doesn't work, we can set a goal time ~ 80 seconds, and divide by number of segments ==> use that as average time
            rewards[i] = self.compute_reward(segment_time, average_time)
        return rewards

    def select_parents(self, rewards):
        population = self.current_population
        # Select parents based on their fitness (roulette wheel selection)
        fitness_probs = rewards / np.sum(rewards)
        selected_indices = np.random.choice(np.arange(population_size), size=2, p=fitness_probs)
        return population[selected_indices]

    def crossover(self, parent1, parent2):
        # Perform crossover to produce offspring
        crossover_point = np.random.randint(1, num_parameters)
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return offspring1, offspring2

    def mutate(self, offspring):
        # Apply mutation to the offspring
        mutation_indices = np.random.rand(num_parameters) < mutation_rate
        mutation_values = np.random.rand(num_parameters) - 0.5  # Small random changes
        offspring[mutation_indices] += mutation_values[mutation_indices]
        return offspring

    def runStep(self):
        rewards = self.evaluate_population()

        parent1, parent2 = self.select_parents(rewards)

        offspring1, offspring2 = self.crossover(parent1, parent2)
        offspring1 = self.mutate(offspring1)
        offspring2 = self.mutate(offspring2)

        self.current_population[np.argsort(rewards)[:2]] = offspring1, offspring2

        best_individual_index = np.argmax(rewards)
        best_individual = self.current_population[best_individual_index]
        print("Best Individual:", best_individual, "Reward:", rewards[best_individual_index])

if __name__ == '__main__':
    ga = GeneticAlgorithm()
    ga.genetic_algorithm()
