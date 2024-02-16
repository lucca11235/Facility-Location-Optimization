import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def slice_integer(N, slicers):
    x_slices = [0] + [int(s*N) for s in slicers] + [N]
    return np.diff(x_slices)

def slice_matrix_by_slicers(Ns,slicers):
    final = np.zeros(slicers.shape + np.array([1,0]))
    for i in range(slicers.shape[1]):
        slices = slice_integer(Ns[i], slicers[:,i])
        final[:,i] = slices
    return final


class SecondSpaceGA:
    def __init__(self, facility_information):
        self.facility_information = facility_information
        self.demands = np.array(list(self.facility_information['demands'].values()))
        self.capacities = np.array(list(self.facility_information['capacities'].values()))
        self.opening_costs = np.array(list(self.facility_information['opening_costs'].values()))
        self.transportation_costs = np.array([list(inner_dict.values()) for inner_dict in self.facility_information['transportation_costs'].values()])
        self.facility_names = list(self.facility_information['capacities'].keys())
        self.demand_points_names = list(self.facility_information['demands'].keys())

    def _create_individual(self):
        # Create a random individual with sorted values for each column
        try: return np.sort(np.random.rand(self.temp_shape[0]-1,self.temp_shape[1]), axis=0)
        except: return np.ones(self.temp_shape[1])

    def _valid_individual(self, individual):
        # Check if an individual's allocations do not exceed facility capacities
        units_facility = np.sum(slice_matrix_by_slicers(self.temp_demands,individual),axis = 1)
        return all(x <= y for x, y in zip(units_facility, self.temp_capacities))

    def _penalty(self, individual):
        # Calculate penalty for exceeding capacities
        units_facility = np.sum(slice_matrix_by_slicers(self.temp_demands,individual),axis = 1)
        penalty = np.sum([np.max([0, x - y]) for x, y in zip(units_facility, self.temp_capacities)])
        return penalty

    def _fitness(self, individual):
        # Calculate fitness as the sum of opening costs, transportation costs, and penalties
        units_matrix = slice_matrix_by_slicers(self.temp_demands,individual)
        open_facilities = np.any(units_matrix != 0, axis=1).astype(int)
        penalty = self._penalty(individual)
        fitness = open_facilities@self.temp_opening_costs + np.sum(units_matrix*self.temp_transportation_costs) + 10*penalty
        return fitness

    def _mutate(self, individual, mutation_rate=0.33, mutation_strength=0.2):
        # Randomly mutate elements of an individual, adding a gaussian noise to 33% of the numbers
        total_elements = individual.shape[0] * individual.shape[1]
        mutations_count = int(total_elements * mutation_rate)
        mutation_indices = np.random.choice(total_elements, mutations_count, replace=False)
        for idx in mutation_indices:
            row, col = divmod(idx, individual.shape[1])
            mutation_value = np.random.normal(0, mutation_strength)
            individual[row, col] += mutation_value
            individual[row, col] = np.clip(individual[row, col], 0, 1)
        individual.sort(axis=0)

    def _crossover(self, parent1, parent2):
        # Combine columns from two parents to produce offspring
        mask = np.random.randint(2, size=parent1.shape[1])
        offspring1 = np.zeros_like(parent1)
        offspring2 = np.zeros_like(parent2)
        for i in range(parent1.shape[1]):  # Iterate over columns
          if mask[i] == 1:
            offspring1[:, i] = parent1[:, i]
            offspring2[:, i] = parent2[:, i]
          else:
            offspring1[:, i] = parent2[:, i]
            offspring2[:, i] = parent1[:, i]
        return offspring1, offspring2

    def _selection(self, population, fitness_scores):
        # Select parents using tournament selection
        num_pairs = len(population) // 2
        def tournament_selection():
            tournament_size = 3
            selected_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_individuals = population[selected_indices]
            tournament_fitness_scores = fitness_scores[selected_indices]
            winner_index = np.argmin(tournament_fitness_scores)
            return tournament_individuals[winner_index]
        selected_parents = [[tournament_selection() for _ in range(2)] for _ in range(num_pairs)]
        return selected_parents

    def report(self,list = False):
        # Print the report as in the .txt file
        print(f"\nBest fitness: {self.best_fitness}\n")

        for facility in np.array(self.facility_names)[self.open_facilities]:
          print(f'Facility {facility} is open.')

        print('\n Matrix of transportation:')
        df = pd.DataFrame(slice_matrix_by_slicers(self.temp_demands,self.best_individual), columns = self.demand_points_names)
        df.index = self.temp_facility_names
        print(df)

        if list:
          routing_matrix = slice_matrix_by_slicers(self.temp_demands,self.best_individual)
          for i in range(routing_matrix.shape[0]):
            for j in range(routing_matrix.shape[1]):
              quantity = routing_matrix[i][j]
              if quantity == 0:
                continue
              facility = self.facility_names[i]
              client = self.demand_points_names[j]
              print(f'Facility {facility} serves {quantity} units to demand point {client}')

        print(f'\nIs valid: {self._valid_individual(self.best_individual)}\n')
        print('\n')
        plt.figure(figsize = (10,5))
        plt.plot(self.fitness_history)
        plt.title("Fitness over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def optimize(self,
                 open_facilities,
                 size=25,
                 iterations=50,
                 mutation_rate=0.1,
                 verbose=True):

        # Optimize to find the best individual
        self._initialize_temporary_parameters(open_facilities)
        self.temp_population = np.array([self._create_individual() for _ in range(size)])
        self.best_individual,self.best_fitness = None, np.inf
        self.fitness_history = []
        self.open_facilities = open_facilities

        for generation in range(iterations):
            fitness_scores = np.array([self._fitness(individual) for individual in self.temp_population])
            valid_indices = [i for i, individual in enumerate(self.temp_population) if self._valid_individual(individual)]
            valid_population,valid_fitness_scores  = self.temp_population[valid_indices],fitness_scores[valid_indices]

            if len(valid_population) > 0:
                best_valid_index = np.argmin(valid_fitness_scores)
                if valid_fitness_scores[best_valid_index] < self.best_fitness:
                    self.best_individual = valid_population[best_valid_index]
                    self.best_fitness = valid_fitness_scores[best_valid_index]

            if verbose:
                print(f"Generation: {generation + 1}, Best fitness: {self.best_fitness}, Valid individuals: {len(valid_population)}")

            self.fitness_history.append(self.best_fitness)

            # Selection
            parents = self._selection(self.temp_population, fitness_scores)
            # Crossover
            children = []
            for parent1, parent2 in parents:
                child1, child2 = self._crossover(parent1, parent2)
                children.append(child1)
                children.append(child2)
            # Mutate
            for child in children:
                if np.random.rand() < mutation_rate:
                    self._mutate(child)

            # Replace population with children
            self.temp_population = np.array(children)

        return self.best_fitness, self.best_individual, len(valid_population)

    def _initialize_temporary_parameters(self, open_facilities):
        # Set temporary parameters for optimization
        self.temp_demands = self.demands
        self.temp_capacities = self.capacities[open_facilities]
        self.temp_opening_costs = self.opening_costs[open_facilities]
        self.temp_transportation_costs = self.transportation_costs[open_facilities]
        self.temp_shape = self.temp_transportation_costs.shape
        self.temp_facility_names = [self.facility_names[i] for i in open_facilities]