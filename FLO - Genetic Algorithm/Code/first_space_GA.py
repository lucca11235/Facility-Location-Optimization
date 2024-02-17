from second_space_GA import SecondSpaceGA
import numpy as np

class FirstSpaceGA:
    def __init__(self, facility_information):
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = None
        self.generation = 0

        self.mutation_rate = 0.50

        self.shape = len(facility_information['capacities'])
        self.secondSpaceGA = SecondSpaceGA(facility_information)

    def _create_individual(self):
        individual = np.ones(self.shape, dtype=int)
        individual[np.random.rand(self.shape) < 0.25] = 0
        if sum(individual) == 0: individual[np.random.randint(len(individual))] = 1
        return individual

    def initialize_population(self, N):
        self.population = np.array([self._create_individual() for _ in range(N)])

    def _fitness(self, individual):
        # Calculate the fitness of an individual based on the routing optimization
        open_facilities = np.where(individual == 1)[0]
        best_route_fitness, _, penalty = self.secondSpaceGA.optimize(open_facilities, verbose=False)
        return best_route_fitness

    def _mutate(self, individual):
        # Mutate an individual by flipping bits with a certain probability guarateeing there is at least one 1
        mutation_mask = np.random.rand(len(individual)) < 0.2
        individual[mutation_mask] = 1 - individual[mutation_mask]
        if sum(individual) == 0: individual[np.random.randint(len(individual))] = 1

    def _crossover(self, individual1, individual2):
        # Perform a single-point crossover between two individuals
        crossover_point = np.random.randint(1, len(individual1))
        return (np.concatenate([individual1[:crossover_point], individual2[crossover_point:]]),
                np.concatenate([individual2[:crossover_point], individual1[crossover_point:]]))

    def _selection(self, population, fitness_scores):
        # Select parents for the next generation using tournament selection
        selected_parents = []
        num_pairs = len(population) // 2
        for _ in range(num_pairs):
            parents = []
            for _ in range(2):
                selected_indices = np.random.choice(len(population), 3, replace=False)
                tournament_individuals = population[selected_indices]
                tournament_fitness_scores = fitness_scores[selected_indices]
                winner_index = np.argmin(tournament_fitness_scores)
                parents.append(tournament_individuals[winner_index])
            selected_parents.append(parents)
        return selected_parents

    def run(self, iterations):
        self.best_individual,self.best_fitness = None, np.inf
        for generation in range(iterations):

            # Calculate fitness for each individual in the population
            fitness_scores = np.array([self._fitness(individual) for individual in self.population])

            # Get indices of sorted fitness scores
            sorted_indices = np.argsort(fitness_scores)  # Negative for descending order

            # Sort the population and fitness scores based on sorted indices
            self.population = self.population[sorted_indices]
            sorted_fitness_scores = fitness_scores[sorted_indices]

            # Update the best individual and best fitness
            if sorted_fitness_scores[0]< self.best_fitness:
              self.best_fitness = sorted_fitness_scores[0]
              self.best_individual = self.population[0]
              self.fitness_history.append(self.best_fitness)

            print(f'Generation: {generation+1} Best Fitness: {self.best_fitness} Best Individual: {self.best_individual}')

            # Selection (you need to implement or update the _select_parents method accordingly)
            parents = self._selection(self.population,sorted_fitness_scores)

            # Crossover and mutation (ensure these methods are compatible with your population structure)
            children = []
            for parent1, parent2 in parents:
                child1, child2 = self._crossover(parent1, parent2)
                children.append(child1)
                children.append(child2)

            # Mutate
            for child in children:
                self._mutate(child)
            # Replace the old population with the new one (make sure to convert children back to a NumPy array if needed)
            self.population = np.array(children)
