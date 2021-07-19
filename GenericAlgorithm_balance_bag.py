# imports
import sys

import numpy as np
import random


class GenericAlgorithm:
    # default parameters
    n_subsets = 2
    n_items = 5
    n_population = 7
    runtime = 5
    # important variables
    population = np.zeros(n_items)
    weight = np.ones(n_items)
    optimum = 0.
    fitness_population = []

    # class constructor
    def __init__(self, n_population, n_items, n_subsets, runtime):
        self.n_population = n_population
        self.n_items = n_items
        self.n_subsets = n_subsets
        self.runtime = runtime
        self.initial_population()
        self.calc_fitness_population()

    # diverse functions
    # create initial population; weights are assigned corresponding to a given weight vector. this vector can be changed here
    def initial_population(self):
        item_number = np.arange(0, self.n_items)  # max items

        # set weight vector here please; IMPORTANT: vector length must match n_items given in argument
        weight = np.array([5,10,8,2,1,4,3,5,2,2,5,8,3,2,3])

        # calculate optimum = value every subset should ideally obtain
        optimum = np.sum(weight) / self.n_subsets
        # print random created items and their corresponding weight
        print('Item No.\tWeight')
        for i in range(item_number.shape[0]):
            print('{0}\t\t\t{1}\n'.format(item_number[i], weight[i]))

        # next we randomly assign each item to a random subset in each chromosome for the initial population
        pop_size = (self.n_population, item_number.shape[0])
        print('Population size = {}'.format(pop_size))
        initial_population = np.random.randint(self.n_subsets,
                                               size=pop_size)  # here is the number of total subsets n_subsets important
        initial_population = initial_population.astype(int)
        print('Initial population: \n{}'.format(initial_population))
        self.population = initial_population
        self.weight = weight
        self.optimum = optimum

    # get total weight of a subset of a specific chromosome
    def get_total_weight(self, chromosome, subset):
        total_weight = 0
        for i in range(len(chromosome)):
            if chromosome[i] == subset:
                total_weight += self.weight[i]
        return total_weight

    # calculate fitness of a single chromosome
    def calc_fitness(self, chromosome):
        # fitness is distance from subset means to the optimum
        # calculate total weight (=accumulated weight) of every subset
        list_subset_weights = []
        for i in range(self.n_subsets):
            list_subset_weights.append(self.get_total_weight(chromosome, i))
        # calculate distance: subtract optimum and square the result (= gain absolute values)
        for i in range(self.n_subsets):
            list_subset_weights[i] -= self.optimum
            list_subset_weights[i] *= list_subset_weights[i]
        # fitniss is 1000 divided by the sum of distances; IMPORTANT: if sum(list_subset_weights) is equal to 0 then we found the optimum and cancel
        # the evolution process
        if sum(list_subset_weights) == 0:
            print("Found optimum solution!")
            print(chromosome)
            self.print_optimum_solution(chromosome)
        else:
            return 1000 / sum(list_subset_weights)

    # calculate fitness of whole population (=all chromosomes)
    def calc_fitness_population(self):
        fitness_population = []
        for chromosome in self.population:
            fitness_population.append(self.calc_fitness(chromosome))
        self.fitness_population = fitness_population

    # pick a pair of parents - probability is higher if fitness score is better
    def select_pair(self):
        return random.choices(self.population, self.fitness_population, k=2)

    # crossover of two parents at a random point within the sequence
    def single_point_crossover(self, a, b):
        if len(a) == len(b) and len(a) >= 2:
            p = random.randint(1, len(a) - 1)
            return np.append(a[0:p], b[p:]), np.append(b[0:p], a[p:])

    # mutation of single values within the sequence; change subset number to a new one (it can still happen that the same own
    # is chosen randomly again
    def mutation(self, chromosome, probability=0.65):  # make it not so easy for a mutation to occur
        for i in range(len(chromosome)):
            if random.uniform(0, 1) <= probability:
                chromosome[i] = chromosome[i]
            else:
                new = random.randrange(0, self.n_subsets)  # random new subset number
                chromosome[i] = new
        return chromosome

    # develop the next generation/population of chromosomes
    def next_generation(self):
        # create next generation with selecting parents from current generation
        next_generation = np.zeros(len(self.population[0]))
        for i in range(int(len(self.population) / 2)):
            parents = self.select_pair()
            offspring_a, offspring_b = self.single_point_crossover(parents[0], parents[1])
            offspring_a = self.mutation(offspring_a)
            offspring_b = self.mutation(offspring_b)
            next_generation = np.vstack((next_generation, offspring_a))
            next_generation = np.vstack((next_generation, offspring_b))
        # if population is odd we add the last element again (which did not get selected)
        if len(self.population) % 2 == 1:
            next_generation = np.vstack((next_generation, self.population[len(self.population) - 1]))
        self.population = next_generation[1:].astype(int)  # update population
        self.calc_fitness_population()  # calculate fitness everytime new after changing population

    # run the evolution process a set number of times = runtime (can be changed in the constructor)
    def run_evolution(self):
        for i in range(1, self.runtime + 1):
            self.next_generation()
            print('Generation {}:'.format(i))
            print(self.population)

    # find the sequence with the best fitness score
    def find_best_solution(self):
        index = np.argmax(self.fitness_population)
        return self.population[index]

    # print solution - subset and its corresponding total weight for an easier comparison
    def print_best_solution(self):
        solution = self.find_best_solution()
        dict = {}
        for i in range(self.n_subsets):
            name = "Subset" + str(i)
            sum = 0
            for index in range(len(solution)):
                if solution[index] == i:
                    sum += self.weight[index]
            dict[name] = sum

        return dict

    def print_optimum_solution(self, optimum_solution):
        solution = optimum_solution
        dict = {}
        for i in range(self.n_subsets):
            name = "Subset" + str(i)
            sum = 0
            for index in range(len(solution)):
                if solution[index] == i:
                    sum += self.weight[index]
            dict[name] = sum

        for i in dict:
            print(f"{i}: sum: {dict[i]}")
        sys.exit(0)

# class end

# main: create instance of the genetic algorithm and print results
# n_population: number of chromosomes in one generation
# n_items: number of items
# n_subsets: number of subsets
# runtime: number of generation
algo = GenericAlgorithm(n_population=10, n_items=15, n_subsets=3, runtime=10)
algo.run_evolution()
print("Best solution found:")
print(algo.find_best_solution())
results = algo.print_best_solution()
for i in results:
    print(f"{i}: sum: {results[i]}")