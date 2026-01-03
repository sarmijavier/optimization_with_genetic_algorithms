import random
import networkx as nx
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

class GeneticAlgorithm:
    def __init__(
        self, 
        filename = None, 
        NUM_COLORS = 0 ,
        POPULATION_SIZE = 100,
        GENERATIONS = 1000,
        MUTATION_RATE = 0.06,
        ELITISM_SIZE = 3,
        TOURNAMENT_SIZE = 3,
        SELECTION_METHOD = 'tournament',
        CROSSOVER_METHOD = 'uniform',
        MUTATE_METHOD = 'independent',
    ):
        self.filename = filename
        # This is our k parameter, to choose the min num of colors 
        self.NUM_COLORS = NUM_COLORS        

        # Parameters, default? 
        self.POPULATION_SIZE = POPULATION_SIZE
        self.GENERATIONS = GENERATIONS
        self.MUTATION_RATE = MUTATION_RATE
        self.ELITISM_SIZE = ELITISM_SIZE
        self.TOURNAMENT_SIZE = TOURNAMENT_SIZE
        self.SELECTION_METHOD = SELECTION_METHOD
        self.CROSSOVER_METHOD = CROSSOVER_METHOD
        self.MUTATE_METHOD = MUTATE_METHOD
        
        self.NUM_VERTICES = 0 # default value
        self.edges = 0 # default value
        self.solution = []

        self.elapsed_time = 0
        self.generations_conflicts = []

    
    # DIMACS .col file reader
    def read_col_file(self):
        edges = []
        num_vertices = 0

        if not self.filename:
            print('Please provide a valid filename')
            return
    
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    parts = line.split()
                    num_vertices = int(parts[2])
                elif line.startswith('e'):
                    _, u, v = line.split()
                    # Convert to 0-based indexing
                    edges.append((int(u) - 1, int(v) - 1))
    
        self.NUM_VERTICES = num_vertices
        self.edges = edges
        
        
    def initialize_population(self, num_colors):
        return [
            [random.randint(0, num_colors - 1) for _ in range(self.NUM_VERTICES)]
            for _ in range(self.POPULATION_SIZE)
        ]

    
    # checks is a solution is fulfilled
    def count_conflicts(self, individual):
        return sum(1 for u, v in self.edges if individual[u] == individual[v])
    
    def fitness(self, individual):
        return 1 / (1 + self.count_conflicts(individual))

    #########
    # SELECTION METHODS
    def roulette_wheel_selection(self, population, fitness_values):
        scores = self.fitness_to_score(fitness_values)
        total_score = sum(scores)
    
        pick = random.uniform(0, total_score)
        current = 0.0
    
        for individual, score in zip(population, scores):
            current += score
            if current >= pick:
                return individual
    

    def rank_selection(self, population, fitness_values):
        # Sort by fitness (ascending = best first)
        sorted_pop = sorted(
            zip(population, fitness_values),
            key=lambda x: x[1]
        )
    
        n = len(sorted_pop)
        ranks = list(range(n, 0, -1))  # best gets highest rank
        total_rank = sum(ranks)
    
        pick = random.uniform(0, total_rank)
        current = 0
    
        for (individual, _), rank in zip(sorted_pop, ranks):
            current += rank
            if current >= pick:
                return individual

    def fitness_to_score(self, fitness_values):
        max_f = max(fitness_values)
        return [(max_f - f) + 1e-6 for f in fitness_values]
    
    def stochastic_universal_sampling(self, population, fitness_values, num_parents):
        scores = self.fitness_to_score(fitness_values)
        total_score = sum(scores)
        step = total_score / num_parents
        start = random.uniform(0, step)
    
        points = [start + i * step for i in range(num_parents)]
    
        parents = []
        cumulative = 0.0
        i = 0
    
        for individual, score in zip(population, scores):
            cumulative += score
            while i < num_parents and cumulative >= points[i]:
                parents.append(individual)
                i += 1
    
        return parents

        
    def tournament_selection(self, population):
        return max(
            random.sample(population, self.TOURNAMENT_SIZE),
            key=self.fitness
        )

                
    def selection_schemes(self, population, fitness_values):
        if self.SELECTION_METHOD == "roulette":
            return self.roulette_wheel_selection(population, fitness_values)
        elif self.SELECTION_METHOD == "rank":
            return self.rank_selection(population, fitness_values)
        elif self.SELECTION_METHOD == "sus":
            return self.stochastic_universal_sampling(population, fitness_values, 1)[0]
        elif self.SELECTION_METHOD == 'tournament':
            return self.tournament_selection(population)
        else:
            print('Invalid selection method')

    #########
    # CROSSOVER METHODS
    
    def uniform_crossover(self, p1, p2):
        c1, c2 = [], []
        for g1, g2 in zip(p1, p2):
            if random.random() < 0.5:
                c1.append(g1)
                c2.append(g2)
            else:
                c1.append(g2)
                c2.append(g1)
        return c1, c2


    def two_point_crossover(self, p1, p2):
        n = len(p1)
        p1_idx, p2_idx = sorted(random.sample(range(n), 2))
    
        c1 = (
            p1[:p1_idx]
            + p2[p1_idx:p2_idx]
            + p1[p2_idx:]
        )
    
        c2 = (
            p2[:p1_idx]
            + p1[p1_idx:p2_idx]
            + p2[p2_idx:]
        )
    
        return c1, c2
    
    def crossover(self, p1, p2):
        if self.CROSSOVER_METHOD == "two_point":
            return self.two_point_crossover(p1, p2)
        elif self.CROSSOVER_METHOD == "uniform":
            return self.uniform_crossover(p1, p2)
        else:
            raise ValueError("Invalid crossover method")

    #########
    # MUTATION METHODS
    def mutate_one_gene(self, chromosome):
        """
        Always mutates exactly one gene (node).
        """
        n = len(chromosome)
        i = random.randrange(n)
    
        old_color = chromosome[i]
        new_color = random.choice([c for c in range(self.NUM_COLORS) if c != old_color])
    
        chromosome[i] = new_color
        return chromosome
        
    def mutate_independent(self, individual):
        for i in range(self.NUM_VERTICES):
            if random.random() < self.MUTATION_RATE:
                old_color = individual[i]
                individual[i] = random.choice(
                    [c for c in range(self.NUM_COLORS) if c != old_color]
                )
        return individual
        
    def mutate(self, chromosome):
        if self.MUTATE_METHOD == "one":
            return self.mutate_one_gene(chromosome)
        elif self.MUTATE_METHOD == "independent":
            return self.mutate_independent(chromosome)
        else:
            raise ValueError("Invalid mutation method")

    def compute_fitness(self, individual, edges):
        conflicts = 0
        for u, v in edges:
            if individual[u] == individual[v]:
                conflicts += 1
        return conflicts
    
    # draw graph using networkx
    def build_nx_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.NUM_VERTICES))
        G.add_edges_from(self.edges)
        return G
    
        
    def draw_colored_graph(self, G, coloring):
        pos = nx.spring_layout(G, seed=42)
    
        node_colors = [coloring[node] for node in G.nodes()]
    
        labels = {node: node + 1 for node in G.nodes()}  # <-- fix labels
    
        plt.figure(figsize=(7, 7))
        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_color=node_colors,
            cmap=plt.cm.tab20,
            node_size=700,
            font_size=10
        )
        plt.show()


    def print_solution(self):
        print("Best coloring found:")
        print(self.solution)
        print("Conflicts:", self.count_conflicts(self.solution))
        print("Number of Colors", set(self.solution))
        print(f"The program took {self.elapsed_time} seconds to run.")
        #self.plot_conflicts()


    def plot_conflicts(self):
        if not self.generations_conflicts:
            print("No data to plot.")
            return
    
        # 1. Extract data
        gens, confs = zip(*self.generations_conflicts)
    
        # 2. Create the plot
        plt.style.use('seaborn-v0_8-darkgrid') # Optional: makes it look modern
        fig, ax = plt.subplots()
        
        ax.plot(gens, confs, label='Fitness (Conflicts)', color='#2ca02c', linewidth=2)
        
        # 3. Add details
        ax.set_title('Evolutionary Progress', fontsize=14)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Conflict Score')
        ax.legend()
    
        plt.show()

    def genetic_algorithm(self):
        start_time = time.time()
        
        self.read_col_file()
        population = self.initialize_population(self.NUM_COLORS)

    
        
        data = [
            {
                'filename': self.filename,
                'POPULATION_SIZE': self.POPULATION_SIZE,
                'GENERATIONS': self.GENERATIONS,
                'MUTATION_RATE': self.MUTATION_RATE,
                'ELITISM_SIZE': self.ELITISM_SIZE,
                'TOURNAMENT_SIZE': self.TOURNAMENT_SIZE,
                'SELECTION_METHOD': self.SELECTION_METHOD,
                'CROSSOVER_METHOD': self.CROSSOVER_METHOD,
                'MUTATE_METHOD': self.MUTATE_METHOD,
            }
        ]
        
        for generation in range(self.GENERATIONS):
            new_population = []

            fitness_values = [
                self.compute_fitness(individual, self.edges)
                for individual in population
            ]
    
            for _ in range(self.POPULATION_SIZE // 2):
                p1 = self.selection_schemes(population, fitness_values)
                p2 = self.selection_schemes(population, fitness_values)
    
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
    
                new_population.extend([c1, c2])
    
            # Elitism
            population.sort(key=self.fitness, reverse=True)
            elites = population[:self.ELITISM_SIZE]
            new_population[:self.ELITISM_SIZE] = elites
            
            population = new_population
            best = max(population, key=self.fitness)
            conflicts = self.count_conflicts(best)

            self.generations_conflicts.append([generation, conflicts])
            if conflicts == 0:
                print(f"Valid coloring found at generation {generation}")
                print(f"Gen {generation}: best conflicts = {conflicts}")

                
                #G = self.build_nx_graph()
                #self.draw_colored_graph(G, best)
                self.solution = best
                self.print_solution()
                
                end_time = time.time()
                self.elapsed_time = end_time - start_time
                data.append(self.elapsed_time)
                data.append(self.NUM_COLORS)
                data.append(conflicts)
                data.append(generation)
                data.append(self.generations_conflicts)
                data.append(best)
                return data
    
            if generation % 50 == 0:
                print(f"Gen {generation}: best conflicts = {conflicts}")
                
        self.solution = max(population, key=self.fitness)
        end_time = time.time()
        self.elapsed_time = end_time - start_time
        data.append('NO_SOLUTION')
        data.append(self.NUM_COLORS)
        data.append(conflicts)
        data.append(generation)
        return data
