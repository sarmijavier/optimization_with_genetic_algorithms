import random
import networkx as nx
import os
import matplotlib.pyplot as plt


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
        
        self.NUM_VERTICES = 0 # default value
        self.edges = 0 # default value
        self.solution = []

    
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

    
    def tournament_selection(self, population):
        return max(
            random.sample(population, self.TOURNAMENT_SIZE),
            key=self.fitness
        )

    
    def crossover(self, p1, p2):
        c1, c2 = [], []
        for g1, g2 in zip(p1, p2):
            if random.random() < 0.5:
                c1.append(g1)
                c2.append(g2)
            else:
                c1.append(g2)
                c2.append(g1)
        return c1, c2

    
    def mutation(self, individual):
        for i in range(self.NUM_VERTICES):
            if random.random() < self.MUTATION_RATE:
                individual[i] = random.randint(0, self.NUM_COLORS - 1)
        return individual


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


    def genetic_algorithm(self):
        self.read_col_file()
        population = self.initialize_population(self.NUM_COLORS)
    
        for generation in range(self.GENERATIONS):
            new_population = []
    
            for _ in range(self.POPULATION_SIZE // 2):
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
    
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
    
                new_population.extend([c1, c2])
    
            # Elitism
            population.sort(key=self.fitness, reverse=True)
            elites = population[:self.ELITISM_SIZE]
            new_population[:self.ELITISM_SIZE] = elites
    
            population = new_population
    
            best = max(population, key=self.fitness)
            conflicts = self.count_conflicts(best)
    
            if conflicts == 0:
                print(f"Valid coloring found at generation {generation}")
                print(f"Gen {generation}: best conflicts = {conflicts}")

                
                G = self.build_nx_graph()
                self.draw_colored_graph(G, best)
                self.solution = best
                self.print_solution()
                return
    
            if generation % 50 == 0:
                print(f"Gen {generation}: best conflicts = {conflicts}")
                
        self.solution = max(population, key=self.fitness)
        


