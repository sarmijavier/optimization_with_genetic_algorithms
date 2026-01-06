import random
import os
import math
from collections import defaultdict
import networkx as nx
import time
import matplotlib.pyplot as plt

class SimulatedAnnealing:
    def __init__(
        self,
        filename = '',
        NUM_COLORS = 0,
        INITIAL_TEMPERATURE = 10.0,
        COOLING_RATE = 0.995,
        MIN_TEMPERATURE = 1e-3,
        MAX_ITERATIONS = 100_000
    ):
        self.filename = filename
        self.NUM_COLORS = NUM_COLORS
        self.INITIAL_TEMPERATURE = INITIAL_TEMPERATURE
        self.COOLING_RATE = COOLING_RATE
        self.MIN_TEMPERATURE = MIN_TEMPERATURE
        self.MAX_ITERATIONS = MAX_ITERATIONS

        self.NUM_VERTICES = 0
        self.vertices = []      
        self.edges = []
        self.conflict_history = []
        self.log_interval = 50
        
    # DIMACS .col file reader
    def read_col_file(self):
        edges = []
        num_vertices = 0
    
        if not self.filename:
            raise ValueError("Please provide a valid filename")
    
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
                    edges.append((int(u) - 1, int(v) - 1))  # 0-based
    
        self.NUM_VERTICES = num_vertices
        self.vertices = list(range(num_vertices)) 
        self.edges = edges

    
    #########
    # Energy function
    def energy(self, coloring: dict) -> int:
        conflicts = 0
        for u, v in self.edges:
            if coloring[u] == coloring[v]:
                conflicts += 1
        return conflicts

    #########
    # Random initial solution
    def initial_solution(self) -> dict:
        return {
            node: random.randint(0, self.NUM_COLORS - 1)
            for node in self.vertices    
        }

    #########
    # Neighborhood definition
    def random_neighbor(self, coloring: dict) -> dict:
        neighbor = coloring.copy()
        node = random.choice(self.vertices) 
    
        new_color = random.randint(0, self.NUM_COLORS - 1)
        while new_color == neighbor[node]:
            new_color = random.randint(0, self.NUM_COLORS - 1)
    
        neighbor[node] = new_color
        return neighbor

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
    
    def plot_conflicts(self):
        if not self.conflict_history:
            print("No data to plot.")
            return
    
        # 1. Extract data
        gens, confs = zip(*self.conflict_history)
    
        # 2. Create the plot
        plt.style.use('seaborn-v0_8-darkgrid') # Optional: makes it look modern
        fig, ax = plt.subplots()
        
        ax.plot(gens, confs, label='Energy (Number of Conflicts)', linewidth=2)
        
        # 3. Add details    
        ax.set_title('Simulated Annealing Convergence', fontsize=14)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Conflicts)')
        ax.legend()
        plt.show()
    
    def coloring_dict_to_list(self, coloring):
        """
        Converts {node: color} dict to a list indexed by node.
        """
        return [coloring[node] for node in range(self.NUM_VERTICES)]

    def simulated_annealing(self):
        start_time = time.time()
        
        data = [
            {
                'filename': self.filename,
                'INITIAL_TEMPERATURE': self.INITIAL_TEMPERATURE,
                'COOLING_RATE': self.COOLING_RATE,
                'MIN_TEMPERATURE': self.MIN_TEMPERATURE,
                'MAX_ITERATIONS': self.MAX_ITERATIONS,
            }
        ]
        
        self.read_col_file()
        
        current = self.initial_solution()
        current_energy = self.energy(current)

        best = current.copy()
        best_energy = current_energy

        iteration = 0

        while self.INITIAL_TEMPERATURE > self.MIN_TEMPERATURE and iteration < self.MAX_ITERATIONS:
            neighbor = self.random_neighbor(current)
            neighbor_energy = self.energy(neighbor)
        
            delta = neighbor_energy - current_energy
        
            if delta <= 0:
                current = neighbor
                current_energy = neighbor_energy
            else:
                if random.random() < math.exp(-delta / self.INITIAL_TEMPERATURE):
                    current = neighbor
                    current_energy = neighbor_energy
        
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
            #########
            # LOG CONFLICTS
            if iteration % self.log_interval == 0:
                self.conflict_history.append(
                    [iteration, current_energy]
                )
                print(
                    f"Iter {iteration:6d} | "
                    f"T={self.INITIAL_TEMPERATURE:.4f} | "
                    f"Current={current_energy} | "
                    f"Best={best_energy}"
                )
        
            self.INITIAL_TEMPERATURE *= self.COOLING_RATE
            iteration += 1
        
            if best_energy == 0:
                print(
                    f"0 conflicts reached |"
                    f"Iter {iteration:6d} | "
                    f"T={self.INITIAL_TEMPERATURE:.4f} | "
                    f"Current={current_energy} | "
                    f"Best={best_energy}"
                )
                break
                
        end_time = time.time()
        self.elapsed_time = end_time - start_time
        data.append(self.elapsed_time)
        data.append(self.NUM_COLORS)
        data.append(best_energy)
        data.append(self.conflict_history)
        data.append(self.coloring_dict_to_list(best))
        return data
