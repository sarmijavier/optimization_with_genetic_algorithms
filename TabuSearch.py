import random
import time
import networkx as nx
import matplotlib.pyplot as plt

class TabuSearch:
    def __init__(self, filename, NUM_COLORS, TABU_LIST_SIZE=10, MAX_ITERATIONS=1000):
        self.filename = filename
        self.NUM_COLORS = NUM_COLORS
        self.TABU_LIST_SIZE = TABU_LIST_SIZE
        self.MAX_ITERATIONS = MAX_ITERATIONS
        
        self.elapsed_time = 0
        self.conflict_history = []


    # DIMACS .col file reader
    def read_col_file(self):
        edges = []
        num_vertices = 0

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
                    edges.append((int(u) - 1, int(v) - 1))

        self.NUM_VERTICES = num_vertices
        self.vertices = list(range(num_vertices))
        self.edges = edges

    #########
    # Objective function
    def objective_function(self, solution):
        """
        Count number of conflicting edges
        """
        conflicts = 0
        for u, v in self.edges:
            if solution[u] == solution[v]:
                conflicts += 1
        return conflicts

    #########
    # Neighborhood generation
    def get_neighbors(self, solution):
        """
        Change the color of ONE vertex
        """
        neighbors = []

        for v in range(self.NUM_VERTICES):
            current_color = solution[v]
            for color in range(self.NUM_COLORS):
                if color != current_color:
                    neighbor = solution[:]
                    neighbor[v] = color
                    neighbors.append(neighbor)

        return neighbors

    #########
    # Initial solution
    def random_initial_solution(self):
        return [
            random.randint(0, self.NUM_COLORS - 1)
            for _ in range(self.NUM_VERTICES)
    
        ]
    def get_conflicting_vertices(self, solution):
        conflicted = set()
        for u, v in self.edges:
            if solution[u] == solution[v]:
                conflicted.add(u)
                conflicted.add(v)
        return list(conflicted)

    
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
        
    def tabu_search(self):
        start_time = time.time()
        
        data = [
            {
                'filename': self.filename,
                'TABU_LIST_SIZE': self.TABU_LIST_SIZE,
                'MAX_ITERATIONS': self.MAX_ITERATIONS,
            }
        ]
        self.read_col_file()
        
        current = self.random_initial_solution()
        best = current[:]
    
        tabu_dict = {} 
        iteration = 0
    
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
    
            conflicted_vertices = self.get_conflicting_vertices(current)
            if not conflicted_vertices:
                break
    
            best_move = None
            best_delta = float('inf')
    
            for v in conflicted_vertices:
                old_color = current[v]
    
                for new_color in range(self.NUM_COLORS):
                    if new_color == old_color:
                        continue
    
                    move = (v, new_color)
    
                    # Try move
                    current[v] = new_color
                    new_conflicts = self.objective_function(current)
                    delta = new_conflicts - self.objective_function(current)
    
                    # Aspiration
                    is_tabu = move in tabu_dict and tabu_dict[move] > iteration
                    if (not is_tabu) or new_conflicts < self.objective_function(best):
                        if new_conflicts < best_delta:
                            best_delta = new_conflicts
                            best_move = (v, old_color, new_color)
    
                    current[v] = old_color
    
            if best_move is None:
                break
    
            v, old_color, new_color = best_move
            current[v] = new_color
    
            # Add tabu
            tabu_dict[(v, old_color)] = iteration + random.randint(5, 15)
    
            if self.objective_function(current) < self.objective_function(best):
                best = current[:]
    
            if iteration % 100 == 0:
                self.conflict_history.append(
                    [iteration, self.objective_function(current)]
                )
                print(
                    f"Iter {iteration} | "
                    f"Current conflicts: {self.objective_function(current)} | "
                    f"Best: {self.objective_function(best)}"
                )
                
        end_time = time.time()
        self.elapsed_time = end_time - start_time
        data.append(self.elapsed_time)
        data.append(self.NUM_COLORS)
        data.append(self.objective_function(current))
        data.append(self.conflict_history)
        data.append(best)    
        return data

