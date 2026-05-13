import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from community import community_louvain  # Requires 'python-louvain' library

# --- CONFIGURATION (Must match the graph you want to load) ---
MIN_BOOKS_PER_USER = 5
MIN_READERS_PER_BOOK = 100
MIN_CO_READERS = 20


def analyze_graph(G):
    """Performs statistical analysis and creates cleaner visualizations."""
    print(">>> Starting Graph Analysis...")

    # 1. Basic Network Statistics
    density = nx.density(G)
    communities = community_louvain.best_partition(G)
    num_communities = len(set(communities.values()))
    # Check if the entire graph is a single connected component
    is_connected = nx.is_connected(G)
    print(f"Is the graph fully connected? {is_connected}")
    if not is_connected:
        # Count how many separate 'islands' (components) there are
        num_components = nx.number_connected_components(G)
        print(f"The graph has {num_components} separate components.")

    print(f"--- Statistics ---")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Network Density: {density:.4f}")
    print(f"Detected Communities: {num_communities}")

    # 2. Centrality (The most 'important' books)
    # Using degree centrality to find books with the most connections
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n--- Top 10 Most Central Books ---")
    for book, score in top_nodes:
        print(f"{score:.3f}: {book}")

    # 3. Visualization 1: Community Detection Map
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.3, seed=42)  # k controls distance between nodes

    # Color nodes by community
    cmap = plt.get_cmap('viridis')
    node_colors = [communities[node] for node in G.nodes()]

    # Scale node size by centrality
    node_sizes = [v * 10000 for v in centrality.values()]

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Draw edges with varying transparency based on weight
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    max_w = weights.max() if weights.size > 0 else 1
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=(weights / max_w) * 5, edge_color='gray')

    plt.title(f"Book Communities & Centrality (Nodes: {G.number_of_nodes()})")
    plt.axis('off')
    plt.show()

    # 4. Visualization 2: Degree Distribution Histogram
    plt.figure(figsize=(10, 6))
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=20, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution (Number of Connections per Book)")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    filename = f"book_graph_B{MIN_BOOKS_PER_USER}_R{MIN_READERS_PER_BOOK}_CR{MIN_CO_READERS}.pkl"

    if os.path.exists(filename):
        print(f">>> Loading graph: {filename}")
        with open(filename, 'rb') as f:
            G_loaded = pickle.load(f)

        analyze_graph(G_loaded)
    else:
        print(f"Error: {filename} not found. Please run basic_graph_creation.py first.")