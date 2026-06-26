import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle

# --- CONFIGURABLE FILTERS (For full graph generation) ---
MIN_BOOKS_PER_USER = 5
MIN_READERS_PER_BOOK = 100
MIN_CO_READERS = 0  # CR placeholder to keep the exact same filename structure

# --- VISUALIZATION LIMITS (Only for display purposes) ---
DISPLAY_TOP_USERS = 7
DISPLAY_TOP_BOOKS = 20


def load_or_compute_bipartite(b_per_u, r_per_b, co_r):
    """Computes and saves the complete Bipartite graph across the whole dataset."""
    filename = f"bipartite_graph_B{b_per_u}_R{r_per_b}_CR{co_r}.pkl"

    if os.path.exists(filename):
        print(f">>> Loading complete cached bipartite graph from {filename}...")
        with open(filename, 'rb') as f:
            return pickle.load(f)

    print(">>> No cache found. Computing full bipartite graph from scratch...")

    # 1. Load Data
    try:
        ratings = pd.read_csv('ratings.csv')
        books = pd.read_csv('books.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return None

    # 2. Full Successive Filtering Pipeline
    user_counts = ratings['User-ID'].value_counts()
    active_users = user_counts[user_counts >= b_per_u].index
    df_active = ratings[ratings['User-ID'].isin(active_users)]

    book_counts = df_active['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= r_per_b].index
    df_final = df_active[df_active['ISBN'].isin(popular_books)].copy()

    if df_final.empty:
        print("Error: No data remains after filtering.")
        return None

    print(f"Full Dataset Filtered: {df_final['ISBN'].nunique()} books, {df_final['User-ID'].nunique()} users.")

    # Map ISBN to Book Title
    isbn_to_title = dict(zip(books['ISBN'], books['Book-Title']))
    df_final['Book-Title'] = df_final['ISBN'].map(lambda x: isbn_to_title.get(x, x))

    # 3. Construct the Full Bipartite Graph
    B_full = nx.Graph()

    unique_users = df_final['User-ID'].unique()
    unique_books = df_final['Book-Title'].unique()

    B_full.add_nodes_from(unique_users, bipartite=0)
    B_full.add_nodes_from(unique_books, bipartite=1)

    # Add all rows as edges with their ratings
    edges = [(row['User-ID'], row['Book-Title'], {'rating': float(row['Book-Rating'])})
             for _, row in df_final.iterrows()]
    B_full.add_edges_from(edges)

    # 4. Save the full graph
    if B_full.number_of_edges() > 0:
        with open(filename, 'wb') as f:
            pickle.dump(B_full, f)
            print(f">>> Full Bipartite graph saved as {filename}")

    return B_full


# --- Execution ---
B_all = load_or_compute_bipartite(MIN_BOOKS_PER_USER, MIN_READERS_PER_BOOK, MIN_CO_READERS)

if B_all and B_all.number_of_edges() > 0:
    print(">>> Extracting a clean visualization slice from the full graph...")

    # 1. Identify top users by looking at their degree in the FULL graph
    all_users = [n for n, d in B_all.nodes(data=True) if d['bipartite'] == 0]
    user_degrees = sorted([(u, B_all.degree(u)) for u in all_users], key=lambda x: x[1], reverse=True)
    top_display_users = [u for u, deg in user_degrees[:DISPLAY_TOP_USERS]]

    # 2. Identify the books most commonly connected to those specific top users
    book_connection_counts = {}
    for u in top_display_users:
        for book in B_all.neighbors(u):
            book_connection_counts[book] = book_connection_counts.get(book, 0) + 1

    top_display_books = sorted(book_connection_counts.items(), key=lambda x: x[1], reverse=True)[:DISPLAY_TOP_BOOKS]
    top_display_books = [b[0] for b in top_display_books]

    # 3. Induce the tiny subgraph containing only our display candidates
    display_nodes = top_display_users + top_display_books
    B_display = B_all.subgraph(display_nodes).copy()

    # --- Render Pipeline ---
    plt.figure(figsize=(14, 11))

    # Use the induced sublist for layout alignment
    pos = nx.bipartite_layout(B_display, top_display_users)
    edge_ratings = np.array([B_display[u][v]['rating'] for u, v in B_display.edges()])

    # Layer Node Plots
    nx.draw_networkx_nodes(B_display, pos, nodelist=top_display_users, node_color='skyblue', node_size=600, alpha=0.9)
    nx.draw_networkx_nodes(B_display, pos, nodelist=top_display_books, node_color='salmon', node_size=600, alpha=0.9)

    # Offset text tracking labels
    label_pos = {node: (coords[0] - 0.06, coords[1]) if node in top_display_users else (coords[0] + 0.06, coords[1])
                 for node, coords in pos.items()}
    nx.draw_networkx_labels(B_display, label_pos, font_size=8, font_weight='bold')

    # Color connection paths by raw ratings
    if len(edge_ratings) > 0:
        edges = nx.draw_networkx_edges(B_display, pos,
                                       edge_color=edge_ratings,
                                       edge_cmap=plt.cm.RdYlGn,
                                       edge_vmin=0.0, edge_vmax=10.0,
                                       width=1, alpha=0.7)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=10))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', pad=0.05, shrink=0.5)
        cbar.set_label('User Rating Score (0 - 10)', fontsize=10, fontweight='bold')

    plt.title(
        f"Filtered Bipartite Render Slice (Displaying Top {DISPLAY_TOP_USERS} Users x {DISPLAY_TOP_BOOKS} Books)\n"
        f"Source Cache File Contains Entire Filtered Base Matrix Node Network Structure",
        fontsize=11, fontweight='bold')

    plt.xlim(-1.6, 1.6)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Graph calculation processing failed.")