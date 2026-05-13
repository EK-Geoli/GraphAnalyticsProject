import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle

# --- CONFIGURABLE FILTERS ---
MIN_BOOKS_PER_USER = 5
MIN_READERS_PER_BOOK = 100
MIN_CO_READERS = 20


def load_or_compute_graph(b_per_u, r_per_b, co_r):
    """Reloads the graph from disk if parameters match, otherwise computes it."""
    # Updated naming convention: B{b_per_u}_R{r_per_b}_CR{co_r}
    filename = f"book_graph_B{b_per_u}_R{r_per_b}_CR{co_r}.pkl"

    if os.path.exists(filename):
        print(f">>> Loading cached graph from {filename}...")
        with open(filename, 'rb') as f:
            return pickle.load(f)

    print(">>> No cache found. Computing from scratch...")

    # 1. Load Data
    try:
        ratings = pd.read_csv('ratings.csv')
        books = pd.read_csv('books.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return None

    # 2. Successive Filtering Pipeline
    # Filter A: Active Readers
    user_counts = ratings['User-ID'].value_counts()
    active_users = user_counts[user_counts >= b_per_u].index
    df_active = ratings[ratings['User-ID'].isin(active_users)]

    # Filter B: Popular Books (Successive)
    book_counts = df_active['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= r_per_b].index
    df_final = df_active[df_active['ISBN'].isin(popular_books)]

    if df_final.empty:
        print("Error: No data remains after filtering with these parameters.")
        return None

    # Stats Output
    print(f"Filtered Data: {df_final['ISBN'].nunique()} books, {df_final['User-ID'].nunique()} users.")

    # 3. Vectorized Matrix Multiplication
    user_item_matrix = pd.crosstab(df_final['User-ID'], df_final['ISBN'])
    co_occurrence = user_item_matrix.T.dot(user_item_matrix)
    np.fill_diagonal(co_occurrence.values, 0)

    # 4. Graph Construction and Edge Weight Filtering
    G = nx.from_pandas_adjacency(co_occurrence)

    # Filter C: Connection Strength (CR)
    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < co_r]
    G.remove_edges_from(weak_edges)
    G.remove_nodes_from(list(nx.isolates(G)))

    # 5. Label with Titles
    isbn_to_title = dict(zip(books['ISBN'], books['Book-Title']))
    G = nx.relabel_nodes(G, {isbn: isbn_to_title.get(isbn, isbn) for isbn in G.nodes()})

    # 6. Save Graph
    with open(filename, 'wb') as f:
        if G and G.number_of_edges() > 0:
            pickle.dump(G, f)
            print(f">>> Graph computed and saved as {filename}")
        else:
            print(">>> Graph is empty after filtering. Not saving.")

    return G


# --- Execution ---
G = load_or_compute_graph(MIN_BOOKS_PER_USER, MIN_READERS_PER_BOOK, MIN_CO_READERS)

if G and G.number_of_edges() > 0:
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    weights_arr = np.array([G[u][v]['weight'] for u, v in G.edges()])

    max_w = weights_arr.max() if weights_arr.size > 0 else 1
    widths = (weights_arr / max_w) * 7

    nx.draw(G, pos, with_labels=True,
            node_color='lightgreen',
            node_size=1200,
            font_size=7,
            width=widths,
            alpha=0.6)
    plt.title(f"Book Network (B:{MIN_BOOKS_PER_USER} R:{MIN_READERS_PER_BOOK} CR:{MIN_CO_READERS})")
    plt.show()
else:
    print("Graph is empty or threshold is too high for visualization.")