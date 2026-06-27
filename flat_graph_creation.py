import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle

# --- CONFIGURABLE FILTERS (For full graph generation & file naming) ---
MIN_BOOKS_PER_USER = 5
MIN_READERS_PER_BOOK = 10

# --- VISUALIZATION LIMITS (Only for display purposes) ---
DISPLAY_TOP_BOOKS = 20  # Display only the top N most central books


def load_or_compute_graph(b_per_u, r_per_b):
    """Reloads the full graph from disk if parameters match, otherwise computes it."""
    # Removed CR entirely from the filename structure
    filename = f"flat_graph_B{b_per_u}_R{r_per_b}.pkl"

    if os.path.exists(filename):
        print(f">>> Loading full cached flattened graph from {filename}...")
        with open(filename, 'rb') as f:
            return pickle.load(f)

    print(">>> No cache found. Computing full graph from scratch...")

    # 1. Load Data
    try:
        ratings = pd.read_csv('ratings.csv')
        books = pd.read_csv('books.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return None

    # 2. Successive Filtering Pipeline
    user_counts = ratings['User-ID'].value_counts()
    active_users = user_counts[user_counts >= b_per_u].index
    df_active = ratings[ratings['User-ID'].isin(active_users)]

    book_counts = df_active['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= r_per_b].index
    df_final = df_active[df_active['ISBN'].isin(popular_books)].copy()

    if df_final.empty:
        print("Error: No data remains after filtering with these parameters.")
        return None

    print(f"Filtered Data: {df_final['ISBN'].nunique()} books, {df_final['User-ID'].nunique()} users.")

    # 3. Vectorized Co-occurrence & Average Rating Calculation
    user_item_binary = pd.crosstab(df_final['User-ID'], df_final['ISBN'])
    co_counts = user_item_binary.T.dot(user_item_binary)

    user_item_ratings = df_final.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
    user_item_ratings = user_item_ratings[user_item_binary.columns]

    rating_sum_matrix = (user_item_ratings.T.dot(user_item_binary) + user_item_binary.T.dot(user_item_ratings)) / 2

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_rating_matrix = rating_sum_matrix / co_counts
        avg_rating_matrix = avg_rating_matrix.fillna(0)

    # Zero out self-loops
    np.fill_diagonal(co_counts.values, 0)
    np.fill_diagonal(avg_rating_matrix.values, 0)

    # 4. Graph Construction (Keeps all valid edges)
    G_full = nx.from_pandas_adjacency(co_counts)

    # FORCEFULLY REMOVE ANY SELF-LOOPS
    G_full.remove_edges_from(nx.selfloop_edges(G_full))

    # Pack the average rating values into edge dictionaries safely
    for u, v in list(G_full.edges()):
        G_full[u][v]['avg_rating'] = float(avg_rating_matrix.at[u, v])

    # Remove completely isolated nodes if they exist
    G_full.remove_nodes_from(list(nx.isolates(G_full)))

    # 5. Label with Titles
    isbn_to_title = dict(zip(books['ISBN'], books['Book-Title']))
    G_full = nx.relabel_nodes(G_full, {isbn: isbn_to_title.get(isbn, isbn) for isbn in G_full.nodes()})

    # Kill the phantom self-loops created by ISBN-to-Title collisions
    G_full.remove_edges_from(nx.selfloop_edges(G_full))

    # 6. Save Complete Graph Array
    if G_full.number_of_edges() > 0:
        with open(filename, 'wb') as f:
            pickle.dump(G_full, f)
            print(f">>> Full flattened graph saved as {filename}")

    return G_full


# --- Execution ---
G_all = load_or_compute_graph(MIN_BOOKS_PER_USER, MIN_READERS_PER_BOOK)

if G_all and G_all.number_of_edges() > 0:
    print(f">>> Extracting top {DISPLAY_TOP_BOOKS} books for visualization...")

    # 1. Sort books by degree (number of connections) to discover the most active/central hubs
    sorted_books = sorted(G_all.degree(), key=lambda x: x[1], reverse=True)
    top_display_books = [book for book, degree in sorted_books[:DISPLAY_TOP_BOOKS]]

    # 2. Induce the display subgraph safely
    G_display = G_all.subgraph(top_display_books).copy()

    # --- Render Visualization Slice ---
    plt.figure(figsize=(15, 11))
    pos = nx.spring_layout(G_display, k=0.6, seed=42)

    # SAFELY get edge data using .edges(data=True) to prevent KeyErrors
    display_edges = G_display.edges(data=True)

    if len(display_edges) > 0:
        weights_arr = np.array([d.get('weight', 1.0) for u, v, d in display_edges])
        ratings_arr = np.array([d.get('avg_rating', 0.0) for u, v, d in display_edges])

        max_w = weights_arr.max() if weights_arr.size > 0 else 1
        widths = (weights_arr / max_w) * 6
    else:
        widths = []
        ratings_arr = []

    # Plot Layout Nodes
    nx.draw_networkx_nodes(G_display, pos, node_color='lightgray', node_size=1200, alpha=0.8)
    nx.draw_networkx_labels(G_display, pos, font_size=8, font_weight='bold')

    # Draw Colored Edge Arrays
    if len(ratings_arr) > 0:
        nx.draw_networkx_edges(G_display, pos,
                               width=widths,
                               edge_color=ratings_arr,
                               edge_cmap=plt.cm.RdYlGn,
                               edge_vmax=10.0,
                               edge_vmin=0.0,
                               alpha=0.7)

        # Scale Metric UI colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=10))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_label('Average Shared Book Rating (0 - 10)', fontsize=10, fontweight='bold')

    plt.title(f"Flattened Book Network Slice (Displaying Top {DISPLAY_TOP_BOOKS} Central Books)\n"
              f"Line Thickness = Co-readers | Line Color = Avg Rating", fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Graph calculation processing failed.")