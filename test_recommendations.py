import os
import pandas as pd
import numpy as np
import torch

from lightCGN_recommendation import (
    load_and_prep_graph,
    create_sparse_adjacency,
    LightGCNRegression,
    default_weights_path,
    GRAPH_FILENAME,
    EMBEDDING_DIM,
    NUM_LAYERS
)
from neighbors_recommendation import build_rec_filename, load_neighbor_rec_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_algorithms():
    print("Loading test data and graph...")
    try:
        df_test = pd.read_csv('test_set.csv')
    except FileNotFoundError:
        print("Error: test_set.csv not found. Run split_data.py first.")
        return

    # Load graph mappings to verify which test users/books survived the training filters
    num_users, num_books, edges, ratings, user2id, book2id = load_and_prep_graph(GRAPH_FILENAME)
    print(f"\nFiltered Graph Size: {num_users} Users, {num_books} Books")

    # --- Initialize LightGCN ---
    weights_path = default_weights_path()
    if not os.path.exists(weights_path):
        print("Error: LightGCN weights not found. Retrain the model on the new ratings.csv.")
        return

    global_mean = float(np.mean(ratings)) if len(ratings) > 0 else 0.0
    model = LightGCNRegression(
        num_users=num_users, num_books=num_books, embed_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, global_mean=global_mean
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # Precompute all graph embeddings once to vastly speed up the loop
    sparse_adj = create_sparse_adjacency(num_users, num_books, edges).to(DEVICE)
    with torch.no_grad():
        users_final, books_final = model(sparse_adj)

    # --- Initialize Neighbors ---
    rec_matrix_path = build_rec_filename(GRAPH_FILENAME)
    if not os.path.exists(rec_matrix_path):
        print("Error: Neighbor matrix not found. Rebuild it using run_recommendations.py.")
        return
    neighbor_matrix = load_neighbor_rec_matrix(rec_matrix_path)

    # --- Evaluation Tracking ---
    lightgcn_errors = []
    neighbor_errors = []
    skipped_users = 0
    skipped_books = 0

    # Build a quick lookup of books seen during training so they don't push the test book down the list
    seen_books_map = {u: [] for u in range(num_users)}
    for u, b in edges:
        seen_books_map[u].append(b)

    print("\nStarting evaluation loop...")

    for _, row in df_test.iterrows():
        orig_u = row['User-ID']
        orig_b = row['ISBN']
        expected_pct = row['Expected-Percentile']

        # Resolve user mapping safely
        u_key = orig_u
        if u_key not in user2id:
            try:
                alt = int(orig_u); u_key = alt if alt in user2id else u_key
            except Exception:
                pass
        if u_key not in user2id:
            try:
                alt = str(orig_u); u_key = alt if alt in user2id else u_key
            except Exception:
                pass

        if u_key not in user2id:
            skipped_users += 1
            continue

        if orig_b not in book2id:
            skipped_books += 1
            continue

        u_idx = user2id[u_key]
        b_idx = book2id[orig_b]

        expected_rank = expected_pct * num_books
        seen_idx = seen_books_map[u_idx]

        # --- Score LightGCN ---
        books_tensor = torch.arange(num_books, device=DEVICE, dtype=torch.long)
        users_tensor = torch.full((num_books,), u_idx, device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            l_scores = model.predict(users_tensor, books_tensor, users_final, books_final).cpu().numpy()

        l_scores[seen_idx] = -np.inf
        # Sort descending
        l_ranked_indices = l_scores.argsort()[::-1]
        l_rank = np.where(l_ranked_indices == b_idx)[0][0] + 1
        lightgcn_errors.append(abs(l_rank - expected_rank))

        # --- Score Neighbors ---
        n_scores = neighbor_matrix[u_idx].copy()
        n_scores[seen_idx] = -np.inf
        n_scores[np.isnan(n_scores)] = -np.inf

        n_ranked_indices = n_scores.argsort()[::-1]
        n_rank = np.where(n_ranked_indices == b_idx)[0][0] + 1
        neighbor_errors.append(abs(n_rank - expected_rank))

    # --- Print Results ---
    print("\n--- Final Evaluation Results ---")
    print(f"Total test cases processed: {len(lightgcn_errors)}")
    print(f"Cases skipped (User filtered out of graph): {skipped_users}")
    print(f"Cases skipped (Book filtered out of graph): {skipped_books}")

    if len(lightgcn_errors) == 0:
        print("No valid test cases remained after graph filtering.")
        return

    mae_l = np.mean(lightgcn_errors)
    mae_n = np.mean(neighbor_errors)

    print(f"\nMean Absolute Error (in Rank Positions out of {num_books} books):")
    print(f"  LightGCN Algorithm : {mae_l:.2f} positions off")
    print(f"  Neighbors Algorithm: {mae_n:.2f} positions off")

    print(f"\nError as a % of Catalog Size:")
    print(f"  LightGCN Algorithm : {(mae_l / num_books) * 100:.2f}%")
    print(f"  Neighbors Algorithm: {(mae_n / num_books) * 100:.2f}%")


if __name__ == "__main__":
    test_algorithms()