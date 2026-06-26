# python
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch

from lightCGN_recommendation import (
    load_and_prep_graph,
    create_sparse_adjacency,
    LightGCNRegression,
    default_weights_path,
)

# New imports for neighbor-based recommender
from neighbors_recommendation import (
    build_neighbor_rec_matrix,
    get_neighbor_recs_for_user,
    build_rec_filename,
)

WEIGHTS_PATH = default_weights_path()
GRAPH_FILENAME = "bipartite_graph_B5_R100_CR0.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_user_key(user2id: dict, user_key):
    """Try several conversions to find the user key present in user2id."""
    if user_key in user2id:
        return user_key
    # try int / str conversions
    try:
        alt = int(user_key)
        if alt in user2id:
            return alt
    except Exception:
        pass
    try:
        alt = str(user_key)
        if alt in user2id:
            return alt
    except Exception:
        pass
    raise KeyError("User not found in graph mapping.")


def get_lightCGN_recommendations_for_existing_user(
    original_user_id,
    top_k: int = 10,
    exclude_seen: bool = True,
    graph_filename: str = GRAPH_FILENAME,
    weights_path: str = WEIGHTS_PATH,
) -> List[Tuple[str, float]]:
    """
    Return top_k recommended book titles and scores for an existing user id
    present in the bipartite graph saved by the preprocessing pipeline.

    Returns a list of (book_title, score) sorted by descending score.
    """
    # Load graph data and mappings
    num_users, num_books, edges, ratings, user2id, book2id = load_and_prep_graph(
        graph_filename
    )

    # Resolve user key as stored in the graph (original node label)
    try:
        user_key = _resolve_user_key(user2id, original_user_id)
    except KeyError:
        raise ValueError("Provided user id not found in graph node mapping.")

    u_idx = user2id[user_key]
    id2book = {v: k for k, v in book2id.items()}

    # Build normalized adjacency and move to device
    sparse_adj = create_sparse_adjacency(num_users, num_books, edges).to(DEVICE)

    # Instantiate model and load weights
    # global mean is approximated by average of provided ratings (same as training logic)
    global_mean = float(np.mean(ratings)) if len(ratings) > 0 else 0.0
    model = LightGCNRegression(
        num_users=num_users,
        num_books=num_books,
        embed_dim=64,
        num_layers=3,
        global_mean=global_mean,
    ).to(DEVICE)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Run training to produce weights."
        )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # Compute final embeddings
    with torch.no_grad():
        users_final, books_final = model(sparse_adj)

    # Prepare tensors for scoring
    books_idx = torch.arange(num_books, device=DEVICE, dtype=torch.long)
    users_idx = torch.full((num_books,), u_idx, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        scores = model.predict(users_idx, books_idx, users_final, books_final).cpu()

    # Optionally exclude already seen books
    if exclude_seen:
        seen = {b for u, b in edges if u == u_idx}
        if seen:
            seen_idx = list(seen)
            scores[seen_idx] = -float("inf")

    # Get top-k indices
    top_k = min(top_k, num_books)
    top_vals, top_idx = torch.topk(scores, k=top_k)
    recommendations = []
    for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
        book_title = id2book.get(idx, f"<book_id_{idx}>")
        recommendations.append((book_title, float(val)))

    return recommendations


if __name__ == "__main__":
    # Minimal usage example (adjust user id as required)
    try:
        sample_user = 277427  # replace with an actual user id from your graph

        # Load graph to extract and print the user's original ratings
        num_users, num_books, edges, ratings, user2id, book2id = load_and_prep_graph(
            GRAPH_FILENAME
        )
        try:
            user_key = _resolve_user_key(user2id, sample_user)
        except KeyError:
            raise ValueError("Provided user id not found in graph node mapping.")
        u_idx = user2id[user_key]
        id2book = {v: k for k, v in book2id.items()}

        # edges and ratings are aligned lists; collect original user ratings
        user_original_ratings = [
            (id2book[b], r) for (u, b), r in zip(edges, ratings) if u == u_idx
        ]

        if user_original_ratings:
            print("User's original ratings:")
            for title, r in user_original_ratings:
                print(f"{r:.1f}  -  {title}")
        else:
            print("No original ratings found for this user.")

        # Ask which recommender to run
        choice = input("Choose recommender ('neighbors', 'lightgcn' or 'all') [lightgcn]: ").strip().lower() or "lightgcn"

        def _short(s: str, maxlen: int = 60):
            if len(s) <= maxlen:
                return s
            return s[: maxlen - 3] + "..."

        if choice == "neighbors" or choice == "n":
            # Ensure neighbor matrix exists (skip rebuild if file exists)
            rec_path = build_rec_filename(GRAPH_FILENAME)
            if not os.path.exists(rec_path):
                print("Neighbor matrix not found, building it (this may take a while)...")
                build_neighbor_rec_matrix(graph_filename=GRAPH_FILENAME, min_common=2, centered=True, force=False)
            recs = get_neighbor_recs_for_user(sample_user, top_k=10, graph_filename=GRAPH_FILENAME, rec_matrix_path=rec_path)
            print("\nNeighbor-based Recommendations:")
            for title, score in recs:
                print(f"{score:.4f}  -  {title}")

        elif choice == "all" or choice == "a":
            # Ensure neighbor matrix exists (skip rebuild if file exists)
            rec_path = build_rec_filename(GRAPH_FILENAME)
            if not os.path.exists(rec_path):
                print("Neighbor matrix not found, building it (this may take a while)...")
                build_neighbor_rec_matrix(graph_filename=GRAPH_FILENAME, min_common=2, centered=True, force=False)

            print("Computing Neighbor-based recommendations...")
            neighbor_recs = get_neighbor_recs_for_user(sample_user, top_k=10, graph_filename=GRAPH_FILENAME, rec_matrix_path=rec_path)

            print("Computing LightGCN recommendations...")
            try:
                lightgcn_recs = get_lightCGN_recommendations_for_existing_user(sample_user, top_k=10)
            except Exception as e:
                lightgcn_recs = []
                print(f"Warning: LightGCN recommendations unavailable: {e}")

            # Print side-by-side table
            print("\nTop recommendations (Neighbor vs LightGCN):")
            col_width = 70
            header = f"{'Rank':<5} | {'Neighbor (score)':<{col_width}} | {'LightGCN (score)':<{col_width}}"
            print(header)
            print("-" * (len(header) + 2))
            rows = max(len(neighbor_recs), len(lightgcn_recs))
            for i in range(rows):
                n_title_score = ""
                l_title_score = ""
                if i < len(neighbor_recs):
                    t, s = neighbor_recs[i]
                    n_title_score = f"{_short(t, col_width-8)} ({s:.4f})"
                if i < len(lightgcn_recs):
                    t, s = lightgcn_recs[i]
                    l_title_score = f"{_short(t, col_width-8)} ({s:.4f})"
                print(f"{i+1:<5} | {n_title_score:<{col_width}} | {l_title_score:<{col_width}}")

        else:
            # Now compute and print recommendations using LightGCN
            recs = get_lightCGN_recommendations_for_existing_user(sample_user, top_k=10)
            print("\nLightGCN Recommendations:")
            for title, score in recs:
                print(f"{score:.4f}  -  {title}")
    except Exception as e:
        print(f"Error: {e}")
