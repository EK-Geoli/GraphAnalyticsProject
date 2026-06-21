import os
import numpy as np
from scipy import sparse as sp
from typing import List, Tuple

# reuse graph loaders from existing module
from lightCGN_recommendation import load_and_prep_graph

# --- Defaults ---
DEFAULT_METHOD = "cosine"  # intersection / sqrt(|A|*|B|)
DEFAULT_MIN_COMMON = 2
DEFAULT_CENTERED = True
DEFAULT_SEED = 42


def build_rec_filename(
    graph_filename: str,
    method: str = DEFAULT_METHOD,
    min_common: int = DEFAULT_MIN_COMMON,
    centered: bool = DEFAULT_CENTERED,
    seed: int = DEFAULT_SEED,
):
    base = os.path.splitext(os.path.basename(graph_filename))[0]
    fname = f"neighbor_rec_matrix_{base}_M{method}_MC{min_common}_C{int(centered)}_S{seed}.npy"
    return fname


def build_neighbor_rec_matrix(
    graph_filename: str = "book_graph_B5_R100_CR0.pkl",
    method: str = DEFAULT_METHOD,
    min_common: int = DEFAULT_MIN_COMMON,
    centered: bool = DEFAULT_CENTERED,
    force: bool = False,
    seed: int = DEFAULT_SEED,
) -> str:
    """
    Build and save a dense recommendation score matrix (num_users x num_books).
    Returns path to the saved .npy file.
    Skips computation if matching file exists (unless force=True).
    """
    out_path = build_rec_filename(graph_filename, method, min_common, centered, seed)
    if os.path.exists(out_path) and (not force):
        print(f"[neighbors] Found existing neighbor rec matrix at '{out_path}', skipping rebuild.")
        return out_path

    print(f"[neighbors] Building neighbor recommendations (method={method}, min_common={min_common}, centered={centered})")
    # Load graph / data
    num_users, num_books, edges, ratings, user2id, book2id = load_and_prep_graph(graph_filename)
    print(f"[neighbors] Graph: {num_users} users, {num_books} books, {len(edges)} edges")

    # Build binary interaction matrix B (users x books)
    users = np.array([u for u, b in edges], dtype=np.int32)
    books = np.array([b for u, b in edges], dtype=np.int32)
    data_bin = np.ones(len(edges), dtype=np.float32)
    B = sp.csr_matrix((data_bin, (users, books)), shape=(num_users, num_books))

    # Degrees (#books read per user)
    deg = np.array(B.sum(axis=1)).flatten()  # shape (num_users,)

    # Book popularity and inverse-popularity
    book_pop = np.array(B.sum(axis=0)).flatten()  # shape (num_books,)
    inv_pop = 1.0 / (book_pop + 1.0)  # avoid div-by-zero

    # Sparse rating matrix R (users x books)
    data_r = np.array(ratings, dtype=np.float32)
    R = sp.csr_matrix((data_r, (users, books)), shape=(num_users, num_books))

    # Per-user mean rating (use only their rated items)
    user_counts = (deg).astype(np.float32)
    user_means = np.zeros(num_users, dtype=np.float32)
    nonzero_users = user_counts > 0
    user_means[nonzero_users] = np.asarray(R.sum(axis=1)).flatten()[nonzero_users] / user_counts[nonzero_users]

    # Compute common counts: C = B * B^T (sparse)
    common = B.dot(B.T).tocoo()

    # Build similarity matrix W as sparse (only keep pairs meeting criteria)
    rows = []
    cols = []
    data = []
    for u, v, c in zip(common.row, common.col, common.data):
        if u == v:
            continue
        if c < min_common:
            continue
        # exclude neighbors that have no books beyond the intersection:
        if deg[v] <= c:
            continue
        denom = np.sqrt(max(1.0, deg[u]) * max(1.0, deg[v]))
        w = c / denom
        if w <= 0:
            continue
        rows.append(u)
        cols.append(v)
        data.append(w)

    print(f"[neighbors] Collected {len(data)} neighbor weights after filtering")
    if len(data) == 0:
        rec_matrix = np.full((num_users, num_books), np.nan, dtype=np.float32)
        np.save(out_path, rec_matrix)
        print(f"[neighbors] Saved empty matrix to '{out_path}'")
        return out_path

    W = sp.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(num_users, num_users))
    # define sum_abs_weights here (fixes previous NameError)
    sum_abs_weights = np.array(np.abs(W).sum(axis=1)).flatten()
    print(f"[neighbors] Weight matrix W built: nnz={W.nnz}")

    # Center ratings efficiently: subtract user_means from their nonzero entries
    R_centered = R.copy().tolil()
    for u in range(num_users):
        if user_counts[u] == 0:
            continue
        if len(R_centered.rows[u]) == 0:
            continue
        R_centered.data[u] = [val - user_means[u] for val in R_centered.data[u]]
    R_centered = R_centered.tocsr()

    # Compute raw scores: S = W * R_centered  (num_users x num_books) sparse result
    S_sparse = W.dot(R_centered)

    # Convert S_sparse to dense array
    S_dense = S_sparse.toarray().astype(np.float32)

    # Apply inverse-popularity weighting to de-emphasize overly popular books
    S_dense *= inv_pop[None, :].astype(np.float32)

    # Normalize by sum of absolute weights per user and add baseline
    nz = sum_abs_weights > 0
    for u in range(num_users):
        if nz[u]:
            S_dense[u, :] = S_dense[u, :] / sum_abs_weights[u]
        else:
            S_dense[u, :] = 0.0
    global_mean = float(np.nanmean(ratings)) if len(ratings) > 0 else 0.0
    baseline = np.where(nonzero_users, user_means, global_mean).astype(np.float32)
    S_dense += baseline[:, None]

    # Save dense matrix
    np.save(out_path, S_dense)
    print(f"[neighbors] Saved neighbor matrix to '{out_path}' (shape={S_dense.shape})")
    return out_path


def load_neighbor_rec_matrix(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neighbor rec matrix not found at '{path}'")
    print(f"[neighbors] Loading neighbor rec matrix from '{path}'")
    return np.load(path, allow_pickle=False)


def get_neighbor_recs_for_user(
    original_user_id,
    top_k: int = 10,
    exclude_seen: bool = True,
    graph_filename: str = "book_graph_B5_R100_CR0.pkl",
    rec_matrix_path: str = None,
) -> List[Tuple[str, float]]:
    """
    Return top_k recommendations for original_user_id using a precomputed neighbor matrix.
    If rec_matrix_path is None, uses default filename based on graph and defaults.
    """
    print(f"[neighbors] Preparing recommendations for user: {original_user_id}")
    num_users, num_books, edges, ratings, user2id, book2id = load_and_prep_graph(graph_filename)

    # Resolve user key (original label must be present in user2id)
    user_key = original_user_id
    if user_key not in user2id:
        try:
            alt = int(original_user_id)
            if alt in user2id:
                user_key = alt
        except Exception:
            pass
    if user_key not in user2id:
        try:
            alt = str(original_user_id)
            if alt in user2id:
                user_key = alt
        except Exception:
            pass
    if user_key not in user2id:
        raise KeyError("User not found in graph mapping.")

    u_idx = user2id[user_key]
    id2book = {v: k for k, v in book2id.items()}

    if rec_matrix_path is None:
        rec_matrix_path = build_rec_filename(graph_filename)

    rec_matrix = load_neighbor_rec_matrix(rec_matrix_path)  # dense numpy array

    if u_idx >= rec_matrix.shape[0]:
        raise IndexError("User index outside recommendation matrix bounds.")

    scores = rec_matrix[u_idx].copy()  # 1D array of length num_books

    # Optionally exclude already seen books
    if exclude_seen:
        seen = {b for u, b in edges if u == u_idx}
        for b in seen:
            if 0 <= b < len(scores):
                scores[b] = -np.inf

    # Get top-k indices (handle NaN or -inf)
    valid_mask = np.isfinite(scores)
    if not np.any(valid_mask):
        print("[neighbors] No finite scores available for this user.")
        return []

    k = min(top_k, np.count_nonzero(valid_mask))
    idx = np.argpartition(-scores, kth=k-1)[:k]
    top_idx = idx[np.argsort(-scores[idx])]
    recommendations = [(id2book.get(int(i), f"<book_id_{i}>"), float(scores[int(i)])) for i in top_idx]
    print(f"[neighbors] Returning top-{len(recommendations)} recommendations for user {original_user_id}")
    return recommendations
