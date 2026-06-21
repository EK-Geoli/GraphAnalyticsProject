import os
import pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
GRAPH_FILENAME = "book_graph_B5_R100_CR0.pkl"
EMBEDDING_DIM = 64
NUM_LAYERS = 3  # K-layers of LightGCN propagation
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4  # L2 Regularization
EPOCHS = 30
BATCH_SIZE = 1024

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --- 1. DATA PREPARATION ---
class RatingDataset(Dataset):
    def __init__(self, edges, ratings):
        self.edges = torch.tensor(edges, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.edges[idx, 0], self.edges[idx, 1], self.ratings[idx]


def load_and_prep_graph(filename):
    print(f"Loading cached graph from {filename}...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find {filename}. Run the data filtering script first.")

    with open(filename, 'rb') as f:
        B_all = pickle.load(f)

    # Separate nodes by bipartite attribute
    users = [n for n, d in B_all.nodes(data=True) if d['bipartite'] == 0]
    books = [n for n, d in B_all.nodes(data=True) if d['bipartite'] == 1]

    # Create mapping from original string IDs to 0-indexed integers
    user2id = {u: i for i, u in enumerate(users)}
    book2id = {b: i for i, b in enumerate(books)}

    num_users = len(users)
    num_books = len(books)

    print(f"Graph loaded: {num_users} Users, {num_books} Books")

    # Extract edges and ratings
    edges = []
    ratings = []
    for u, b, data in B_all.edges(data=True):
        # Ensure 'u' is the user and 'b' is the book
        if B_all.nodes[u]['bipartite'] == 1:
            u, b = b, u
        edges.append([user2id[u], book2id[b]])
        ratings.append(data['rating'])

    return num_users, num_books, edges, ratings, user2id, book2id


def create_sparse_adjacency(num_users, num_books, edges):
    """Creates the normalized symmetric adjacency matrix required for LightGCN."""
    print("Building normalized adjacency matrix...")

    # 1. Create a boolean user-book interaction matrix
    R = sp.dok_matrix((num_users, num_books), dtype=np.float32)
    for u, b in edges:
        R[u, b] = 1.0
    R = R.tolil()

    # 2. Build the full symmetric adjacency matrix for the bipartite graph
    # [[0, R],
    #  [R.T, 0]]
    adj = sp.bmat([[None, R], [R.T, None]])

    # 3. Normalize the matrix: D^(-0.5) * A * D^(-0.5)
    rowsum = np.array(adj.sum(1))
    d_inv_sq = np.power(rowsum, -0.5).flatten()
    d_inv_sq[np.isinf(d_inv_sq)] = 0.0
    d_mat_inv_sq = sp.diags(d_inv_sq)

    norm_adj = d_mat_inv_sq.dot(adj).dot(d_mat_inv_sq).tocoo()

    # 4. Convert to PyTorch sparse tensor
    indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)

    sparse_adj = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_adj


# --- 2. THE LIGHTGCN MODEL ---
class LightGCNRegression(nn.Module):
    def __init__(self, num_users, num_books, embed_dim, num_layers, global_mean):
        super(LightGCNRegression, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.num_layers = num_layers

        # Core Embeddings (Randomly initialized)
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.book_emb = nn.Embedding(num_books, embed_dim)

        # Biases for the Rating scale constraint
        self.user_bias = nn.Embedding(num_users, 1)
        self.book_bias = nn.Embedding(num_books, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_mean, dtype=torch.float32))

        # Initialization logic
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.book_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.book_bias.weight)

    def forward(self, sparse_adj):
        """Propagates embeddings across the graph structure."""
        # Layer 0 embeddings
        ego_embeddings = torch.cat([self.user_emb.weight, self.book_emb.weight], dim=0)
        all_embeddings = [ego_embeddings]

        # Multi-hop message passing
        for layer in range(self.num_layers):
            ego_embeddings = torch.sparse.mm(sparse_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Average embeddings from all layers
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)

        # Split back into users and books
        users_final, books_final = torch.split(final_embeddings, [self.num_users, self.num_books])
        return users_final, books_final

    def predict(self, users, books, users_final, books_final):
        """Calculates the 1-10 rating using dot product + biases."""
        u_emb = users_final[users]
        b_emb = books_final[books]

        u_bias = self.user_bias(users).squeeze()
        b_bias = self.book_bias(books).squeeze()

        # The core interaction
        dot_product = (u_emb * b_emb).sum(dim=1)

        # The final prediction constrained by global and local baseline trends
        prediction = self.global_bias + u_bias + b_bias + dot_product
        return prediction


# --- helper: deterministic weights filename ---
def build_weights_filename(
    graph_filename=GRAPH_FILENAME,
    embed_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    seed=SEED,
):
    base = os.path.splitext(os.path.basename(graph_filename))[0]
    fname = (
        f"lightgcn_weights_{base}_D{embed_dim}_L{num_layers}"
        f"_LR{learning_rate}_WD{weight_decay}_EP{epochs}_BS{batch_size}_S{seed}.pth"
    )
    return fname


def default_weights_path():
    return build_weights_filename()


# --- 3. TRAINING LOOP ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # If a model trained with the exact same configuration already exists, skip training.
    weights_path = build_weights_filename()
    if os.path.exists(weights_path):
        print(f"Found existing model weights at '{weights_path}' matching current configuration. Skipping training.")
        return

    # Load Data
    num_users, num_books, edges, ratings, user2id, book2id = load_and_prep_graph(GRAPH_FILENAME)

    global_mean_rating = np.mean(ratings)
    print(f"Global Mean Rating: {global_mean_rating:.2f}")

    # Build Graph Structure
    sparse_adj = create_sparse_adjacency(num_users, num_books, edges).to(device)

    # Dataloader (use deterministic generator seeded above)
    dataset = RatingDataset(edges, ratings)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=dl_generator)

    # Initialize Model
    model = LightGCNRegression(
        num_users=num_users,
        num_books=num_books,
        embed_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        global_mean=global_mean_rating
    ).to(device)

    # MSE Loss + Optimizer (Weight Decay handles the L2 Regularization)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_users, batch_books, batch_ratings in dataloader:
            batch_users = batch_users.to(device)
            batch_books = batch_books.to(device)
            batch_ratings = batch_ratings.to(device)

            optimizer.zero_grad()

            # 1. MOVED INSIDE THE LOOP: Generate fresh graph embeddings for this batch step
            users_final, books_final = model(sparse_adj)

            # 2. Predict ratings
            predictions = model.predict(batch_users, batch_books, users_final, books_final)

            # 3. Calculate loss and backpropagate
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_ratings)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{EPOCHS} | MSE Loss: {avg_loss:.4f} | RMSE: {np.sqrt(avg_loss):.4f}")

    print("Training Complete.")

    # Save the model state to the deterministic path
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to '{weights_path}'")


if __name__ == "__main__":
    train()
