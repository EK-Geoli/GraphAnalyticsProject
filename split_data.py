import pandas as pd
import random

def split_data():
    print("Loading original_ratings.csv...")
    try:
        df = pd.read_csv('original_ratings.csv')
    except FileNotFoundError:
        print("Error: 'original_ratings.csv' not found. Please ensure you have copied the raw data to this filename.")
        return

    print(f"Original data size: {len(df)} rows.")

    # 1. Remove 0 ratings
    df_explicit = df[df['Book-Rating'] > 0].copy()
    print(f"Data size after removing 0s: {len(df_explicit)} rows.")

    # 2. Calculate distribution and expected percentiles
    rating_counts = df_explicit['Book-Rating'].value_counts().sort_index(ascending=False)
    total_explicit = len(df_explicit)

    print("\n--- Global Rating Distribution (Explicit Only) ---")
    percentile_map = {}
    cum_percent = 0.0

    for rating in range(10, 0, -1):
        if rating in rating_counts:
            count = rating_counts[rating]
            freq = count / total_explicit
            # Expected percentile is cumulative frequency above this rating + half the frequency of this rating bin
            expected_pct = cum_percent + (freq / 2)
            percentile_map[rating] = expected_pct
            print(f"Rating {rating:2d}: {freq*100:5.2f}% of data | Expected Percentile: {expected_pct:.4f}")
            cum_percent += freq

    # 3. Select 50% of users to provide one test case
    unique_users = df_explicit['User-ID'].unique()
    test_users = set(random.sample(list(unique_users), k=len(unique_users) // 2))

    test_rows = []
    train_indices = []

    print(f"\nSplitting data: Selecting 1 review from {len(test_users)} users for the test set...")

    # Optimize splitting by grouping
    grouped = df_explicit.groupby('User-ID')

    for user, group in grouped:
        if user in test_users:
            # Pick one random row index to hold out
            test_idx = random.choice(group.index.tolist())
            row_data = df_explicit.loc[test_idx].copy()
            row_data['Expected-Percentile'] = percentile_map[row_data['Book-Rating']]
            test_rows.append(row_data)

            # Keep the rest for training
            train_indices.extend([idx for idx in group.index if idx != test_idx])
        else:
            train_indices.extend(group.index.tolist())

    df_test = pd.DataFrame(test_rows)
    df_train = df_explicit.loc[train_indices]

    # 4. Save files
    df_train.to_csv('ratings.csv', index=False)
    df_test.to_csv('test_set.csv', index=False)

    print(f"\nSplit complete!")
    print(f"Train set saved to 'ratings.csv' ({len(df_train)} rows).")
    print(f"Test set saved to 'test_set.csv' ({len(df_test)} rows).")

if __name__ == "__main__":
    # Fix seed for reproducibility across runs
    random.seed(42)
    split_data()