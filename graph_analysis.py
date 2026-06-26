import pandas as pd

df = pd.read_csv('ratings.csv')

# find if there are any duplicate user IDs-book IDs pairs
duplicates = df.duplicated(subset=['User-ID', 'ISBN'], keep=False)
print("Are there any duplicate user-book pairs?")
print(duplicates.any())
