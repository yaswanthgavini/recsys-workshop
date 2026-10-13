"""
16 - Alternating Least Squares (ALS) from scratch (Lab 3, short demo)

The schedule says "Matrix factorization (SVD/ALS)". Files 09-12 used gradient descent (SGD).
ALS reaches the same goal differently:

  SGD: nudge one rating at a time.
  ALS: fix the movie factors, solve for ALL user factors exactly; then fix the user
       factors and solve for ALL movie factors exactly. Repeat.

Each half-step is an ordinary least-squares problem, so we can solve it in one line.

What this file does:
  Step 1. Split the data (same split as files 06 and 10)
  Step 2. Build the training matrix and a mask of known cells
  Step 3. Alternate: solve the movie factors, then the user factors, 10 times
  Step 4. Measure the test RMSE after each round

Run with:  python 16_als_from_scratch.py      (takes about 30 seconds)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("ratings.csv")


# ------------------------------------------------------------
# Step 1: split
# ------------------------------------------------------------
train, test = train_test_split(ratings, test_size=0.2, random_state=42,
                               stratify=ratings["userId"])


# ------------------------------------------------------------
# Step 2: training matrix
#   known[u, m] is True where a rating exists.
#   We subtract the overall average, so 0 means "average" (there are no biases in this demo).
# ------------------------------------------------------------
table = train.pivot_table(index="userId", columns="movieId", values="rating")
overall_average = train["rating"].mean()

known = table.notna().to_numpy()
centred = np.nan_to_num(table.to_numpy() - overall_average)
number_of_users, number_of_movies = centred.shape

k = 20                  # number of hidden factors
regularisation = 15.0   # a small value (0.1) over-fits badly; try it and see

rng = np.random.default_rng(0)
U = rng.normal(0, 0.1, size=(number_of_users, k))      # user factors
V = np.zeros((number_of_movies, k))                    # movie factors (solved first)
identity = regularisation * np.eye(k)

# positions of the test ratings inside the matrix (skip users / movies not seen in training)
rows = table.index.get_indexer(test["userId"])
cols = table.columns.get_indexer(test["movieId"])
inside = (rows >= 0) & (cols >= 0)


def test_rmse():
    prediction = overall_average + np.sum(U[rows[inside]] * V[cols[inside]], axis=1)
    prediction = np.clip(prediction, 0.5, 5.0)
    return np.sqrt(np.mean((test["rating"].to_numpy()[inside] - prediction) ** 2))


# ------------------------------------------------------------
# Step 3: alternate
#   For one movie m, let U_m = the factors of the users who rated it, r_m = their ratings.
#   The best factors for the movie solve:   (U_m' U_m + lambda I) v = U_m' r_m
# ------------------------------------------------------------
for round_number in range(1, 11):

    for m in range(number_of_movies):                  # fix U, solve every movie
        raters = known[:, m]
        U_m = U[raters]
        V[m] = np.linalg.solve(U_m.T @ U_m + identity, U_m.T @ centred[raters, m])

    for u in range(number_of_users):                   # fix V, solve every user
        rated = known[u]
        V_u = V[rated]
        U[u] = np.linalg.solve(V_u.T @ V_u + identity, V_u.T @ centred[u, rated])

    # Step 4: error after this round
    print("Round", round_number, "  test RMSE =", round(test_rmse(), 4))

print()
print("ALS reaches a good RMSE in very few rounds (SGD needed 20 epochs), but each round is heavier.")
print("ALS also parallelises easily (every user and every movie is solved independently),")
print("which is why it is used on very large data sets such as Spark.")
