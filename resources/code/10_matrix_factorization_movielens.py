"""
10 - Matrix factorization from scratch on MovieLens (the same idea as Surprise's SVD)

Model:   predicted rating = overall average + user bias + movie bias + (user factors . movie factors)
           user bias  = how generous this user is
           movie bias = how good this movie is in general
           factors    = the personal "taste" part

What this file does:
  Step 1. Split the data (same split as file 06)
  Step 2. Give every user and every movie a row number
  Step 3. Start with small random factors and zero biases
  Step 4. Train with stochastic gradient descent for 20 rounds (epochs)
  Step 5. Measure the error on the test ratings

Run with:  python 10_matrix_factorization_movielens.py     (about 20-40 seconds)
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
# Step 2: row numbers for users and movies
#   user_row[17] = the row in U that belongs to user 17
# ------------------------------------------------------------
user_row = {}
for user in train["userId"].unique():
    user_row[user] = len(user_row)

movie_row = {}
for movie in train["movieId"].unique():
    movie_row[movie] = len(movie_row)


# ------------------------------------------------------------
# Step 3: settings and starting values
# ------------------------------------------------------------
k = 50                  # number of hidden factors
learning_rate = 0.005
regularisation = 0.02
epochs = 20

rng = np.random.default_rng(0)                            # fixed seed -> same result every run
U = rng.normal(0, 0.1, size=(len(user_row), k))           # user factors
V = rng.normal(0, 0.1, size=(len(movie_row), k))          # movie factors
user_bias = np.zeros(len(user_row))
movie_bias = np.zeros(len(movie_row))
overall_average = train["rating"].mean()


def predict(user, movie):
    # unknown user or movie -> best we can do is the overall average
    if user not in user_row or movie not in movie_row:
        return overall_average
    u = user_row[user]
    m = movie_row[movie]
    prediction = overall_average + user_bias[u] + movie_bias[m] + np.dot(U[u], V[m])
    return min(max(prediction, 0.5), 5.0)


def test_rmse():
    errors = []
    for row in test.itertuples():
        errors.append(row.rating - predict(row.userId, row.movieId))
    return np.sqrt(np.mean(np.array(errors) ** 2))


# ------------------------------------------------------------
# Step 4: training
# ------------------------------------------------------------
train_rows = list(train[["userId", "movieId", "rating"]].itertuples(index=False))

for epoch in range(1, epochs + 1):
    rng.shuffle(train_rows)                    # visit the ratings in a random order

    for user, movie, rating in train_rows:
        u = user_row[user]
        m = movie_row[movie]

        # 1. how wrong is the current prediction?
        prediction = overall_average + user_bias[u] + movie_bias[m] + np.dot(U[u], V[m])
        error = rating - prediction

        # 2. nudge the biases
        user_bias[u] = user_bias[u] + learning_rate * (error - regularisation * user_bias[u])
        movie_bias[m] = movie_bias[m] + learning_rate * (error - regularisation * movie_bias[m])

        # 3. nudge the factors (use the OLD values for both updates)
        old_user_factors = U[u].copy()
        U[u] = U[u] + learning_rate * (error * V[m] - regularisation * U[u])
        V[m] = V[m] + learning_rate * (error * old_user_factors - regularisation * V[m])

    if epoch in [1, 5, 10, 15, 20]:
        print("Epoch", epoch, "  test RMSE =", round(test_rmse(), 4))


# ------------------------------------------------------------
# Step 5: a look inside
# ------------------------------------------------------------
movies = pd.read_csv("movies.csv")
movie_title = movies.set_index("movieId")["title"]
row_to_movie = {row: movie for movie, row in movie_row.items()}

print()
print("Movies with the largest movie bias (liked by almost everyone):")
for m in np.argsort(-movie_bias)[:5]:
    print("   ", round(movie_bias[m], 2), "-", movie_title[row_to_movie[m]])
