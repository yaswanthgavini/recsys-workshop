"""
09 - Matrix factorization by hand on a small example (5 users, 4 movies)

Idea: describe each user with 2 numbers and each movie with 2 numbers,
      so that  rating  ~  user numbers  .  movie numbers   (dot product)

What this file does:
  Step 1. The rating table (0 = unknown)
  Step 2. Start with the same random numbers as in the class notes
  Step 3. Repeat 5,000 times: for each known rating, nudge the numbers to reduce the error
  Step 4. Print the learned numbers and the predictions for the unknown cells

Run with:  python 09_matrix_factorization_small_example.py
"""

import numpy as np


# ------------------------------------------------------------
# Step 1: rating table, 0 = unknown
# ------------------------------------------------------------
Y = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]], dtype=float)
number_of_users, number_of_movies = Y.shape
k = 2              # number of hidden factors
alpha = 0.0002     # learning rate (size of each nudge)
beta = 0.02        # regularisation (keeps numbers small)


# ------------------------------------------------------------
# Step 2: starting numbers (same as the class notes)
#   U: one row per user, V: one row per movie
# ------------------------------------------------------------
U = np.array([[0.5904876, 0.18520637],
              [0.89651422, 0.62040181],
              [0.97932976, 0.31239695],
              [0.71248937, 0.00531318],
              [0.2988344, 0.7485681]])
V = np.array([[0.93627203, 0.35834808],
              [0.78030931, 0.04508127],
              [0.01137135, 0.01842057],
              [0.31557944, 0.10453503]])


def error_on_known_ratings(U, V):
    prediction = U @ V.T                 # @ = matrix multiplication -> 5 x 4 table
    known = Y > 0
    return np.sqrt(np.mean((Y[known] - prediction[known]) ** 2))


# ------------------------------------------------------------
# Step 3: gradient descent
# ------------------------------------------------------------
for step in range(5001):

    if step in [0, 1500, 3000, 5000]:
        print("After", step, "steps: error on known ratings =", round(error_on_known_ratings(U, V), 3))
        print(np.round(U @ V.T, 2))
        print()

    if step == 5000:
        break

    for i in range(number_of_users):
        for j in range(number_of_movies):
            if Y[i, j] > 0:                              # only known ratings
                error = Y[i, j] - np.dot(U[i], V[j])     # how far off we are
                for f in range(k):                       # nudge each factor
                    U[i, f] = U[i, f] + alpha * (2 * error * V[j, f] - beta * U[i, f])
                    V[j, f] = V[j, f] + alpha * (2 * error * U[i, f] - beta * V[j, f])


# ------------------------------------------------------------
# Step 4: results
# ------------------------------------------------------------
print("Learned user factors U (think: how much each user likes 'action' and 'romance'):")
print(np.round(U, 2))
print("Learned movie factors V (how much 'action' and 'romance' each movie has):")
print(np.round(V, 2))
print()

prediction = U @ V.T
print("Predictions for the unknown cells:")
for i in range(number_of_users):
    for j in range(number_of_movies):
        if Y[i, j] == 0:
            print("   User", i + 1, "Movie", j + 1, "->", round(prediction[i, j], 2))
