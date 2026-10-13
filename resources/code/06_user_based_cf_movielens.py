"""
06 - User-based collaborative filtering on the real MovieLens data

What this file does:
  Step 1. Split the ratings: 80% to learn from, 20% hidden for testing
  Step 2. Make the user x movie table from the training ratings
  Step 3. Subtract each user's average and fill empty cells with 0
  Step 4. Compute the similarity between every pair of users
  Step 5. Write a function that predicts one rating
  Step 6. Recommend 10 movies to one user
  Step 7. Measure the error (RMSE, MAE) on the hidden 20%
  Step 8. Try different numbers of neighbours k

Run with:  python 06_user_based_cf_movielens.py      (takes about 1 minute)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
movie_title = movies.set_index("movieId")["title"]    # look up a title: movie_title[1]


# ------------------------------------------------------------
# Step 1: train / test split
#   stratify=userId keeps 80/20 for EVERY user, so every user is in both parts
# ------------------------------------------------------------
train, test = train_test_split(ratings, test_size=0.2, random_state=42,
                               stratify=ratings["userId"])
print("Training ratings:", len(train), "   Test ratings:", len(test))


# ------------------------------------------------------------
# Step 2: user x movie table (rows = users, columns = movies, NaN = not rated)
# ------------------------------------------------------------
table = train.pivot_table(index="userId", columns="movieId", values="rating")
print("Table size:", table.shape)


# ------------------------------------------------------------
# Step 3: subtract each user's average, then fill the empty cells with 0
#   0 now means "average opinion", not "hated it"
# ------------------------------------------------------------
user_average = table.mean(axis=1)
centred = table.sub(user_average, axis=0)
centred = centred.fillna(0)


# ------------------------------------------------------------
# Step 4: similarity between every pair of users (a 610 x 610 table)
# ------------------------------------------------------------
similarity = cosine_similarity(centred)
similarity = pd.DataFrame(similarity, index=table.index, columns=table.index)

print("Most similar users to user 1:")
print(similarity.loc[1].drop(1).sort_values(ascending=False).head(5).round(3))
print()


# ------------------------------------------------------------
# Step 5: predict how much `user` would like `movie`
# ------------------------------------------------------------
def predict_rating(user, movie, k=30):
    # Case 1: nobody rated this movie in training -> use the user's average
    if movie not in table.columns:
        return user_average[user]

    # Users (other than our user) who rated this movie
    raters = table[movie].dropna().index
    raters = raters.drop(user, errors="ignore")

    # Their similarity to our user; keep the k most similar, positive ones
    sims = similarity.loc[user, raters]
    sims = sims.sort_values(ascending=False).head(k)
    sims = sims[sims > 0]

    # Case 2: no similar user rated it -> use the user's average
    if len(sims) == 0:
        return user_average[user]

    # Case 3: mean-centred weighted average
    top = 0
    bottom = 0
    for neighbour, sim in sims.items():
        how_much_more_they_liked_it = table.loc[neighbour, movie] - user_average[neighbour]
        top = top + sim * how_much_more_they_liked_it
        bottom = bottom + abs(sim)

    prediction = user_average[user] + top / bottom

    # keep the prediction inside the rating scale 0.5 ... 5
    prediction = min(max(prediction, 0.5), 5.0)
    return prediction


print("Predicted rating of user 1 for Toy Story (movieId 1):", round(predict_rating(1, 1), 2))
print()


# ------------------------------------------------------------
# Step 6: recommend 10 movies to a user
# ------------------------------------------------------------
def recommend(user, how_many=10):
    already_rated = table.loc[user].dropna().index
    scores = {}
    for movie in table.columns:
        if movie not in already_rated:
            scores[movie] = predict_rating(user, movie)
    best = sorted(scores, key=scores.get, reverse=True)[:how_many]
    for movie in best:
        print("   ", round(scores[movie], 2), "-", movie_title[movie])


print("User 10's favourite movies (from training data):")
favourites = table.loc[10].dropna().sort_values(ascending=False).head(5)
for movie, rating in favourites.items():
    print("   ", rating, "-", movie_title[movie])
print("Recommended for user 10:")
recommend(10)
print("Notice: every prediction hits the 5.0 cap, and most are little-known films.")
print("A movie rated 5 by one or two similar users is enough to reach the top.")
print()


# ------------------------------------------------------------
# Step 7: error on the hidden test ratings
#   RMSE = square root of the average squared error (punishes big mistakes)
#   MAE  = average absolute error
# ------------------------------------------------------------
def rmse(errors):
    return np.sqrt(np.mean(np.array(errors) ** 2))


def mae(errors):
    return np.mean(np.abs(np.array(errors)))


errors_cf = []
errors_global = []
errors_user = []
global_average = train["rating"].mean()

for row in test.itertuples():
    errors_cf.append(row.rating - predict_rating(row.userId, row.movieId))
    errors_global.append(row.rating - global_average)
    errors_user.append(row.rating - user_average[row.userId])

print("Error on", len(test), "hidden ratings:")
print("   Always predict the global average  RMSE =", round(rmse(errors_global), 4), " MAE =", round(mae(errors_global), 4))
print("   Predict the user's own average     RMSE =", round(rmse(errors_user), 4), " MAE =", round(mae(errors_user), 4))
print("   User-based CF (k = 30)             RMSE =", round(rmse(errors_cf), 4), " MAE =", round(mae(errors_cf), 4))
print()


# ------------------------------------------------------------
# Step 8: how many neighbours? (on 2,000 test ratings to save time)
# ------------------------------------------------------------
sample = test.sample(2000, random_state=0)
print("RMSE for different numbers of neighbours k:")
for k in [5, 10, 20, 30, 50, 100, 200]:
    errors = []
    for row in sample.itertuples():
        errors.append(row.rating - predict_rating(row.userId, row.movieId, k))
    print("   k =", k, "  RMSE =", round(rmse(errors), 4))
