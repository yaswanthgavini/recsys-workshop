"""
12 - Using the trained model: top-10 lists, Precision@10 and cold start

What this file does:
  Step 1. Train a tuned SVD on 80% of the ratings
  Step 2. Recommend 10 movies to one user
  Step 3. Precision@10 and Recall@10 on the test ratings
  Step 4. A brand-new user (cold start): fall back to popular movies

Run with:  python 12_recommend_with_svd.py
"""

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
movie_title = movies.set_index("movieId")["title"]


# ------------------------------------------------------------
# Step 1: train
# ------------------------------------------------------------
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD(n_factors=100, n_epochs=40, lr_all=0.01, reg_all=0.1, random_state=0)
model.fit(trainset)


# ------------------------------------------------------------
# Step 2: top-10 for one user
# ------------------------------------------------------------
def top_10(user):
    already_rated = set(ratings.loc[ratings["userId"] == user, "movieId"])
    scores = {}
    for movie in movies["movieId"]:
        if movie not in already_rated:
            scores[movie] = model.predict(user, movie).est
    best = sorted(scores, key=scores.get, reverse=True)[:10]
    return best


user = 1
print("Top 10 for user", user)
for movie in top_10(user):
    print("   ", movie_title[movie])
print()


# ------------------------------------------------------------
# Step 3: Precision@10 and Recall@10
#   For each user, sort their TEST movies by predicted rating.
#   A movie is "relevant" if the user really rated it 4 or more.
# ------------------------------------------------------------
predictions = model.test(testset)

by_user = {}
for p in predictions:
    if p.uid not in by_user:
        by_user[p.uid] = []
    by_user[p.uid].append((p.est, p.r_ui))      # (predicted, true)

precisions = []
recalls = []
for user, pairs in by_user.items():
    pairs.sort(reverse=True)                     # best predicted first
    top = pairs[:10]
    relevant_in_top = 0
    for predicted, true in top:
        if true >= 4:
            relevant_in_top = relevant_in_top + 1
    relevant_total = 0
    for predicted, true in pairs:
        if true >= 4:
            relevant_total = relevant_total + 1
    precisions.append(relevant_in_top / 10)
    if relevant_total > 0:
        recalls.append(relevant_in_top / relevant_total)

print("Precision@10 =", round(np.mean(precisions), 3))
print("Recall@10    =", round(np.mean(recalls), 3))
print()


# ------------------------------------------------------------
# Step 4: cold start - a user the model has never seen
# ------------------------------------------------------------
new_user = 99999
print("New user, Toy Story: predicted rating =", round(model.predict(new_user, 1).est, 2),
      "(only the overall average + movie bias - nothing personal)")

# Fallback: the popular-movie list from file 02
movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
overall_average = ratings["rating"].mean()
m = 50
movie_stats["score"] = (movie_stats["count"] * movie_stats["mean"] + m * overall_average) / (movie_stats["count"] + m)
popular = movie_stats.sort_values("score", ascending=False).head(5).index

print("So we show a new user the popular list instead:")
for movie in popular:
    print("   ", movie_title[movie])
