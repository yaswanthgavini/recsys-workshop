"""
14 - Ranking quality of the user-based CF recommender (Lab 2, finish)

File 06 measured RMSE/MAE. File 08 showed Precision, Recall and NDCG on tiny lists.
This file puts them together: how good are the TOP-10 lists of our CF model?

What this file does:
  Step 1. Rebuild the CF model from file 06 (same split, same predictions)
  Step 2. Precision@10, Recall@10 and NDCG@10 for CF and for the popularity list
  Step 3. RMSE as a function of k (and save a plot)
  Step 4. Significance weighting: shrink similarities that rest on few shared movies
  Step 5. Random split vs time split: how many cold users / cold movies appear?

Steps 3-5 are extensions: do them if there is time.

Run with:  python 14_ranking_metrics_cf.py      (takes about 2 minutes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("ratings.csv")


# ------------------------------------------------------------
# Step 1: the CF model from file 06
# ------------------------------------------------------------
train, test = train_test_split(ratings, test_size=0.2, random_state=42,
                               stratify=ratings["userId"])

table = train.pivot_table(index="userId", columns="movieId", values="rating")
user_average = table.mean(axis=1)
centred = table.sub(user_average, axis=0).fillna(0)
similarity = pd.DataFrame(cosine_similarity(centred), index=table.index, columns=table.index)


def predict_rating(user, movie, k=30, sim_table=similarity):
    if movie not in table.columns:
        return user_average[user]
    raters = table[movie].dropna().index.drop(user, errors="ignore")
    sims = sim_table.loc[user, raters].sort_values(ascending=False).head(k)
    sims = sims[sims > 0]
    if len(sims) == 0:
        return user_average[user]
    top = 0
    bottom = 0
    for neighbour, sim in sims.items():
        top = top + sim * (table.loc[neighbour, movie] - user_average[neighbour])
        bottom = bottom + abs(sim)
    return min(max(user_average[user] + top / bottom, 0.5), 5.0)


def rmse(true, predicted):
    return np.sqrt(np.mean((np.array(true) - np.array(predicted)) ** 2))


test = test.copy()
test["cf"] = [predict_rating(u, m) for u, m in zip(test["userId"], test["movieId"])]
print("Step 1: CF predictions for", len(test), "hidden ratings. RMSE =", round(rmse(test["rating"], test["cf"]), 4))
print()


# ------------------------------------------------------------
# Step 2: ranking metrics on each user's hidden ratings
#   Each user sorts their hidden movies by the model's score.
#   relevant = really rated 4 or more.
#   Relevance grade for NDCG: 0 = under 3, 1 = 3 to 3.5, 2 = 4 to 4.5, 3 = 5
# ------------------------------------------------------------
def dcg(grades):
    total = 0
    for position, grade in enumerate(grades, start=1):
        total = total + (2 ** grade - 1) / np.log2(position + 1)
    return total


def ndcg(grades):
    best = dcg(sorted(grades, reverse=True))
    return dcg(grades) / best if best > 0 else 0


def grade(rating):
    if rating >= 5:
        return 3
    if rating >= 4:
        return 2
    if rating >= 3:
        return 1
    return 0


# the popularity score from file 02 (damped average), computed on the training data only
stats = train.groupby("movieId")["rating"].agg(["mean", "count"])
overall = train["rating"].mean()
stats["score"] = (stats["count"] * stats["mean"] + 50 * overall) / (stats["count"] + 50)
test["popularity"] = test["movieId"].map(stats["score"]).fillna(overall)


def ranking_scores(score_column, K=10):
    precisions, recalls, ndcgs = [], [], []
    for user, rows in test.groupby("userId"):
        relevant_total = (rows["rating"] >= 4).sum()
        if len(rows) < K or relevant_total == 0:
            continue                                  # not enough hidden movies to fill a top-10
        top = rows.sort_values(score_column, ascending=False).head(K)
        hits = (top["rating"] >= 4).sum()
        precisions.append(hits / K)
        recalls.append(hits / relevant_total)
        ndcgs.append(ndcg([grade(r) for r in top["rating"]]))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs), len(precisions)


print("Step 2: ranking quality of the top-10 (among each user's hidden movies)")
print("   Method               Precision@10   Recall@10   NDCG@10")
for name, column in [("Popularity (file 02)", "popularity"), ("User-based CF (k=30)", "cf")]:
    p, r, n, users = ranking_scores(column)
    print("   ", name.ljust(21), str(round(p, 3)).ljust(14), str(round(r, 3)).ljust(11), round(n, 3))
print("   (", users, "users with at least 10 hidden movies and one relevant movie )")
print("CF is only slightly better than the popularity list here. We rank only the movies a user")
print("really rated, which is an easy test, and MovieLens users mostly rate movies they expect to like.")
print("So always compare a fancy model with the simple baseline on BOTH RMSE and ranking.")
print()


# ------------------------------------------------------------
# Step 3 (extension): RMSE as a function of k
# ------------------------------------------------------------
sample = test.sample(2000, random_state=0)
k_values = [5, 10, 20, 30, 50, 100, 200]
k_rmse = []
print("Step 3: RMSE for different numbers of neighbours k (2,000 hidden ratings)")
for k in k_values:
    predictions = [predict_rating(u, m, k) for u, m in zip(sample["userId"], sample["movieId"])]
    k_rmse.append(rmse(sample["rating"], predictions))
    print("   k =", str(k).ljust(4), "RMSE =", round(k_rmse[-1], 4))

plt.figure(figsize=(6, 4))
plt.plot(k_values, k_rmse, marker="o")
plt.xscale("log")
plt.title("User-based CF: RMSE against number of neighbours k")
plt.xlabel("k (log scale)")
plt.ylabel("RMSE")
plt.savefig("plot_4_rmse_vs_k.png")
plt.close()
print("Saved: plot_4_rmse_vs_k.png   (too few neighbours = noisy, too many = blurry)")
print()


# ------------------------------------------------------------
# Step 4 (extension): significance weighting
#   Two users who share only 3 movies can have a similarity of 1.0 by luck.
#   Multiply by min(shared, 20) / 20 to trust such pairs less.
# ------------------------------------------------------------
rated = table.notna().astype(int)
shared = rated @ rated.T                               # shared[u, v] = movies both users rated
beta = 20
weighted_similarity = similarity * np.minimum(shared, beta) / beta

plain = [predict_rating(u, m, 30) for u, m in zip(sample["userId"], sample["movieId"])]
weighted = [predict_rating(u, m, 30, weighted_similarity) for u, m in zip(sample["userId"], sample["movieId"])]
print("Step 4: significance weighting (beta = 20, k = 30, 2,000 hidden ratings)")
print("   plain similarity     RMSE =", round(rmse(sample["rating"], plain), 4))
print("   weighted similarity  RMSE =", round(rmse(sample["rating"], weighted), 4))
print("Here it does not help (k = 30 already keeps only strong neighbours). Check, do not assume.")
print()


# ------------------------------------------------------------
# Step 5 (extension): random split vs time split
#   A random split hides ratings from every user, so no user is ever "new".
#   A time split (train on the past, test on the future) is closer to real life.
# ------------------------------------------------------------
cut = ratings["timestamp"].quantile(0.8)
train_time = ratings[ratings["timestamp"] <= cut]
test_time = ratings[ratings["timestamp"] > cut]

cold_users = set(test_time["userId"]) - set(train_time["userId"])
cold_movies = set(test_time["movieId"]) - set(train_time["movieId"])
cold_user_rows = test_time["userId"].isin(cold_users)
cold_movie_rows = test_time["movieId"].isin(cold_movies)

print("Step 5: time split (last 20% of time is the test set)")
print("   train =", len(train_time), "  test =", len(test_time))
print("   users only in test:", len(cold_users), "  movies only in test:", len(cold_movies))
print("   test ratings from users never seen in training:", round(cold_user_rows.mean() * 100, 1), "%")
print("   test ratings of movies never seen in training:  ", round(cold_movie_rows.mean() * 100, 1), "%")
print("A time split creates real cold-start cases, a random split hides them.")
