"""
15 - Tune SVD properly, look inside it, and check beyond accuracy (Lab 3, finish)

File 11 tried a few settings by hand. This file does it the standard way and then opens the model.

What this file does:
  Step 1. Grid search with cross-validation to find good SVD settings
  Step 2. Leaderboard: baseline, KNN, NMF and tuned SVD (3-fold cross-validation)
  Step 3. Look inside the trained model: biases and factors
  Step 4. Draw the movie space (the factors squeezed to 2 numbers)
  Step 5. Beyond accuracy: coverage, novelty and personalisation of the top-10 lists

Run with:  python 15_surprise_tuning_and_inside.py      (takes about 3 minutes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from surprise import Dataset, Reader, SVD, NMF, BaselineOnly, KNNWithMeans, NormalPredictor
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
movie_title = movies.set_index("movieId")["title"]

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)


# ------------------------------------------------------------
# Step 1: grid search
#   Tries every combination, scores each with 3-fold cross-validation.
#   3 x 2 x 2 = 12 settings x 3 folds = 36 model fits.
# ------------------------------------------------------------
grid = {"n_factors": [50, 100],
        "n_epochs": [20, 40],
        "lr_all": [0.01],
        "reg_all": [0.05, 0.1, 0.2]}

search = GridSearchCV(SVD, grid, measures=["rmse"], cv=3, n_jobs=-1)
search.fit(data)

print("Step 1: grid search (12 settings, 3 folds each)")
print("   best RMSE   :", round(search.best_score["rmse"], 4))
print("   best settings:", search.best_params["rmse"])
print("Note: more factors usually needs more regularisation.")
print()


# ------------------------------------------------------------
# Step 2: leaderboard
#   cross_validate trains and tests 3 times and averages, so the ranking is more reliable
#   than one split.  (SVD++ is also in Surprise but takes several minutes, so it is left out.)
# ------------------------------------------------------------
models = {
    "Random guess":           NormalPredictor(),
    "Baseline (biases only)": BaselineOnly(verbose=False),
    "User-based KNN":         KNNWithMeans(k=40, sim_options={"name": "pearson", "user_based": True}, verbose=False),
    "NMF":                    NMF(random_state=0),
    "SVD (default)":          SVD(random_state=0),
    "SVD (tuned)":            SVD(random_state=0, **search.best_params["rmse"]),
}

print("Step 2: leaderboard (3-fold cross-validation, RMSE)")
for name, algorithm in models.items():
    result = cross_validate(algorithm, data, measures=["RMSE"], cv=3, verbose=False)
    print("   ", name.ljust(24), round(result["test_rmse"].mean(), 4))
print()


# ------------------------------------------------------------
# Step 3: look inside
#   After fit(), the model keeps its numbers:
#     model.bu = user biases    model.bi = movie biases
#     model.pu = user factors   model.qi = movie factors
# ------------------------------------------------------------
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD(random_state=0, **search.best_params["rmse"])
model.fit(trainset)

print("Step 3: inside the tuned SVD")
print("   user factors :", model.pu.shape, "  (users x factors)")
print("   movie factors:", model.qi.shape, "  (movies x factors)")

inner_to_movie = {inner: int(trainset.to_raw_iid(inner)) for inner in range(trainset.n_items)}

print("   Movies with the biggest positive bias (loved by almost everyone):")
for inner in np.argsort(-model.bi)[:5]:
    print("      ", round(model.bi[inner], 2), "-", movie_title[inner_to_movie[inner]])
print("   Movies with the biggest negative bias:")
for inner in np.argsort(model.bi)[:3]:
    print("      ", round(model.bi[inner], 2), "-", movie_title[inner_to_movie[inner]])
print("   Most generous user bias: +", round(model.bu.max(), 2), "  strictest: ", round(model.bu.min(), 2))
print()


# ------------------------------------------------------------
# Step 4: draw the movie space
#   Each movie has many factors. PCA squeezes them to 2 numbers so we can plot them.
#   Only the 300 most-rated movies, so the picture is not too crowded.
# ------------------------------------------------------------
counts = ratings.groupby("movieId").size().sort_values(ascending=False)
popular_inner = [trainset.to_inner_iid(m) for m in counts.index[:300] if trainset.knows_item(m)]
points = PCA(n_components=2).fit_transform(model.qi[popular_inner])

plt.figure(figsize=(7, 6))
plt.scatter(points[:, 0], points[:, 1], s=12, alpha=0.5)
for number, inner in enumerate(popular_inner[:12]):
    plt.annotate(movie_title[inner_to_movie[inner]][:22], points[number], fontsize=7)
plt.title("The 300 most-rated movies in factor space (PCA to 2D)")
plt.savefig("plot_5_movie_space.png", dpi=120)
plt.close()
print("Step 4: saved plot_5_movie_space.png (movies that people rate alike sit close together)")
print()


# ------------------------------------------------------------
# Step 5: beyond accuracy, on the top-10 lists of 100 users
#   coverage        = share of the catalogue that appears in anyone's list
#   novelty         = how unpopular the recommended movies are (higher = less obvious)
#   personalisation = 1 - average overlap between two users' lists (0 = everyone gets the same list)
# ------------------------------------------------------------
seen = ratings.groupby("userId")["movieId"].apply(set)
all_movies = ratings["movieId"].unique()
popularity = counts / counts.sum()

lists = []
for user in ratings["userId"].unique()[:100]:
    candidates = [m for m in all_movies if m not in seen[user]]
    scores = np.array([model.predict(user, m).est for m in candidates])
    lists.append([candidates[j] for j in np.argsort(-scores)[:10]])

shown = set(m for one_list in lists for m in one_list)
novelty = np.mean([-np.log2(popularity[m]) for one_list in lists for m in one_list])

overlaps = []
for a in range(len(lists)):
    for b in range(a + 1, len(lists)):
        overlaps.append(len(set(lists[a]) & set(lists[b])) / 10)

print("Step 5: beyond accuracy (top-10 lists for 100 users)")
print("   coverage       =", round(len(shown) / len(all_movies) * 100, 1), "% of all movies")
print("   novelty        =", round(novelty, 2), "  (bits; popular movies score low)")
print("   personalisation=", round(1 - np.mean(overlaps), 3))
print("A low RMSE does not guarantee varied lists. Check these too.")
