"""
11 - Matrix factorization with the Surprise library

Surprise does in a few lines what file 10 did by hand.

What this file does:
  Step 1. Load the ratings into Surprise and split 80/20
  Step 2. Train SVD and measure RMSE / MAE
  Step 3. Predict one rating
  Step 4. Compare several algorithms on the same split
  Step 5. Try different numbers of factors
  Step 6. Try different amounts of regularisation

Run with:  python 11_surprise_svd.py      (about 1 minute)
"""

import pandas as pd
from surprise import Dataset, Reader, SVD, BaselineOnly, KNNWithMeans, NormalPredictor, accuracy
from surprise.model_selection import train_test_split

ratings = pd.read_csv("ratings.csv")


# ------------------------------------------------------------
# Step 1: load into Surprise
#   The table must have exactly 3 columns: user, item, rating
# ------------------------------------------------------------
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# ------------------------------------------------------------
# Step 2: train SVD
# ------------------------------------------------------------
model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)
predictions = model.test(testset)

print("SVD with 50 factors:")
accuracy.rmse(predictions)
accuracy.mae(predictions)
print()


# ------------------------------------------------------------
# Step 3: predict one rating
# ------------------------------------------------------------
one = model.predict(uid=10, iid=1)          # user 10, movie 1 (Toy Story)
print("User 10, Toy Story: predicted rating =", round(one.est, 2))
print()


# ------------------------------------------------------------
# Step 4: compare algorithms on the same split
# ------------------------------------------------------------
models = {
    "Random guess":          NormalPredictor(),
    "Baseline (biases only)": BaselineOnly(verbose=False),
    "User-based KNN":        KNNWithMeans(k=40, sim_options={"name": "pearson", "user_based": True}, verbose=False),
    "Item-based KNN":        KNNWithMeans(k=40, sim_options={"name": "pearson", "user_based": False}, verbose=False),
    "SVD (default)":         SVD(random_state=0),
}

print("Comparison on the same test set:")
for name, algorithm in models.items():
    algorithm.fit(trainset)
    rmse = accuracy.rmse(algorithm.test(testset), verbose=False)
    print("   ", name.ljust(24), "RMSE =", round(rmse, 4))
print()


# ------------------------------------------------------------
# Step 5: number of factors
# ------------------------------------------------------------
print("SVD with different numbers of factors:")
for n in [2, 10, 50, 100, 200]:
    algorithm = SVD(n_factors=n, random_state=0)
    algorithm.fit(trainset)
    rmse = accuracy.rmse(algorithm.test(testset), verbose=False)
    print("    factors =", n, "  RMSE =", round(rmse, 4))
print()


# ------------------------------------------------------------
# Step 6: regularisation (how strongly numbers are kept small)
# ------------------------------------------------------------
print("SVD (100 factors, 40 epochs) with different regularisation:")
for reg in [0.02, 0.05, 0.1, 0.2]:
    algorithm = SVD(n_factors=100, n_epochs=40, lr_all=0.01, reg_all=reg, random_state=0)
    algorithm.fit(trainset)
    rmse = accuracy.rmse(algorithm.test(testset), verbose=False)
    print("    reg_all =", reg, "  RMSE =", round(rmse, 4))
