"""
05 - Item-based collaborative filtering on the same small example

Instead of "users like User3", we look for "items like Item1".

What this file does:
  Step 1. Subtract each user's average from their ratings
  Step 2. Compare two item columns (adjusted cosine similarity)
  Step 3. Predict User3's rating of Item1 and Item6 from the items User3 HAS rated

Run with:  python 05_item_based_cf_small_example.py
"""

import numpy as np
import pandas as pd

ratings = pd.DataFrame(
    [[7,      6,      7,      4, 5, 4],
     [6,      7,      np.nan, 4, 3, 4],
     [np.nan, 3,      3,      1, 1, np.nan],
     [1,      2,      2,      3, 3, 4],
     [1,      np.nan, 1,      2, 3, 3]],
    index=["User1", "User2", "User3", "User4", "User5"],
    columns=["Item1", "Item2", "Item3", "Item4", "Item5", "Item6"])

target_user = "User3"


# ------------------------------------------------------------
# Step 1: subtract each USER's average (removes generous / strict raters)
# ------------------------------------------------------------
user_average = ratings.mean(axis=1)
centred = ratings.sub(user_average, axis=0)     # subtract row by row

print("Ratings after subtracting each user's average:")
print(centred.round(2))
print()


# ------------------------------------------------------------
# Step 2: adjusted cosine similarity between two items
# ------------------------------------------------------------
def adjusted_cosine(item_a, item_b):
    both_rated = centred[item_a].notna() & centred[item_b].notna()
    a = centred.loc[both_rated, item_a]
    b = centred.loc[both_rated, item_b]
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


for target_item in ["Item1", "Item6"]:
    print("Similarity of", target_item, "with the other items:")
    for item in ratings.columns:
        print("  ", item, round(adjusted_cosine(target_item, item), 3))
    print()


# ------------------------------------------------------------
# Step 3: predict from the 2 most similar items that User3 has rated
# ------------------------------------------------------------
for target_item in ["Item1", "Item6"]:
    # similarities to items that User3 has rated
    sims = {}
    for item in ratings.columns:
        if item != target_item and not np.isnan(ratings.loc[target_user, item]):
            sims[item] = adjusted_cosine(target_item, item)

    best_two = sorted(sims, key=sims.get, reverse=True)[:2]

    top = 0
    bottom = 0
    for item in best_two:
        top = top + sims[item] * ratings.loc[target_user, item]
        bottom = bottom + abs(sims[item])

    print("Prediction for", target_user, target_item, "using", best_two, "=", round(top / bottom, 2))

print()
print("Same conclusion as the user-based method: recommend Item1, not Item6.")
