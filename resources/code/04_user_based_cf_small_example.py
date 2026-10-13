"""
04 - User-based collaborative filtering on a small example (5 users, 6 items)

The question: what would User3 rate Item1 and Item6?

What this file does:
  Step 1. Make the small rating table (NaN = not rated)
  Step 2. Compute each user's average rating
  Step 3. Compute the Pearson similarity of User3 with every other user
  Step 4. Pick the 2 most similar users (the "neighbours")
  Step 5. Predict with a plain weighted average       -> too high
  Step 6. Predict with the mean-centred formula        -> sensible
  Step 7. Predict with z-scores

Example from Aggarwal, "Recommender Systems: The Textbook", Table 2.1
Run with:  python 04_user_based_cf_small_example.py
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Step 1: the rating table
# ------------------------------------------------------------
ratings = pd.DataFrame(
    [[7,      6,      7,      4, 5, 4],
     [6,      7,      np.nan, 4, 3, 4],
     [np.nan, 3,      3,      1, 1, np.nan],
     [1,      2,      2,      3, 3, 4],
     [1,      np.nan, 1,      2, 3, 3]],
    index=["User1", "User2", "User3", "User4", "User5"],
    columns=["Item1", "Item2", "Item3", "Item4", "Item5", "Item6"])

print("Rating table:")
print(ratings)
print()

target_user = "User3"


# ------------------------------------------------------------
# Step 2: each user's average rating (NaN is skipped automatically)
# ------------------------------------------------------------
user_average = ratings.mean(axis=1)
print("Average rating of each user:")
print(user_average)
print()


# ------------------------------------------------------------
# Step 3: Pearson similarity between two users
#         (only items that BOTH users have rated are used)
# ------------------------------------------------------------
def pearson(user_a, user_b):
    both_rated = ratings.loc[user_a].notna() & ratings.loc[user_b].notna()
    a = ratings.loc[user_a, both_rated] - user_average[user_a]
    b = ratings.loc[user_b, both_rated] - user_average[user_b]
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


def cosine(user_a, user_b):
    both_rated = ratings.loc[user_a].notna() & ratings.loc[user_b].notna()
    a = ratings.loc[user_a, both_rated]
    b = ratings.loc[user_b, both_rated]
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


similarity = {}
print("Similarity of", target_user, "with the other users:")
for other_user in ratings.index:
    if other_user == target_user:
        continue
    similarity[other_user] = pearson(target_user, other_user)
    print("  ", other_user, " cosine =", round(cosine(target_user, other_user), 3),
          "  pearson =", round(similarity[other_user], 3))
print("Look at User4: cosine says 'similar', Pearson says 'exact opposite'.")
print()


# ------------------------------------------------------------
# Step 4: the 2 most similar users
# ------------------------------------------------------------
neighbours = sorted(similarity, key=similarity.get, reverse=True)[:2]
print("The 2 nearest neighbours of", target_user, ":", neighbours)
print()


# ------------------------------------------------------------
# Step 5 and 6: predictions for Item1 and Item6
# ------------------------------------------------------------
for item in ["Item1", "Item6"]:
    raw_top = 0          # for the plain weighted average
    centred_top = 0      # for the mean-centred formula
    bottom = 0

    for n in neighbours:
        sim = similarity[n]
        rating = ratings.loc[n, item]
        raw_top = raw_top + sim * rating
        centred_top = centred_top + sim * (rating - user_average[n])
        bottom = bottom + abs(sim)

    plain_prediction = raw_top / bottom
    centred_prediction = user_average[target_user] + centred_top / bottom

    print(item, ": plain weighted average =", round(plain_prediction, 2),
          "   mean-centred =", round(centred_prediction, 2))

print("User3 never rates above 3, so 6.49 is clearly wrong.")
print("The mean-centred formula uses how much the neighbours liked the item")
print("compared with THEIR OWN average, and adds that to User3's average.")
print()


# ------------------------------------------------------------
# Step 7: z-score version (also removes differences in spread)
# ------------------------------------------------------------
user_std = ratings.std(axis=1)    # standard deviation of each user's ratings

for item in ["Item1", "Item6"]:
    top = 0
    bottom = 0
    for n in neighbours:
        z = (ratings.loc[n, item] - user_average[n]) / user_std[n]
        top = top + similarity[n] * z
        bottom = bottom + abs(similarity[n])
    prediction = user_average[target_user] + user_std[target_user] * top / bottom
    print(item, ": z-score prediction =", round(prediction, 2))
