"""
08 - Evaluation metrics on small hand-made examples

What this file does:
  Part A. Rating error:   RMSE and MAE
  Part B. Ranking quality: Precision@K and Recall@K
  Part C. Order matters:   NDCG@K

Run with:  python 08_evaluation_metrics.py
"""

import numpy as np


# ------------------------------------------------------------
# Part A: RMSE and MAE
# ------------------------------------------------------------
true_ratings = np.array([5, 4, 1, 2, 5, 3])
predicted = np.array([4.0, 4.5, 2.0, 3.0, 3.5, 3.0])

errors = true_ratings - predicted
rmse = np.sqrt(np.mean(errors ** 2))
mae = np.mean(np.abs(errors))

print("Part A")
print("  true:     ", true_ratings)
print("  predicted:", predicted)
print("  errors:   ", errors)
print("  RMSE =", round(rmse, 3), "   MAE =", round(mae, 3))
print()


# ------------------------------------------------------------
# Part B: Precision@K and Recall@K
#   "relevant" = the user really rated it 4 or more
#   Precision@K = how many of the K shown items are relevant, divided by K
#   Recall@K    = how many of ALL relevant items made it into the top K
# ------------------------------------------------------------
K = 2
order = np.argsort(-predicted)          # item positions, best predicted first
shown = order[:K]                       # the K items the user would see
relevant = true_ratings >= 4            # True / False for each item

hits = relevant[shown].sum()
precision = hits / K
recall = hits / relevant.sum()

print("Part B")
print("  items shown (best predicted first):", shown)
print("  relevant items:", np.where(relevant)[0])
print("  Precision@2 =", precision, "   Recall@2 =", round(recall, 3))
print()


# ------------------------------------------------------------
# Part C: NDCG@K - rewards putting the BEST items at the TOP
#   relevance: 3 = loved it, 2 = liked, 1 = ok, 0 = not relevant
# ------------------------------------------------------------
def dcg(relevances):
    total = 0
    for position, rel in enumerate(relevances, start=1):
        total = total + (2 ** rel - 1) / np.log2(position + 1)
    return total


def ndcg(relevances):
    best_possible = dcg(sorted(relevances, reverse=True))
    return dcg(relevances) / best_possible


list_a = [3, 2, 0, 1, 2]     # good items near the top
list_b = [0, 1, 2, 2, 3]     # same items, good ones at the bottom

print("Part C")
print("  list A", list_a, "  NDCG =", round(ndcg(list_a), 3))
print("  list B", list_b, "  NDCG =", round(ndcg(list_b), 3))
print("  Same items, so Precision is the same - but NDCG prefers list A.")
