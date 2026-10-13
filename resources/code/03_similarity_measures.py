"""
03 - How similar are two users?

What this file does:
  Writes three similarity measures by hand and tries them on small examples:
    - Cosine similarity
    - Pearson correlation (cosine after subtracting each user's average)
    - Jaccard similarity (for clicks / "watched or not" data)

Run with:  python 03_similarity_measures.py
"""

import numpy as np


# ------------------------------------------------------------
# The three similarity functions
# ------------------------------------------------------------
def cosine_similarity(a, b):
    # multiply matching ratings and add them up
    top = np.sum(a * b)
    # length of each rating vector
    bottom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    return top / bottom


def pearson_similarity(a, b):
    # subtract each user's own average first ...
    a_centred = a - np.mean(a)
    b_centred = b - np.mean(b)
    # ... then use cosine
    return cosine_similarity(a_centred, b_centred)


def jaccard_similarity(items_a, items_b):
    common = items_a.intersection(items_b)    # watched by both
    together = items_a.union(items_b)         # watched by at least one
    return len(common) / len(together)


# ------------------------------------------------------------
# Example 1: two users, five movies both have rated
# ------------------------------------------------------------
user_1 = np.array([5, 3, 4, 4, 1])
user_2 = np.array([3, 1, 2, 3, 2])

print("Example 1: User 1 =", user_1, " User 2 =", user_2)
print("  cosine  =", round(cosine_similarity(user_1, user_2), 3))
print("  pearson =", round(pearson_similarity(user_1, user_2), 3))
print()


# ------------------------------------------------------------
# Example 2: same taste, but one user is generous and one is strict
# ------------------------------------------------------------
generous = np.array([5, 5, 4, 5])
strict = np.array([2, 2, 1, 2])

print("Example 2: generous =", generous, " strict =", strict)
print("  cosine  =", round(cosine_similarity(generous, strict), 3))
print("  pearson =", round(pearson_similarity(generous, strict), 3), " <- Pearson sees the same taste")
print()


# ------------------------------------------------------------
# Example 3: opposite taste
# ------------------------------------------------------------
opposite = np.array([1, 1, 5, 1])

print("Example 3: generous =", generous, " opposite =", opposite)
print("  cosine  =", round(cosine_similarity(generous, opposite), 3), " <- cosine still says 'quite similar'")
print("  pearson =", round(pearson_similarity(generous, opposite), 3), " <- Pearson says 'exact opposite'")
print()


# ------------------------------------------------------------
# Example 4: Jaccard on "which items did they watch"
# ------------------------------------------------------------
watched_by_user_1 = {1, 2, 3, 4, 6}
watched_by_user_2 = {1, 2, 3, 4, 5, 6}

print("Example 4: user 1 watched", watched_by_user_1, " user 2 watched", watched_by_user_2)
print("  jaccard =", round(jaccard_similarity(watched_by_user_1, watched_by_user_2), 3))
