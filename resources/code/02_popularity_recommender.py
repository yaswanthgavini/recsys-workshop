"""
02 - The simplest recommender: recommend popular movies

What this file does:
  Step 1. Compute the average rating and number of ratings of every movie
  Step 2. Attempt 1: sort by average rating          -> goes wrong
  Step 3. Attempt 2: sort by a "damped" average       -> works
  Step 4. See how the damping changes single movies

Run with:  python 02_popularity_recommender.py
"""

import pandas as pd

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")


# ------------------------------------------------------------
# Step 1: average rating and number of ratings per movie
# ------------------------------------------------------------
movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])

# add the movie titles so the output is readable
movie_stats = movie_stats.join(movies.set_index("movieId")["title"])

print("A few rows of movie_stats:")
print(movie_stats.head())
print()


# ------------------------------------------------------------
# Step 2: Attempt 1 - sort by the plain average
# ------------------------------------------------------------
by_average = movie_stats.sort_values(["mean", "count"], ascending=False)

print("Top 5 by plain average rating:")
print(by_average.head(5))
print("Problem: these movies have only 1 or 2 ratings. We cannot trust that.")
print()


# ------------------------------------------------------------
# Step 3: Attempt 2 - damped (Bayesian) average
#
#   score = (count * movie_average + m * overall_average) / (count + m)
#
#   It is like giving every movie m extra "average" ratings.
#   Movies with few ratings are pulled towards the overall average.
#   Movies with many ratings keep (almost) their own average.
# ------------------------------------------------------------
overall_average = ratings["rating"].mean()
m = 50

movie_stats["score"] = ((movie_stats["count"] * movie_stats["mean"] + m * overall_average)
                        / (movie_stats["count"] + m))

by_score = movie_stats.sort_values("score", ascending=False)

print("Overall average rating:", round(overall_average, 2))
print("Top 10 by damped average (m = 50):")
print(by_score.head(10).round(2))
print()


# ------------------------------------------------------------
# Step 4: what the damping does to single movies
# ------------------------------------------------------------
def damped_average(count, movie_average):
    return (count * movie_average + m * overall_average) / (count + m)


print("Effect of damping (m = 50):")
print("  1 rating,    average 5.0  -> score", round(damped_average(1, 5.0), 2))
print("  5 ratings,   average 5.0  -> score", round(damped_average(5, 5.0), 2))
print("  300 ratings, average 4.4  -> score", round(damped_average(300, 4.4), 2))
print()
print("Note: everyone gets the SAME list. That is why we need personalised methods.")
