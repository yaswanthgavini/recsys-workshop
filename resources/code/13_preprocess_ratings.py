"""
13 - Data preprocessing before modelling (Lab 1)

Lab 1 is called "preprocessing and EDA". File 01 does the EDA. This file does the
preprocessing: turn the raw CSV files into clean tables that the recommenders can use.

What this file does:
  Step 1. Load the data and convert the timestamp to a real date
  Step 2. Pull the release year out of the title and split the genres
  Step 3. Remove movies and users with too few ratings
  Step 4. Build the user x movie matrix and compare the sparsity before and after
  Step 5. Mean-centre the ratings (what the CF methods need)
  Step 6. Save the clean table

Run with:  python 13_preprocess_ratings.py
"""

import pandas as pd

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")


# ------------------------------------------------------------
# Step 1: timestamp -> date
#   The timestamp is "seconds since 1970". pandas can turn it into a date.
# ------------------------------------------------------------
ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")

print("Ratings span from", ratings["date"].min().date(), "to", ratings["date"].max().date())
print()


# ------------------------------------------------------------
# Step 2: movie information
#   Title looks like "Toy Story (1995)"; the year is the last 4 digits in brackets.
#   Genres look like "Adventure|Animation|Children"; split them into a list.
# ------------------------------------------------------------
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]     # text -> year (or NaN)
movies["genre_list"] = movies["genres"].str.split("|")

print("Movies without a year in the title:", movies["year"].isnull().sum())
print("Movies with no genre listed:      ", (movies["genres"] == "(no genres listed)").sum())

genre_counts = movies.explode("genre_list")["genre_list"].value_counts()   # one row per (movie, genre)
print("Most common genres:")
print(genre_counts.head(5))
print()


# ------------------------------------------------------------
# Step 3: remove rare movies and rare users
#   A movie with 1 rating cannot have reliable neighbours (see file 01, the long tail).
#   Every MovieLens user already has >= 20 ratings, but we check anyway.
# ------------------------------------------------------------
min_movie_ratings = 5
min_user_ratings = 20

movie_counts = ratings.groupby("movieId").size()
user_counts = ratings.groupby("userId").size()

keep_movies = movie_counts[movie_counts >= min_movie_ratings].index
keep_users = user_counts[user_counts >= min_user_ratings].index

clean = ratings[ratings["movieId"].isin(keep_movies) & ratings["userId"].isin(keep_users)]

print("Before cleaning:", len(ratings), "ratings,", ratings["movieId"].nunique(), "movies,",
      ratings["userId"].nunique(), "users")
print("After cleaning: ", len(clean), "ratings,", clean["movieId"].nunique(), "movies,",
      clean["userId"].nunique(), "users")
print("We lose", round((1 - len(clean) / len(ratings)) * 100, 1), "% of the ratings but",
      round((1 - clean["movieId"].nunique() / ratings["movieId"].nunique()) * 100, 1), "% of the movies.")
print()


# ------------------------------------------------------------
# Step 4: user x movie matrix and its sparsity
# ------------------------------------------------------------
def empty_share(table):
    return 1 - table.notna().sum().sum() / table.size


matrix_before = ratings.pivot_table(index="userId", columns="movieId", values="rating")
matrix_after = clean.pivot_table(index="userId", columns="movieId", values="rating")

print("Matrix before cleaning:", matrix_before.shape, " empty =", round(empty_share(matrix_before) * 100, 1), "%")
print("Matrix after cleaning: ", matrix_after.shape, " empty =", round(empty_share(matrix_after) * 100, 1), "%")
print("Cleaning makes the matrix smaller and a bit denser, but it is still mostly empty.")
print()


# ------------------------------------------------------------
# Step 5: mean-centre each user's ratings
#   centred rating = rating - the user's own average
#   A generous user and a strict user now become comparable (see file 03).
# ------------------------------------------------------------
user_average = matrix_after.mean(axis=1)
centred = matrix_after.sub(user_average, axis=0)

print("Generous vs strict users (average rating):")
print("   most generous user:", round(user_average.max(), 2), "   strictest user:", round(user_average.min(), 2))
print("After centring, every user's average is", round(centred.mean(axis=1).abs().max(), 6))
print()


# ------------------------------------------------------------
# Step 6: save the clean table
# ------------------------------------------------------------
clean[["userId", "movieId", "rating", "timestamp"]].to_csv("ratings_clean.csv", index=False)
print("Saved: ratings_clean.csv (", len(clean), "rows )")
