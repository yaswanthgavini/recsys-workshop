"""
07 - Item-based collaborative filtering: "people who liked this also liked ..."

What this file does:
  Step 1. Keep movies with at least 20 ratings (rare movies give strange results)
  Step 2. Subtract each user's average rating
  Step 3. Compute the similarity between every pair of movies
  Step 4. Show the most similar movies for a few famous titles

Run with:  python 07_item_based_similar_movies.py
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
movie_title = movies.set_index("movieId")["title"]


# ------------------------------------------------------------
# Step 1: keep movies with at least 20 ratings
# ------------------------------------------------------------
ratings_per_movie = ratings.groupby("movieId").size()
popular_movies = ratings_per_movie[ratings_per_movie >= 20].index
ratings_popular = ratings[ratings["movieId"].isin(popular_movies)]
print("Movies kept:", len(popular_movies))


# ------------------------------------------------------------
# Step 2: user x movie table, subtract each user's average, empty -> 0
# ------------------------------------------------------------
table = ratings_popular.pivot_table(index="userId", columns="movieId", values="rating")
user_average = table.mean(axis=1)
centred = table.sub(user_average, axis=0).fillna(0)


# ------------------------------------------------------------
# Step 3: similarity between movies = similarity between COLUMNS
#   .T turns columns into rows, because cosine_similarity compares rows
# ------------------------------------------------------------
movie_similarity = cosine_similarity(centred.T)
movie_similarity = pd.DataFrame(movie_similarity, index=table.columns, columns=table.columns)


# ------------------------------------------------------------
# Step 4: most similar movies
# ------------------------------------------------------------
def similar_movies(title, how_many=5):
    movie_id = movies.loc[movies["title"] == title, "movieId"].values[0]
    sims = movie_similarity[movie_id].drop(movie_id)          # drop the movie itself
    best = sims.sort_values(ascending=False).head(how_many)
    print("Movies most similar to", title)
    for other_id, sim in best.items():
        print("   ", round(sim, 3), "-", movie_title[other_id])
    print()


similar_movies("Toy Story (1995)")
similar_movies("Matrix, The (1999)")
similar_movies("Godfather, The (1972)")
similar_movies("Titanic (1997)")
print("Titanic's neighbours look odd: similarity follows WHO rated the movies, not the genre.")
