"""
01 - Explore the MovieLens data

What this file does:
  Step 1. Load the ratings and the movie titles
  Step 2. Count users, movies and ratings
  Step 3. Check the data quality
  Step 4. See how many ratings each user and each movie has
  Step 5. Measure how empty the user x movie table is
  Step 6. Draw three plots (saved as .png files)

Before running: put ratings.csv and movies.csv in the same folder as this file.
Run with:  python 01_explore_data.py
"""

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Step 1: Load the data
# ------------------------------------------------------------
ratings = pd.read_csv("ratings.csv")   # columns: userId, movieId, rating, timestamp
movies = pd.read_csv("movies.csv")     # columns: movieId, title, genres

print("First 5 rows of the ratings table:")
print(ratings.head())
print()


# ------------------------------------------------------------
# Step 2: Basic counts
# ------------------------------------------------------------
number_of_users = ratings["userId"].nunique()     # nunique() = how many different values
number_of_movies = ratings["movieId"].nunique()
number_of_ratings = len(ratings)                  # len() = number of rows
average_rating = ratings["rating"].mean()

print("Number of users:  ", number_of_users)
print("Number of movies: ", number_of_movies)
print("Number of ratings:", number_of_ratings)
print("Average rating:   ", round(average_rating, 2))
print()


# ------------------------------------------------------------
# Step 3: Data quality
# ------------------------------------------------------------
print("Missing values in each column:")
print(ratings.isnull().sum())             # isnull() marks empty cells, sum() counts them
print()

duplicates = ratings.duplicated(subset=["userId", "movieId"]).sum()
print("Same user rating the same movie twice:", duplicates)
print()

print("How often each rating value is used:")
print(ratings["rating"].value_counts().sort_index())
print()


# ------------------------------------------------------------
# Step 4: Ratings per user and per movie
# ------------------------------------------------------------
ratings_per_user = ratings.groupby("userId").size()     # for each user: how many ratings
ratings_per_movie = ratings.groupby("movieId").size()   # for each movie: how many ratings

print("Ratings per user:  smallest =", ratings_per_user.min(),
      " median =", ratings_per_user.median(),
      " largest =", ratings_per_user.max())
print("Ratings per movie: smallest =", ratings_per_movie.min(),
      " median =", ratings_per_movie.median(),
      " largest =", ratings_per_movie.max())
print()

print("The 10 most-rated movies:")
most_rated = ratings_per_movie.sort_values(ascending=False).head(10)
for movie_id, count in most_rated.items():
    title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
    print("  ", count, "ratings -", title)
print()

# The "long tail": a few movies get most of the ratings
movies_with_one_rating = (ratings_per_movie == 1).sum()
print("Movies with only ONE rating:", movies_with_one_rating)

sorted_counts = ratings_per_movie.sort_values(ascending=False)
top_ten_percent = int(len(sorted_counts) * 0.10)                 # how many movies make up 10%
share = sorted_counts.head(top_ten_percent).sum() / sorted_counts.sum()
print("The top 10% of movies receive", round(share * 100, 1), "% of all ratings")
print()


# ------------------------------------------------------------
# Step 5: How empty is the user x movie table?
# ------------------------------------------------------------
total_cells = number_of_users * number_of_movies    # every possible (user, movie) pair
filled_cells = number_of_ratings                    # pairs that actually have a rating

print("Cells in the user x movie table:", total_cells)
print("Filled:", round(filled_cells / total_cells * 100, 2), "%")
print("Empty: ", round((1 - filled_cells / total_cells) * 100, 2), "%")


# ------------------------------------------------------------
# Step 6: Plots  (in Jupyter you can use plt.show() instead of savefig)
# ------------------------------------------------------------
# Plot 1: how often each rating is given
plt.figure(figsize=(6, 4))
ratings["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("How often each rating is given")
plt.xlabel("rating")
plt.ylabel("number of ratings")
plt.savefig("plot_1_rating_distribution.png")
plt.close()

# Plot 2: how many ratings each user gave
plt.figure(figsize=(6, 4))
plt.hist(ratings_per_user, bins=50)
plt.title("How many ratings each user gave")
plt.xlabel("number of ratings")
plt.ylabel("number of users")
plt.savefig("plot_2_ratings_per_user.png")
plt.close()

# Plot 3: the long tail
plt.figure(figsize=(6, 4))
plt.plot(sorted_counts.values)
plt.yscale("log")                      # log scale, otherwise the tail is flat on the floor
plt.title("The long tail: ratings per movie")
plt.xlabel("movies, most popular first")
plt.ylabel("number of ratings (log scale)")
plt.savefig("plot_3_long_tail.png")
plt.close()

print()
print("Saved: plot_1_rating_distribution.png, plot_2_ratings_per_user.png, plot_3_long_tail.png")
