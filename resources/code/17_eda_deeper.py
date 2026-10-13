"""
17 - Exploring the data in more depth (Lab 1)

File 01 counted things. This file asks the questions that matter for a recommender:
who rates, what do they rate, and how does that skew the data?

What this file does:
  Step 1. Check the values and types (is the data valid?)
  Step 2. Ratings over time
  Step 3. User behaviour: generous vs strict users, heavy vs light users
  Step 4. Movie behaviour: do popular movies get higher ratings?
  Step 5. Genres and release decades
  Step 6. Four plots in one picture (saved as a .png file)

Run with:  python 17_eda_deeper.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings["year"] = ratings["date"].dt.year


# ------------------------------------------------------------
# Step 1: are the values valid?
# ------------------------------------------------------------
print("Step 1: values and types")
print("   data types:", dict(ratings.dtypes.astype(str)[["userId", "movieId", "rating"]]))
print("   rating values:", sorted(ratings["rating"].unique()))
print("   ids are whole numbers, ratings run from 0.5 to 5.0 in half steps")

whole = (ratings["rating"] == ratings["rating"].round()).mean()
print("   whole-star ratings:", round(whole * 100, 1), "%  half-star ratings:", round((1 - whole) * 100, 1), "%")
print()


# ------------------------------------------------------------
# Step 2: ratings over time
# ------------------------------------------------------------
per_year = ratings.groupby("year").size()
print("Step 2: ratings per year")
print("   busiest year:", per_year.idxmax(), "with", per_year.max(), "ratings")
print("   years with fewer than 1,000 ratings:", (per_year < 1000).sum(), "of", len(per_year))

first_rating = ratings.groupby("userId")["date"].min().dt.year
print("   most users joined in:", first_rating.value_counts().head(3).to_dict(), " (year: users)")
print()


# ------------------------------------------------------------
# Step 3: user behaviour
#   generosity = a user's average rating
#   activity   = how many ratings a user gave (split into 4 equal-sized groups)
# ------------------------------------------------------------
users = ratings.groupby("userId")["rating"].agg(["mean", "count"])
users["activity"] = pd.qcut(users["count"], 4, labels=["light", "medium", "heavy", "very heavy"])

print("Step 3: users")
print("   user average rating: smallest", round(users["mean"].min(), 2),
      " median", round(users["mean"].median(), 2), " largest", round(users["mean"].max(), 2))
print("   -> some users are far more generous or strict than others (rating-scale bias)")

top_users = users["count"].nlargest(int(0.1 * len(users))).sum() / len(ratings)
print("   the top 10% of users give", round(top_users * 100), "% of all ratings (selection bias)")

print("   average rating by activity group:")
print(users.groupby("activity", observed=True)["mean"].agg(["mean", "count"]).round(2))
print()


# ------------------------------------------------------------
# Step 4: movie behaviour
#   Do movies with many ratings get better ratings? (popularity bias)
# ------------------------------------------------------------
stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
correlation = stats["mean"].corr(np.log(stats["count"]))

print("Step 4: movies")
print("   correlation between average rating and log(number of ratings):", round(correlation, 2))

stats["group"] = pd.cut(stats["count"], [0, 4, 19, 99, 10000], labels=["1-4", "5-19", "20-99", "100+"])
print("   average rating by number of ratings the movie received:")
print(stats.groupby("group", observed=True)["mean"].agg(["mean", "count"]).round(2))
print("   Popular movies are rated higher, and rarely-rated movies are noisy.")
print()


# ------------------------------------------------------------
# Step 5: genres and decades
# ------------------------------------------------------------
movies["year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0])
movies["decade"] = (movies["year"] // 10 * 10)

with_genre = ratings.merge(movies[["movieId", "genres"]], on="movieId")
with_genre["genre"] = with_genre["genres"].str.split("|")
by_genre = with_genre.explode("genre").groupby("genre")["rating"].agg(["mean", "count"])
by_genre = by_genre[by_genre["count"] >= 1000].sort_values("mean", ascending=False)

print("Step 5: genres (with at least 1,000 ratings)")
print("   highest rated:", ", ".join(by_genre.index[:3]), " lowest rated:", ", ".join(by_genre.index[-3:]))
print("   most rated genre:", by_genre["count"].idxmax(), "(", by_genre["count"].max(), "ratings )")

with_year = ratings.merge(movies[["movieId", "decade"]], on="movieId")
by_decade = with_year.groupby("decade")["rating"].agg(["mean", "count"])
print("   ratings by release decade (last 4 decades):")
print(by_decade.tail(4).round(2))
print()


# ------------------------------------------------------------
# Step 6: four plots in one picture
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].plot(per_year.index, per_year.values, marker="o")
axes[0, 0].set_title("Ratings per year")

axes[0, 1].hist(users["mean"], bins=30)
axes[0, 1].set_title("Average rating of each user")
axes[0, 1].set_xlabel("user's average rating")

axes[1, 0].scatter(stats["count"], stats["mean"], s=4, alpha=0.4)
axes[1, 0].set_xscale("log")
axes[1, 0].set_title("Movie average vs number of ratings")
axes[1, 0].set_xlabel("number of ratings (log scale)")

axes[1, 1].barh(by_genre.index, by_genre["mean"])
axes[1, 1].set_xlim(3.0, 4.2)
axes[1, 1].set_title("Average rating by genre")

plt.tight_layout()
plt.savefig("plot_6_eda_deeper.png", dpi=110)
plt.close()
print("Step 6: saved plot_6_eda_deeper.png")
