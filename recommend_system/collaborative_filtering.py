# https://github.com/lsjsj92/recommender_system_with_Python/blob/master/003.%20recommender%20system%20basic%20with%20Python%20-%202%20Collaborative%20Filtering.ipynb
# 추천 시스템 - 아이템 기반 협업 필터링(item based collaborative filtering)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('./data/movies/ratings_small.csv')

data = data.pivot_table('rating', index = 'userId', columns = 'movieId')

ratings = pd.read_csv('./data/movies/ratings_small.csv')

movies = pd.read_csv('./data/tmdb_5000_movies.csv')
movies = movies.rename(columns = {'id': 'movieId'})

ratings_movies = pd.merge(ratings, movies, on = 'movieId')

data = ratings_movies.pivot_table('rating', index = 'userId', columns='title').fillna(0)

data = data.transpose()

movie_sim = cosine_similarity(data, data)
movie_sim_df = pd.DataFrame(data = movie_sim, index = data.index, columns = data.index)

# print(movie_sim_df["X-Men Origins: Wolverine"].sort_values(ascending=False)[1:10])
# print("===================================================================================================")
# print(movie_sim_df["Harry Potter and the Half-Blood Prince"].sort_values(ascending=False)[1:10])
# print("===================================================================================================")
# print(movie_sim_df["Harry Potter and the Half-Blood Prince"].sort_values(ascending=False)[:10])
