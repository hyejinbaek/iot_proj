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
print(data)

