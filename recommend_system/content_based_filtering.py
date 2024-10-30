# content based filtering
# dataset : TMDB 5000 Movie Dataset ( https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv )
# reference : https://github.com/lsjsj92/recommender_system_with_Python/blob/master/002.%20recommender%20system%20basic%20with%20Python%20-%201%20content%20based%20filtering.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. data preprocessing
data = pd.read_csv('./data/tmdb_5000_movies.csv')
data = data[['id','genres', 'vote_average', 'vote_count','popularity','title',  'keywords', 'overview']]

m = data['vote_count'].quantile(0.9) # m = 1838.4000000000015
data = data.loc[data['vote_count'] >= m]

C = data['vote_average'].mean() # C = 6.962993762993763

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return ( v / (v+m) * R ) + (m / (m + v) * C)

data['score'] = data.apply(weighted_rating, axis = 1)

data['genres'] = data['genres'].apply(literal_eval) # literal_eval : type 변경
data['keywords'] = data['keywords'].apply(literal_eval)
data['genres'] = data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x: " ".join(x))
data['keywords'] = data['keywords'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

# data.to_csv('./movie_data/pre_tmdb_5000_movies.csv', index = False)

# 2. 콘텐츠 기반 필터링 추천(Content based filtering)
movie_data = pd.read_csv('./data/movies/movies_metadata.csv')
movie_data =  movie_data.loc[movie_data['original_language'] == 'en', :]
movie_data = movie_data[['id', 'title', 'original_language', 'genres']]

print(movie_data.shape)
movie_data.head()
