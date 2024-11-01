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

movie_keyword = pd.read_csv('./data/movies/keywords.csv')
movie_data.id = movie_data.id.astype(int)
movie_data = pd.merge(movie_data, movie_keyword, on='id')

movie_data['genres'] = movie_data['genres'].apply(literal_eval)
movie_data['genres'] = movie_data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

movie_data['keywords'] = movie_data['keywords'].apply(literal_eval)
movie_data['keywords'] = movie_data['keywords'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

tfidf_vector = TfidfVectorizer()
tfidf_matrix = tfidf_vector.fit_transform(movie_data['genres'] + " " + movie_data['keywords']).toarray()
tfidf_matrix_feature = tfidf_vector.get_feature_names_out()

tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=tfidf_matrix_feature, index = movie_data.title)

# 유사도 구하기
# tf-idf vector를 코사인 유사도를 활용해 유사도 값을 구함
cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(cosine_sim, index = movie_data.title, columns = movie_data.title)

# 3. Content Based Recommend
# target title(추천 결과를 조회할 영화 제목)에 따라 코사인 유사도를 구한 matrix에서 유사도 데이터를 가져옴
# 유사도 데이터 중 가장 유사도 값이 큰 데이터를 가져옴(가져올 대 top K개를 가져옴)

def genre_recommendations(target_title, matrix, items, k=10):
    recom_idx = matrix.loc[:, target_title].values.reshape(1, -1).argsort()[:, ::-1].flatten()[1:k+1]
    recom_title = items.iloc[recom_idx, :].title.values
    recom_genre = items.iloc[recom_idx, : ].genres.values
    target_title_list = np.full(len(range(k)), target_title)
    target_genre_list = np.full(len(range(k)), items[items.title == target_title].genres.values)
    d = {
        'target_title' : target_title_list,
        'target_genre' : target_genre_list,
        'recom_title' : recom_title,
        'recom_genre' : recom_genre
    }
    return pd.DataFrame(d)

res = genre_recommendations('The Dark Knight Rises', cosine_sim_df, movie_data)
print(res)