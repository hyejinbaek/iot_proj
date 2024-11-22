# 협업 필터링 Matrix Factorization
# https://github.com/lsjsj92/recommender_system_with_Python/blob/master/004.%20recommender%20system%20basic%20with%20Python%20-%203%20Matrix%20Factorization.ipynb

import kagglehub
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Download latest version
path = kagglehub.dataset_download("sengzhaotoo/movielens-small")

# print("Path to dataset files:", path)
# 예: 다운로드 경로와 파일 이름 조합
rating_file_path = f"{path}/ratings.csv"  # 데이터셋에 따라 파일명 변경
movie_file_path = f"{path}/movies.csv"


rating_data = pd.read_csv(rating_file_path)
movie_data = pd.read_csv(movie_file_path)

# inplace=True : None 반환(변수에 덮어씀)
# inplace=False(기본값) : 수정된 새로운 df 반환
rating_data.drop('timestamp', axis = 1, inplace =True)
# print(rating_data)

movie_data.drop('genres', axis = 1, inplace=True)
# print(movie_data)

# pivot table 생성
# pivot table 생성 이유 : 데이터 계산, 요약 및 분석하는 강력한 도구임. 데이터 비교/패턴 및 추세를 보는데 사용 가능
user_movie_data = pd.merge(rating_data, movie_data, on = 'movieId')
# print(user_movie_data)

# 사용자-영화 기준 데이터 pivot table
user_movie_rating = user_movie_data.pivot_table('rating', 
                                                index = 'userId', 
                                                columns='title').fillna(0)
# print(user_movie_rating.head())


# 영화-사용자 기준 데이터
movie_user_rating = user_movie_rating.values.T
# print(movie_user_rating)

# SVD(Singular Value Decomposion) : 특이값 분해
# TruncatedSVD : 시그마행렬(대각원소) 가운데 상위 n개만 골라낸 것
SVD = TruncatedSVD(n_components=12)
# 차원축소
matrix = SVD.fit_transform(movie_user_rating)
# print(matrix[0])

# 피어슨 상관계수
corr = np.corrcoef(matrix)

corr2 = corr[:200, :200]

plt.figure(figsize=(16, 10))
sns.heatmap(corr2)
# plt.title("Correlation Heatmap")
# plt.savefig("correlation_heatmap.png")  # 히트맵 이미지 저장
# plt.show()

movie_title = user_movie_rating.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("Guardians of the Galaxy (2014)")

corr_coffey_hands = corr[coffey_hands]

# 가디언즈 갤럭시 영화 기준으로 비슷한 영화 추출
# print(list(movie_title[(corr_coffey_hands >= 0.9)])[:50])


# 한 사용자에게 개인 추천 해주기
df_ratings = pd.read_csv(rating_file_path)
df_movies = pd.read_csv(movie_file_path)

df_user_movie_ratings = df_ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# print(df_user_movie_ratings.head())

# pivot table을 numpy matrix으로 변경
matrix = df_user_movie_ratings.to_numpy()

user_ratings_mean = np.mean(matrix, axis=1)

matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)
# print(pd.DataFrame(matrix_user_mean, columns = df_user_movie_ratings.columns).head())

# scipy에서 제공해주는 svd.  
# U 행렬, sigma 행렬, V 전치 행렬을 반환.

U, sigma, Vt = svds(matrix_user_mean, k = 12)
# print(U.shape)
# print(sigma.shape)
# print(Vt.shape)

# 시그마 행렬이 1차원 행렬이기 때문에 0이 포함된 대칭행렬로 변환할 때는 numpy의 diag 사용
sigma = np.diag(sigma)


# U, Sigma, Vt의 내적을 수행하면, 다시 원본 행렬로 복원이 된다. 
# 거기에 + 사용자 평균 rating을 적용한다.
svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = df_user_movie_ratings.columns)


def recommend_movies(df_svd_preds, user_id, ori_movies_df, ori_ratings_df, num_recommendations=5):
    
    #현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
    user_row_number = user_id - 1 
    
    # 최종적으로 만든 pred_df에서 사용자 index에 따라 영화 데이터 정렬 -> 영화 평점이 높은 순으로 정렬 됌
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
    
    # 원본 평점 데이터에서 user id에 해당하는 데이터를 뽑아낸다. 
    user_data = ori_ratings_df[ori_ratings_df.userId == user_id]
    
    # 위에서 뽑은 user_data와 원본 영화 데이터를 합친다. 
    user_history = user_data.merge(ori_movies_df, on = 'movieId').sort_values(['rating'], ascending=False)
    
    # 원본 영화 데이터에서 사용자가 본 영화 데이터를 제외한 데이터를 추출
    recommendations = ori_movies_df[~ori_movies_df['movieId'].isin(user_history['movieId'])]
    # 사용자의 영화 평점이 높은 순으로 정렬된 데이터와 위 recommendations을 합친다. 
    recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = 'movieId')
    # 컬럼 이름 바꾸고 정렬해서 return
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
                      

    return user_history, recommendations


already_rated, predictions = recommend_movies(df_svd_preds, 330, df_movies, df_ratings, 10)

print(predictions)