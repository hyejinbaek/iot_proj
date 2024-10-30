import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# 1. load data(open dataset - movie recommendation data)
movies = pd.read_csv('./data/movies_recommendation_data.csv', index_col = 'Movie ID')
# print(movies)

# 2. create recommendation system (Assume "The Post")
## 정해진 값 : IMDB Rating = 7.2, Biography = Yes(1로 표시됨), Drama = Yes, Thriller = No(0으로 표시됨), Comedy = No, Crime = No, Mystery = No, History = Yes
post_data = {'IMDB Rating' : [7.2], 'Biography' : 1, 'Drama' : 1, 'Thriller' : 0, 'Comedy' : 0, 'Crime' : 0, 'Mystery' : 0, "History" : 1}
the_post = pd.DataFrame(data = post_data, index = None)
# print(post_data)
print(the_post)

feature_col = movies.drop(['Movie Name', 'Label'], axis = 1)
x = feature_col
# print(x)

# 3. using NN model to k=1|3|5
## algorithm : 'ball_tree', 'kd_tree', 'brute', 'auto'
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(x)
# distance : "the post"의 영화와 가까운 영화간의 거리값, 거리가 작을수록 더 유사한 영화
# indices : "the post"의 영화와 가까운 영화들의 인덱스 번호
distances, indices = model.kneighbors(the_post)
print(" === distances === ", distances)
print(" === indices === ", indices)

print('Recommendations for "The Post":\n')
for i in range(len(distances.flatten())):
  print('{0}: {1}, with a distance of {2}.'.format(i+1, movies['Movie Name'].iloc[indices.flatten()[i]],distances.flatten()[i]))