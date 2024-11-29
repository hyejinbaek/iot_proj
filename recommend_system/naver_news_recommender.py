# 005. naver news recommender
# 1. 가변 길이의 콘텐츠를 고정 길이의 벡터로 만듬
# 2. 사용자 히스토리 생성
# 3. 가변 길이의 콘텐츠를 활용해 만든 고정 길이의 벡터를 기반으로 평균화
# 4. 최정적으로 평균화 된 벡터값을 활용해 각 content끼리의 cosine similarity를 구하고 가장 유사한 뉴스 기사 추천
# 경제, 정치, IT/과학
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab


def make_doc2vec_models(tagged_data, tok, vector_size = 128, window = 3, epochs = 40, min_count = 0, workers = 4):
    model = Doc2Vec(tagged_data, vector_size = vector_size, window = window, epochs = epochs, min_count = min_count, workers = workers)
    model.save(f'./datas/{tok}_news_model.doc2vec')

def get_data():
    economy = pd.read_csv('./data/naver_news/economy.csv')
    policy = pd.read_csv('./data/naver_news/policy.csv')
    it = pd.read_csv('./data/naver_news/it.csv')
    columns = ['date', 'category', 'company', 'title', 'content', 'url']
    economy.columns = columns
    policy.columns = columns
    it.columns = columns
    
    data = pd.concat([economy, policy, it], axis = 0)
    data.reset_index(drop=True, inplace=True)
    
    return data

def get_preprocessing_data(data):
    data.drop(['date', 'company', 'url'], axis = 1, inplace =True)
    
    category_mapping = {
    'economy' : 0,
    'policy' : 1,
    'it' : 2
    }

    data['category'] = data['category'].map(category_mapping)
    data['title_content'] = data['title'] + " " + data['content']
    data.drop(['title', 'content'], axis = 1, inplace = True)
    
    return data


def make_doc2vec_data(data, column, t_document=False):
    data_doc = []
    for tag, doc in zip(data.index, data[column]):
        # Ensure doc is a string before calling split
        if isinstance(doc, str):
            doc = doc.split(" ")
        else:
            doc = []  # If it's not a string, set it to an empty list
        data_doc.append(([tag], doc))
    
    if t_document:
        data = [TaggedDocument(words=text, tags=tag) for tag, text in data_doc]
        return data
    else:
        return data_doc

    
def get_recommened_contents(user, data_doc, model):
    scores = []
    
    for tags, text in data_doc:
        trained_doc_vec = model.dv[tags[0]]
        scores.append(cosine_similarity(user.reshape(-1, 128), trained_doc_vec.reshape(-1, 128)))
        
    scores = np.array(scores).reshape(-1)
    scores = np.argsort(-scores)[:5]
    
    return data.loc[scores, :]

def make_user_embedding(index_list, data_doc, model):
    user = []
    user_embedding = []
    for i in index_list:
        user.append(data_doc[i][0][0])
    for i in user:
        user_embedding.append(model.dv[i])
    user_embedding = np.array(user_embedding)
    user = np.mean(user_embedding, axis = 0)
    return user

def view_user_history(data):
    print(data[['category', 'title_content']])
    
    
data = get_data()
# print(data[['title', 'content']].isnull().sum())  # 결측치 개수 확인
# print(data[['title', 'content']].head())  # 샘플 데이터 확인

processed_data = get_preprocessing_data(data)  # 전처리 수행

# processed_data.to_csv("processed_data.csv", index = False)
# print(processed_data.head())
# data_doc_title_tag = make_doc2vec_data(data, 'title_content',)

prerpo = pd.read_csv('./data/naver_news/processed_data_with_morphs.csv')

data_doc_title_content_tag = make_doc2vec_data(prerpo, 'title_content', t_document=True)
data_doc_title_content = make_doc2vec_data(prerpo, 'title_content')
data_doc_tok_tag = make_doc2vec_data(prerpo, 'mecab_tok', t_document=True)
data_doc_tok = make_doc2vec_data(prerpo, 'mecab_tok')

# doc2vec model 생성
make_doc2vec_model = make_doc2vec_models(data_doc_title_content_tag, tok=False)
make_doc2vec_model_2 = make_doc2vec_models(data_doc_tok_tag, tok=True)

model_title_content = Doc2Vec.load('./datas/False_news_model.doc2vec')
model_tok = Doc2Vec.load('./datas/True_news_model.doc2vec')

user_category_1 = data.loc[random.sample(data.loc[data.category == 0, :].index.values.tolist(), 5), :]  #경제

user_category_2 = data.loc[random.sample(data.loc[data.category == 1, :].index.values.tolist(), 5), :]  #정치

user_category_3 = data.loc[random.sample(data.loc[data.category == 2, :].index.values.tolist(), 5), :]  #IT 과학

user_1 = make_user_embedding(user_category_1.index.values.tolist(), data_doc_title_content, model_title_content) # 경제
user_2 = make_user_embedding(user_category_2.index.values.tolist(), data_doc_title_content, model_title_content) # 정치
user_3 = make_user_embedding(user_category_3.index.values.tolist(), data_doc_title_content, model_title_content) # IT과학

result = get_recommened_contents(user_1, data_doc_title_content, model_title_content)
print(pd.DataFrame(result.loc[:, ['category', 'title_content']]))

result = get_recommened_contents(user_2, data_doc_title_content, model_title_content)
# print(pd.DataFrame(result.loc[:, ['category', 'title_content']]))

result = get_recommened_contents(user_3, data_doc_title_content, model_title_content)
# print(pd.DataFrame(result.loc[:, ['category', 'title_content']]))

# 형태소 분석 후 결과
user_1 = make_user_embedding(user_category_1.index.values.tolist(), data_doc_tok, model_tok) # 경제
user_2 = make_user_embedding(user_category_2.index.values.tolist(), data_doc_tok, model_tok) # 정치
user_3 = make_user_embedding(user_category_3.index.values.tolist(), data_doc_tok, model_tok) # IT과학


result = get_recommened_contents(user_1, data_doc_tok, model_tok)
print(pd.DataFrame(result.loc[:, ['category', 'title_content']]))