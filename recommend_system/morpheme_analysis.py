# 데이터 형태소 분석
from konlpy.tag import Hannanum
import pandas as pd

# Hannanum 객체 생성
hannanum = Hannanum()

# 데이터 로드
data = pd.read_csv('./data/naver_news/processed_data.csv')

# 'title_content' 열의 NaN 값을 빈 문자열로 채우기
data['title_content'] = data['title_content'].fillna('')

# 형태소 분석하여 'mecab_tok' 컬럼에 추가
data['mecab_tok'] = data['title_content'].apply(lambda x: hannanum.morphs(str(x)))

# 결과를 CSV 파일로 저장
data.to_csv('./data/naver_news/processed_data_with_morphs.csv', index=False)

print("CSV 파일이 저장되었습니다.")
