"""
타이타닉 데이터 서비스
판다스, 넘파이, 사이킷런을 사용한 데이터 처리 및 머신러닝 서비스
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, ParamSpecArgs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from icecream import ic
from app.titanic.titanic_method import TitanicMethod

# 공통 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TitanicService:
    """타이타닉 데이터 처리 및 머신러닝 서비스"""
    
    def __init__(self):
        pass

    def preprocess(self):
        ic("😎😎 전처리 시작")
        the_method = TitanicMethod()
        df_train = the_method.new_model('train.csv')
        this_train = the_method.create_train(df_train, 'Survived')
        ic(f'1. Train 의 type \n {type(this_train)} ')
        ic(f'2. Train 의 column \n {this_train.columns} ')
        ic(f'3. Train 의 상위 5개 행\n {this_train.head(5)} ')
        ic(f'4. Train 의 null 의 갯수\n {the_method.check_null(this_train)}개')
        
        drop_features = ['SibSp', 'Parch', 'Cabin', 'Ticket']
        this_train = the_method.drop_feature(this_train, *drop_features)
        this_train = the_method.pclass_ordinal(this_train)
        this_train = the_method.fare_ordinal(this_train)
        this_train = the_method.embarked_ordinal(this_train)
        this_train = the_method.gender_nominal(this_train)
        this_train = the_method.age_ratio(this_train)
        this_train = the_method.title_nominal(this_train)
        drop_name = ['Name']
        this_train = the_method.drop_feature(this_train, *drop_name)
        ic("😎😎 전처리 완료")
        ic(f'1. Train 의 type \n {type(this_train)} ')
        ic(f'2. Train 의 column \n {this_train.columns} ')
        ic(f'3. Train 의 상위 5개 행\n {this_train.head(5)} ')
        ic(f'4. Train 의 null 의 갯수\n {the_method.check_null(this_train)}개')
        

    def modeling(self):
        ic("😎😎 모델링 시작")
        ic("😎😎 모델링 완료")

    def learning(self):
        ic("😎😎 학습 시작")
        ic("😎😎 학습 완료")

    def evaluate(self):
        ic("😎😎 평가 시작")
        ic("😎😎 평가 완료")


    def submit(self):
        ic("😎😎 제출 시작")
        ic("😎😎 제출 완료")