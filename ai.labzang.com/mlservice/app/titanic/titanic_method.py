from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from pandas import DataFrame
from app.titanic.titanic_dataset import TitanicDataSet
import logging

logger = logging.getLogger(__name__)

class TitanicMethod(object): 

    def __init__(self):
        self.dataset = TitanicDataSet()

    def read_csv(self, fname: str) -> pd.DataFrame:
        return pd.read_csv(fname)

    def create_df(self, df: DataFrame, label: str) -> pd.DataFrame:
        """DataFrame에서 label 컬럼을 제거 (컬럼이 존재하는 경우에만)"""
        if label in df.columns:
            return df.drop(columns=[label])
        else:
            # label 컬럼이 없으면 그대로 반환 (test 데이터의 경우)
            return df

    def create_label(self, df: DataFrame, label: str) -> pd.DataFrame:
        return df[[label]]

    def drop_feature(self, this, *feature: str) -> object:
        [i.drop(j, axis=1, inplace=True) for j in feature for i in [this.train,this.test ] ]

        # for i in [this.train, this.test]:
        #     for j in feature:
        #         i.drop(j, axis=1, inplace=True)
 
        return this

    def check_null(self, data) -> int:
        """
        데이터셋의 null 값 개수를 확인하고 반환
        Args:
            data: DataFrame 또는 TitanicDataSet 객체
        Returns:
            null 값의 총 개수
        """
        if isinstance(data, DataFrame):
            # DataFrame을 직접 받은 경우
            total_nulls = int(data.isnull().sum().sum())
            if total_nulls > 0:
                null_counts = data.isnull().sum()
                null_cols = null_counts[null_counts > 0]
                logger.info(f"[Null 값 확인] 총 {total_nulls}개")
                if len(null_cols) > 0:
                    logger.info(f"  Null이 있는 컬럼:\n{null_cols.to_string()}")
            return total_nulls
        else:
            # TitanicDataSet 객체를 받은 경우
            for dataset_name, dataset in [("Train", data.train), ("Test", data.test)]:
                null_counts = dataset.isnull().sum()
                total_nulls = null_counts.sum()
                logger.info(f"[{dataset_name} Null 값 확인]")
                logger.info(f"  총 Null 값 개수: {total_nulls}개")
                if total_nulls > 0:
                    null_cols = null_counts[null_counts > 0]
                    logger.info(f"  Null이 있는 컬럼:\n{null_cols.to_string()}")
            return int(data.train.isnull().sum().sum() + data.test.isnull().sum().sum())
    
    def extract_title_from_name(self, this):
        # for i in [this.train, this.test]:
        #     i['Title'] = i['Name'].str.extract('([A-Za-z]+)\.', expand=False) 

        [i.__setitem__('Title', i['Name'].str.extract('([A-Za-z]+)\.', expand=False)) 
         for i in [this.train, this.test]]
            # expand=False 는 시리즈 로 추출
        return this
    

    def remove_duplicate_title(self, this):
        a = []
        for i in [this.train, this.test]:
            # a.append(i['Title'].unique())
            a += list(set(i['Title'])) # train, test 두번을 누적해야 해서서
        a = list(set(a)) # train, test 각각은 중복이 아니지만, 합치면서 중복발생
        logger.info(f"[Title 목록] {sorted(a)}")
        # ['Mr', 'Miss', 'Dr', 'Major', 'Sir', 'Ms', 'Master', 'Capt', 'Mme', 'Mrs', 
        #  'Lady', 'Col', 'Rev', 'Countess', 'Don', 'Mlle', 'Dona', 'Jonkheer']
        '''
        ['Mr', 'Sir', 'Major', 'Don', 'Rev', 'Countess', 'Lady', 'Jonkheer', 'Dr',
        'Miss', 'Col', 'Ms', 'Dona', 'Mlle', 'Mme', 'Mrs', 'Master', 'Capt']
        Royal : ['Countess', 'Lady', 'Sir']
        Rare : ['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme' ]
        Mr : ['Mlle']
        Ms : ['Miss']
        Master
        Mrs
        '''
        title_mapping = {'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        
        return title_mapping
    

    def title_nominal(self, this):
        # Title 매핑 정의
        title_mapping = {
            'Mr': 1,
            'Ms': 2,
            'Mrs': 3,
            'Master': 4,
            'Royal': 5,
            'Rare': 6
        }
        
        for i in [this.train, this.test]:
            i['Title'] = i['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            i['Title'] = i['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme'], 'Rare')
            i['Title'] = i['Title'].replace(['Mlle'], 'Mr')
            i['Title'] = i['Title'].replace(['Miss'], 'Ms')
            # Master 는 변화없음
            # Mrs 는 변화없음
            i['Title'] = i['Title'].fillna(0)
            i['Title'] = i['Title'].map(title_mapping)
            # 매핑되지 않은 값은 0으로 처리
            i['Title'] = i['Title'].fillna(0)
            
        return this      
        


    def pclass_ordinal(self, this):
        return this

    def gender_nominal(self, this):

        gender_mapping = {'male': 0, 'female': 1}
        # for i in [this.train, this.test]:
        #     i["Gender"] = i["Sex"].map(gender_mapping)
        [i.__setitem__('Gender',i['Sex'].map(gender_mapping)) 
         for i in [this.train, this.test]]
        return this

    def age_ratio(self, this):
        
        self.get_count_of_null(this,"Age")
        for i in [this.train, this.test]:
            i['Age'] = i['Age'].fillna(-0.5)
        self.get_count_of_null(this,"Age")
        train_max_age = max(this.train['Age'])
        test_max_age = max(this.test['Age'])
        max_age = max(train_max_age, test_max_age)
        logger.info(f"[Age 처리] 최고령자: {max_age}세")
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult', 'Senior']
        age_mapping = {'Unknown':0 , 'Baby': 1, 'Child': 2, 'Teenager' : 3, 'Student': 4,
                       'Young Adult': 5, 'Adult':6,  'Senior': 7}
        for i in [this.train, this.test]:
            i['AgeGroup'] = pd.cut(i['Age'], bins, labels=labels).map(age_mapping)
        
        return this
    
    def get_count_of_null( self, this , feature):
        for dataset_name, dataset in [("Train", this.train), ("Test", this.test)]:
            null_count = dataset[feature].isnull().sum()
            logger.info(f"[{dataset_name}] {feature} 컬럼의 Null 값 개수: {null_count}개")
    

    def fare_ordinal(self, this):
        for i in [this.train, this.test]:
            i['FareBand'] = pd.qcut(i['Fare'], 4, labels={1,2,3,4})

        this.train = this.train.fillna({'FareBand': 1})
        this.test = this.test.fillna({'FareBand': 1})
        
        return this


    def embarked_nominal(self, this):
        for i in [this.train, this.test]:
            i['Embarked'] = i['Embarked'].fillna('S')# 사우스햄튼이 가장 많으니까
        embarked_mapping = {'S':1, 'C':2, 'Q':3}
        this.train['Embarked'] = this.train['Embarked'].map(embarked_mapping)
        this.test['Embarked'] = this.test['Embarked'].map(embarked_mapping)
        return this

    def kwargs_sample(**kwargs) -> None:
        # for key, value in kwargs.items():
        #     print(f'키워드: {key} 값: {value}')
        {print(''.join(f'키워드: {key} 값: {value}')) for key, value in kwargs.items()}

