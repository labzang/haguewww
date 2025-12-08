from pathlib import Path
import pandas as pd
import numpy as np
from pandas import DataFrame
from app.titanic.titanic_dataset import TitanicDataSet
from icecream import ic

class TitanicMethod(object): 

    def __init__(self):
        self.dataset = TitanicDataSet()

    def new_model(self, fname: str) -> pd.DataFrame:
        return pd.read_csv(fname)

    def create_train(self, df: DataFrame, label: str) -> pd.DataFrame:
        return df.drop(columns=[label])

    def create_label(self, df: DataFrame, label: str) -> pd.DataFrame:
        return df[[label]]

    def drop_feature(self, df: DataFrame, *feature: str) -> pd.DataFrame:
        return df.drop(columns=[x for x in feature])

    def check_null(self, df: DataFrame) -> int:
        return int(df.isnull().sum().sum())

    # 척도 : nominal , ordinal, interval, ratio

    def pclass_ordinal(self, df: DataFrame) -> pd.DataFrame:
        """
        Pclass: 객실 등급 (1, 2, 3)
        - 서열형 척도(ordinal)로 처리합니다.
        - 1등석 > 2등석 > 3등석이므로, 생존률 관점에서 1이 가장 좋고 3이 가장 안 좋습니다.
        """
        # Pclass는 이미 ordinal이므로 그대로 사용하되, 명시적으로 정수형으로 변환
        df = df.copy()
        df["Pclass"] = df["Pclass"].astype(int)
        # 기존 Pclass는 유지 (필요시 drop_feature로 제거 가능)
        return df

    def fare_ordinal(self, df: DataFrame) -> pd.DataFrame:
        """
        Fare: 요금 (연속형 ratio 척도이지만, 여기서는 구간화하여 서열형으로 사용)
        - 결측치를 중앙값으로 채우고, 사분위수로 binning하여 ordinal 피처 생성
        """
        df = df.copy()
        
        # 결측치를 중앙값으로 채우기
        if df["Fare"].isnull().any():
            median_fare = df["Fare"].median()
            df["Fare"].fillna(median_fare, inplace=True)
            ic(f"Fare 결측치 {df['Fare'].isnull().sum()}개를 중앙값 {median_fare}로 채웠습니다")
        
        # 사분위수로 binning하여 ordinal 피처 생성
        try:
            df["Fare_ordinal"] = pd.qcut(
                df["Fare"], 
                q=4, 
                labels=[0, 1, 2, 3],
                duplicates="drop"
            ).astype(int)
        except ValueError:
            # 중복값이 많아 qcut이 실패할 경우, cut 사용
            df["Fare"] = pd.cut(
                df["Fare"],
                bins=4,
                labels=[0, 1, 2, 3]
            ).astype(int)
        
        # 원본 Fare 컬럼은 유지
        return df

    def embarked_ordinal(self, df: DataFrame) -> pd.DataFrame:
        """
        Embarked: 탑승 항구 (C, Q, S)
        - 본질적으로는 nominal(명목) 척도이므로 one-hot encoding 사용
        """
        df = df.copy()
        
        # 결측치를 최빈값으로 채우기
        if df["Embarked"].isnull().any():
            mode_embarked = df["Embarked"].mode()[0] if not df["Embarked"].mode().empty else "S"
            df["Embarked"].fillna(mode_embarked, inplace=True)
            ic(f"Embarked 결측치를 최빈값 {mode_embarked}로 채웠습니다")
        
        # One-hot encoding
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
        df = pd.concat([df, embarked_dummies], axis=1)
        
        # 원본 Embarked 컬럼은 유지 (필요시 drop_feature로 제거 가능)
        return df

    def gender_nominal(self, df: DataFrame) -> pd.DataFrame:
        """
        Sex: 성별 (male, female)
        - nominal 척도이므로 이진 인코딩 또는 one-hot encoding 사용
        """
        df = df.copy()
        
        # One-hot encoding 사용
        sex_dummies = pd.get_dummies(df["Sex"], prefix="Sex")
        df = pd.concat([df, sex_dummies], axis=1)
        
        # 또는 이진 인코딩 (선택사항)
        # df["Sex_male"] = (df["Sex"] == "male").astype(int)
        # df["Sex_female"] = (df["Sex"] == "female").astype(int)
        
        # 원본 Sex 컬럼을 "Gender" 로 변경
        df.rename(columns={"Sex": "Gender"}, inplace=True)
        
        return df

    def age_ratio(self, df: DataFrame) -> pd.DataFrame:
        """
        Age: 나이
        - 원래는 ratio 척도지만, 나이를 구간으로 나눈 ordinal 피처를 생성
        - bins: [-1, 0, 5, 12, 18, 24, 35, 60, inf]
          구간 의미:
          0: 미상/유아 (0-5세)
          1: 어린이 (6-12세)
          2: 청소년 (13-18세)
          3: 청년 (19-24세)
          4: 성인 (25-35세)
          5: 중년 (36-60세)
          6: 노년 (60세 이상)
        """
        df = df.copy()
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        
        # 결측치를 중앙값으로 채우기
        if df["Age"].isnull().any():
            median_age = df["Age"].median()
            df["Age"].fillna(median_age, inplace=True)
            ic(f"Age 결측치 {df['Age'].isnull().sum()}개를 중앙값 {median_age}로 채웠습니다")
        
        # 나이를 구간화하여 ordinal 피처 생성
        df["Age"] = pd.cut(
            df["Age"],
            bins=bins,
            labels=[0, 1, 2, 3, 4, 5, 6, 7],
            include_lowest=True
        ).astype(int)
        
        # 원본 Age 컬럼은 유지
        return df
    
    def title_nominal(self, df: DataFrame) -> pd.DataFrame:
        """
        Title: 명칭 (Mr, Mrs, Miss, Master, Dr, etc.)
        - Name 컬럼에서 추출한 타이틀
        - nominal 척도이므로 one-hot encoding 또는 LabelEncoding 사용
        """
        df = df.copy()
        
        # Name 컬럼에서 Title 추출 (정규표현식 사용)
        # 예: "Braund, Mr. Owen Harris" -> "Mr"
        df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.', expand=False)
        
        # 희소한 타이틀을 "Rare" 그룹으로 묶기
        # 일반적인 타이틀: Mr, Mrs, Miss, Master
        common_titles = ["Mr", "Mrs", "Miss", "Master"]
        df["Title"] = df["Title"].apply(
            lambda x: x if x in common_titles else "Rare"
        )
        
        # 결측치 처리 (혹시 모를 경우를 대비)
        if df["Title"].isnull().any():
            df["Title"].fillna("Mr", inplace=True)  # 가장 많은 타이틀로 채우기
        
        # One-hot encoding
        title_dummies = pd.get_dummies(df["Title"], prefix="Title")
        df = pd.concat([df, title_dummies], axis=1)
        
        # 원본 Title 컬럼은 유지 (필요시 drop_feature로 제거 가능)
        return df

