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
from app.titanic.titanic_method import TitanicMethod
from app.titanic.titanic_dataset import TitanicDataSet

# 공통 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 로깅 설정
try:
    from common.utils import setup_logging
    logger = setup_logging("titanic_service")
except ImportError:
    import logging
    logger = logging.getLogger("titanic_service")


class TitanicService:
    """타이타닉 데이터 처리 및 머신러닝 서비스"""
    
    def __init__(self):
        # CSV 파일 경로 설정
        # titanic_service.py 위치: app/titanic/titanic_service.py
        # CSV 파일 위치: app/resources/titanic/
        current_file = Path(__file__).resolve()
        # app/titanic/titanic_service.py -> app/ -> app/resources/titanic/
        app_dir = current_file.parent.parent  # app/
        resources_dir = app_dir / "resources" / "titanic"
        
        self.train_csv_path = resources_dir / "train.csv"
        self.test_csv_path = resources_dir / "test.csv"
        
        # 경로 검증
        if not self.train_csv_path.exists():
            logger.warning(f"Train CSV 파일을 찾을 수 없습니다: {self.train_csv_path}")
        if not self.test_csv_path.exists():
            logger.warning(f"Test CSV 파일을 찾을 수 없습니다: {self.test_csv_path}")
    
    def _get_csv_path(self, filename: str) -> Path:
        """
        CSV 파일의 전체 경로를 반환
        Args:
            filename: CSV 파일명 (train.csv 또는 test.csv)
        Returns:
            CSV 파일의 Path 객체
        """
        if filename == "train.csv":
            return self.train_csv_path
        elif filename == "test.csv":
            return self.test_csv_path
        else:
            # 기본적으로 resources/titanic 폴더에서 찾기
            current_file = Path(__file__).resolve()
            app_dir = current_file.parent.parent  # app/
            resources_dir = app_dir / "resources" / "titanic"
            return resources_dir / filename

    def preprocess(self) -> Dict[str, Any]:
        """
        타이타닉 데이터 전처리 실행
        Returns:
            전처리 결과 정보 딕셔너리
        """
        logger.info("=" * 80)
        logger.info("전처리 시작")
        logger.info("=" * 80)
        
        the_method = TitanicMethod()

        train_csv_path = self._get_csv_path('train.csv')
        logger.info(f"Train CSV 파일 경로: {train_csv_path}")
        logger.info(f"Train CSV 파일 존재 여부: {train_csv_path.exists()}")
        
        df_train = the_method.read_csv(str(train_csv_path))
        this_train = the_method.create_df(df_train, 'Survived')
        
        logger.info("-" * 80)
        logger.info("[Train 데이터셋 정보]")
        logger.info(f"  타입: {type(this_train).__name__}")
        logger.info(f"  컬럼 수: {len(this_train.columns)}")
        logger.info(f"  컬럼 목록: {', '.join(this_train.columns.tolist())}")
        logger.info(f"  행 수: {len(this_train)}")
        logger.info(f"  Null 값 개수: {the_method.check_null(this_train)}개")
        logger.info("-" * 80)
        logger.info("[Train 데이터 상위 5개 행]")
        logger.info(f"\n{this_train.head(5).to_string()}\n")

        test_csv_path = self._get_csv_path('test.csv')
        logger.info(f"Test CSV 파일 경로: {test_csv_path}")
        logger.info(f"Test CSV 파일 존재 여부: {test_csv_path.exists()}")
        
        df_test = the_method.read_csv(str(test_csv_path))
        this_test = the_method.create_df(df_test, 'Survived')
        
        logger.info("-" * 80)
        logger.info("[Test 데이터셋 정보]")
        logger.info(f"  타입: {type(this_test).__name__}")
        logger.info(f"  컬럼 수: {len(this_test.columns)}")
        logger.info(f"  컬럼 목록: {', '.join(this_test.columns.tolist())}")
        logger.info(f"  행 수: {len(this_test)}")
        logger.info(f"  Null 값 개수: {the_method.check_null(this_test)}개")
        logger.info("-" * 80)
        logger.info("[Test 데이터 상위 5개 행]")
        logger.info(f"\n{this_test.head(5).to_string()}\n")
        
        this = TitanicDataSet()

        this.train = this_train
        this.test = this_test

        drop_features = ['SibSp', 'Parch', 'Cabin', 'Ticket']
        this = the_method.drop_feature(this, *drop_features)
        this = the_method.pclass_ordinal(this)
        this = the_method.fare_ordinal(this)
        this = the_method.embarked_nominal(this)
        this = the_method.gender_nominal(this)
        this = the_method.extract_title_from_name(this)  # Name에서 Title 추출
        this = the_method.age_ratio(this)
        this = the_method.title_nominal(this)
        drop_name = ['Name']
        this = the_method.drop_feature(this, *drop_name)

        logger.info("=" * 80)
        logger.info("[Train 전처리 완료]")
        logger.info("-" * 80)
        logger.info(f"  타입: {type(this.train).__name__}")
        logger.info(f"  컬럼 수: {len(this.train.columns)}")
        logger.info(f"  컬럼 목록: {', '.join(this.train.columns.tolist())}")
        logger.info(f"  행 수: {len(this.train)}")
        logger.info(f"  Null 값 개수: {the_method.check_null(this.train)}개")
        logger.info("-" * 80)
        logger.info("[Train 전처리 후 상위 5개 행]")
        logger.info(f"\n{this.train.head(5).to_string()}\n")

        logger.info("=" * 80)
        logger.info("[Test 전처리 완료]")
        logger.info("-" * 80)
        logger.info(f"  타입: {type(this.test).__name__}")
        logger.info(f"  컬럼 수: {len(this.test.columns)}")
        logger.info(f"  컬럼 목록: {', '.join(this.test.columns.tolist())}")
        logger.info(f"  행 수: {len(this.test)}")
        logger.info(f"  Null 값 개수: {the_method.check_null(this.test)}개")
        logger.info("-" * 80)
        logger.info("[Test 전처리 후 상위 5개 행]")
        logger.info(f"\n{this.test.head(5).to_string()}\n")
        
        # 전처리 결과 정보 반환
        return {
            "status": "success",
            "rows": len(this_train),
            "columns": this_train.columns.tolist(),
            "column_count": len(this_train.columns),
            "null_count": int(the_method.check_null(this_train)),
            "sample_data": this_train.head(5).to_dict(orient="records"),
            "dtypes": this_train.dtypes.astype(str).to_dict()
        }

    def modeling(self):
        logger.info("모델링 시작")
        logger.info("모델링 완료")

    def learning(self):
        logger.info("학습 시작")
        logger.info("학습 완료")

    def evaluate(self):
        logger.info("평가 시작")
        logger.info("평가 완료")

    def submit(self):
        logger.info("제출 시작")
        logger.info("제출 완료")