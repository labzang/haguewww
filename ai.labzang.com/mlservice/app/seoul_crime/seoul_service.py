import sys
from pathlib import Path
import pandas as pd
import numpy as np
from app.seoul_crime.seoul_method import SeoulMethod
from app.seoul_crime.seoul_data import SeoulData
try:
    from common.utils import setup_logging
    logger = setup_logging("seoul_service")
except ImportError:
    import logging
    logger = logging.getLogger("seoul_service")
class SeoulService:
    
    def __init__(self):
        self.data = SeoulData()
        self.method = SeoulMethod()
        self.crime_rate_columns = ['살인검거율', '강도검거율', '강간검거율', '절도검거율', '폭력검거율']
        self.crime_columns = ['살인', '강도', '강간', '절도', '폭력']

    def preprocess(self):
        data_dir = Path(self.data.dname)
        cctv_path = data_dir / "cctv.csv"
        crime_path = data_dir / "crime.csv"
        pop_path = data_dir / "pop.xls"
        
        # 데이터 로드
        cctv = self.method.csv_to_df(str(cctv_path))
        crime = self.method.csv_to_df(str(crime_path))
        pop = self.method.xlsx_to_df(str(pop_path))
        
        logger.info(f"  cctv 탑 5 : {cctv.head(5).to_string()}")
        logger.info(f"  crime 탑 5 : {crime.head(5).to_string()}")
        logger.info(f"  pop 탑 5 : {pop.head(5).to_string()}")
        
        # cctv와 pop 머지 전략
        # - cctv의 "기관명"과 pop의 "자치구"를 키로 사용
        # - 중복된 feature가 없도록 처리
        # - "기관명"과 "자치구"는 같은 값이지만 컬럼명이 다르므로 left_on, right_on 사용
        
        # 머지 전에 컬럼명 확인 및 중복 컬럼 체크
        logger.info(f"cctv 컬럼: {cctv.columns.tolist()}")
        logger.info(f"pop 컬럼: {pop.columns.tolist()}")
        
        # 중복되는 컬럼 확인 (키 컬럼 제외)
        cctv_cols = set(cctv.columns) - {'기관명'}
        pop_cols = set(pop.columns) - {'자치구'}
        duplicate_cols = cctv_cols & pop_cols
        
        if duplicate_cols:
            logger.warning(f"중복되는 컬럼이 발견되었습니다: {duplicate_cols}")
            logger.info("머지 시 suffixes를 사용하여 중복 컬럼을 구분합니다.")
        
        # cctv의 "기관명"과 pop의 "자치구"를 키로 머지
        cctv_pop = self.method.df_merge(
            left=cctv,
            right=pop,
            left_on='기관명',
            right_on='자치구',
            how='inner'
        )
        
        # 머지 후 "자치구" 컬럼 제거 (기관명과 동일한 값이므로)
        if '자치구' in cctv_pop.columns and '기관명' in cctv_pop.columns:
            # 두 컬럼의 값이 동일한지 확인
            if cctv_pop['기관명'].equals(cctv_pop['자치구']):
                cctv_pop = cctv_pop.drop(columns=['자치구'])
                logger.info("'자치구' 컬럼을 제거했습니다 (기관명과 동일한 값).")
            else:
                logger.warning("'기관명'과 '자치구'의 값이 다릅니다. 두 컬럼 모두 유지합니다.")
        
        logger.info(f"머지 완료: cctv_pop shape = {cctv_pop.shape}")
        logger.info(f"cctv_pop 컬럼: {cctv_pop.columns.tolist()}")
        logger.info(f"cctv_pop 탑 5:\n{cctv_pop.head(5).to_string()}")

        # 구별 고령자 비율과 CCTV 의 상관계수
        # 구별 외국인 비율과 CCTV 의 상관계수


        
        return {
            "status": "success",
            "cctv_rows": len(cctv),
            "crime_rows": len(crime),
            "pop_rows": len(pop),
            "message": "데이터 전처리가 완료되었습니다"
        }
        