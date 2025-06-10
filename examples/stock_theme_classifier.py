import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm # 진행률 표시를 위한 라이브러리 (설치 필요: pip install tqdm)

# 실제 Creon API 클라이언트를 임포트합니다.
# 이 모듈이 올바른 경로에 있는지 확인해야 합니다.
from api.creon_api import CreonAPIClient



# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Creon API 클라이언트 래퍼 클래스 ---
# 기존 MockCreonAPIClient를 실제 CreonAPIClient를 감싸는 래퍼로 변경합니다.
class CreonClientWrapper:
    def __init__(self):
        logging.info("Initializing Creon API Client.")
        self.creon_api = CreonAPIClient()
        self.connected = self.creon_api.connected # CreonAPIClient는 'connected' 속성을 가질 것으로 가정
        if not self.connected:
            logging.error("Failed to connect to Creon API. Please check Creon Plus setup.")
        else:
            logging.info("Successfully connected to Creon API.")

        # 종목 이름-코드 매핑 (실제 API에서 가져올 수 있다면 이 부분은 제거)
        # 현재는 예시를 위해 고정된 매핑을 사용합니다.
        self.stock_name_to_code = {
            '삼성전자': 'A005930', 'SK하이닉스': 'A000660', 'DB하이텍': 'A000990',
            '네패스아크': 'A330080', '와이아이케이': 'A232140',
            'LG에너지솔루션': 'A373220', '삼성SDI': 'A006400', 'SK이노베이션': 'A096770',
            '에코프로비엠': 'A247540', '포스코퓨처엠': 'A003670', 'LG화학': 'A051910',
            '일진머티리얼즈': 'A020150', '엘앤에프': 'A066970',
            '삼성바이오로직스': 'A207940', '셀트리온': 'A068270', 'SK바이오사이언스': 'A302440',
            '유한양행': 'A000100', '한미약품': 'A128940',
            'NAVER': 'A035420', '카카오': 'A035720', '크래프톤': 'A259960',
            '엔씨소프트': 'A036570', '넷마블': 'A251270',
            '현대차': 'A005380', '기아': 'A000270', '현대모비스': 'A012330',
            '만도': 'A204320', '한온시스템': 'A018880',
            'POSCO홀딩스': 'A005490', '고려아연': 'A010130', '롯데케미칼': 'A011170',
            '금호석유': 'A011780', '효성첨단소재': 'A298050',
            'KB금융': 'A105560', '신한지주': 'A055550', '하나금융지주': 'A086790',
            '우리금융지주': 'A316140', '메리츠금융지주': 'A138040',
            'SK텔레콤': 'A017670', 'KT': 'A030200', 'LG유플러스': 'A032640',
            'SK스퀘어': 'A402340',
            'CJ제일제당': 'A097950', '오리온': 'A271560', '롯데쇼핑': 'A023530',
            '이마트': 'A139480', 'BGF리테일': 'A282330',
            '현대건설': 'A000720', '대우건설': 'A047040', 'GS건설': 'A006360',
            '두산에너빌리티': 'A034020', '두산밥캣': 'A241560',
            '한국조선해양': 'A009540', '삼성중공업': 'A010140', '대한항공': 'A003490',
            '현대미포조선': 'A060980',
            '한국전력': 'A015760', '한국가스공사': 'A036460', '두산퓨얼셀': 'A336260',
            '에스디바이오센서': 'A137150',
            '원익IPS': 'A240810', '피에스케이': 'A031980', '주성엔지니어링': 'A036930',
            '테스': 'A095610', '에이피티씨': 'A089150',
            'LG디스플레이': 'A034220', '덕산네오룩스': 'A213420', '동운아나텍': 'A094610',
            '매크로젠': 'A038290',
            '한화에어로스페이스': 'A012450', 'LIG넥스원': 'A079550', '한화시스템': 'A272210',
            '현대로템': 'A064350',
            'KODEX 국고채30년액티브': 'A439870'
        }

    def get_stock_code(self, stock_name):
        # 실제 API에서 종목 코드를 가져오는 로직이 있다면 이곳을 수정해야 합니다.
        # 현재는 미리 정의된 매핑을 사용합니다.
        return self.stock_name_to_code.get(stock_name)

    def get_daily_ohlcv(self, stock_code, start_date_str, end_date_str, count=0):
        if not self.connected:
            logging.error("Creon API is not connected. Cannot fetch daily OHLCV data.")
            return pd.DataFrame() # 연결되지 않으면 빈 DataFrame 반환

        # 실제 CreonAPIClient의 get_daily_ohlcv 메서드를 호출합니다.
        # CreonAPIClient가 반환하는 DataFrame의 컬럼명과 인덱스 형식이 적절한지 확인해야 합니다.
        df = self.creon_api.get_daily_ohlcv(stock_code, start_date_str, end_date_str)
        
        if df is None: # CreonAPIClient가 실패 시 None을 반환할 수 있음
            logging.warning(f"No data returned from Creon API for {stock_code} from {start_date_str} to {end_date_str}.")
            return pd.DataFrame()
        
        # DataFrame의 인덱스가 datetime 형식이고 이름이 'date'인지 확인하고 설정합니다.
        if not isinstance(df.index, pd.DatetimeIndex):
            # 만약 인덱스가 datetime이 아니라면, datetime으로 변환을 시도합니다.
            # Creon API가 날짜 컬럼을 제공하고 있다면 해당 컬럼을 인덱스로 설정해야 합니다.
            # 여기서는 반환된 DataFrame의 인덱스가 이미 날짜/시간 정보를 포함하고 있다고 가정합니다.
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logging.error(f"Error converting index to datetime for {stock_code}: {e}")
                return pd.DataFrame() # 변환 실패 시 빈 DataFrame 반환

        df.index.name = 'date'
        
        # 필요한 'close' 컬럼이 있는지 확인 (Creon API에서 다른 컬럼명을 사용한다면 수정 필요)
        if 'close' not in df.columns:
            logging.error(f"DataFrame for {stock_code} does not contain a 'close' column. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()

        return df

# --- 특징 추출 함수 ---
def extract_price_features(daily_df, lookback_period=20): # 20거래일 = 약 1개월
    if daily_df.empty or len(daily_df) < lookback_period + 1:
        return None

    # 직전 lookback_period+1 일 데이터 사용 (pct_change를 위해)
    # 데이터프레임이 날짜 기준으로 정렬되어 있어야 합니다.
    recent_data = daily_df['close'].tail(lookback_period + 1)
    
    # 모멘텀 (lookback_period일 수익률)
    # 데이터가 충분한지 다시 확인
    if len(recent_data) < lookback_period + 1 or recent_data.iloc[0] == 0:
        return None # 데이터 부족하거나 첫 가격이 0이면 계산 불가

    momentum = (recent_data.iloc[-1] / recent_data.iloc[0] - 1) * 100
    
    # 변동성 (lookback_period일 일일 수익률의 표준편차)
    daily_returns = recent_data.pct_change().dropna()
    if daily_returns.empty:
        return None # 일일 수익률 계산 불가 시
        
    volatility = daily_returns.std() * np.sqrt(252) * 100 # 연율화된 변동성 (백분율)

    return pd.Series({
        'momentum_20d': momentum,
        'volatility_20d': volatility
    })

# --- 메인 시뮬레이션 로직 ---
if __name__ == '__main__':
    logging.info("Starting stock theme classification simulation using Random Forest.")

    # 1. 미리 정의된 테마 및 종목 리스트
    sector_stocks = {
        '반도체': [
            ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT'),
            ('네패스아크', 'IT'), ('와이아이케이', 'IT')
        ],
        '2차전지': [
            ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지'),
            ('에코프로비엠', '2차전지'), ('포스코퓨처엠', '2차전지'), ('LG화학', '2차전지'),
            ('일진머티리얼즈', '2차전지'), ('엘앤에프', '2차전지')
        ],
        '바이오': [
            ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오'),
            ('유한양행', '바이오'), ('한미약품', '바이오')
        ],
        '플랫폼/인터넷': [
            ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT'),
            ('엔씨소프트', 'IT'), ('넷마블', 'IT')
        ],
        '자동차': [
            ('현대차', '자동차'), ('기아', '자동차'), ('현대모비스', '자동차'),
            ('만도', '자동차'), ('한온시스템', '자동차')
        ],
        '철강/화학': [
            ('POSCO홀딩스', '철강'), ('고려아연', '철강'), ('롯데케미칼', '화학'),
            ('금호석유', '화학'), ('효성첨단소재', '화학')
        ],
        '금융': [
            ('KB금융', '금융'), ('신한지주', '금융'), ('하나금융지주', '금융'),
            ('우리금융지주', '금융'), ('메리츠금융지주', '금융')
        ],
        '통신': [
            ('SK텔레콤', '통신'), ('KT', '통신'), ('LG유플러스', '통신'),
            ('SK스퀘어', '통신')
        ],
        '유통/소비재': [
            ('CJ제일제당', '소비재'), ('오리온', '소비재'), ('롯데쇼핑', '유통'),
            ('이마트', '유통'), ('BGF리테일', '유통')
        ],
        '건설/기계': [
            ('현대건설', '건설'), ('대우건설', '건설'), ('GS건설', '건설'),
            ('두산에너빌리티', '기계'), ('두산밥캣', '기계')
        ],
        '조선/항공': [
            ('한국조선해양', '조선'), ('삼성중공업', '조선'), ('대한항공', '항공'),
            ('현대미포조선', '조선')
        ],
        '에너지': [
            ('한국전력', '에너지'), ('한국가스공사', '에너지'), ('두산퓨얼셀', '에너지'),
            ('에스디바이오센서', '에너지') # 실제는 바이오/진단 키트지만, 예시 테마 분류에 따라 포함
        ],
        '반도체장비': [
            ('원익IPS', 'IT'), ('피에스케이', 'IT'), ('주성엔지니어링', 'IT'),
            ('테스', 'IT'), ('에이피티씨', 'IT')
        ],
        '디스플레이': [
            ('LG디스플레이', 'IT'), ('덕산네오룩스', 'IT'), ('동운아나텍', 'IT'),
            ('매크로젠', 'IT') # 실제는 바이오지만, 예시 테마 분류에 따라 포함
        ],
        '방산': [
            ('한화에어로스페이스', '방산'), ('LIG넥스원', '방산'), ('한화시스템', '방산'),
            ('현대로템', '방산')
        ]
    }

    # CreonClientWrapper를 사용하여 실제 Creon API 클라이언트를 초기화합니다.
    creon_api_client = CreonClientWrapper()

    # API 연결 실패 시 프로그램 종료
    if not creon_api_client.connected:
        logging.error("Failed to connect to Creon API. Exiting program.")
        exit()

    all_stock_codes = []
    stock_code_to_theme = {}
    for theme, stocks in sector_stocks.items():
        for name, sub_theme in stocks:
            code = creon_api_client.get_stock_code(name)
            if code:
                all_stock_codes.append(code)
                stock_code_to_theme[code] = theme
            else:
                logging.warning(f"Could not find stock code for '{name}'. Skipping.")

    if not all_stock_codes:
        logging.error("No valid stock codes found for simulation. Exiting.")
        exit()

    # 2. 학습 데이터 생성
    logging.info("Generating training data...")
    # 학습 기간 설정 (예: 2023년 1월 1일 ~ 2024년 12월 31일)
    train_start_date = datetime.date(2023, 1, 1)
    train_end_date = datetime.date(2024, 12, 31)

    X_train_data = [] # 특징 데이터
    y_train_labels = [] # 테마 레이블

    # 매월 첫 영업일을 기준으로 특징 추출
    current_date = train_start_date
    while current_date <= train_end_date:
        # 매월 첫 영업일 찾기 (간단화: 실제는 pandas business date range 등으로 처리)
        # 매월 1일이 주말이면 다음 월요일로 이동
        if current_date.day == 1:
            while current_date.weekday() >= 5: # 5: 토요일, 6: 일요일
                current_date += datetime.timedelta(days=1)
            
            logging.debug(f"Processing features for training on {current_date.isoformat()}")

            for stock_code in tqdm(all_stock_codes, desc=f"Extracting features for {current_date.isoformat()}"):
                # 특징 추출에 필요한 기간 (현재 날짜 기준 이전 1개월) 데이터 요청
                # 약 20거래일 전부터 현재 날짜까지의 데이터 필요 (모멘텀, 변동성 계산)
                fetch_start_date = current_date - datetime.timedelta(days=35) # 넉넉하게 35일 전부터
                
                # 실제 Creon API를 통해 일봉 데이터 가져오기
                daily_df_history = creon_api_client.get_daily_ohlcv(
                    stock_code,
                    fetch_start_date.strftime('%Y%m%d'),
                    current_date.strftime('%Y%m%d')
                )

                features = extract_price_features(daily_df_history, lookback_period=20) # 20거래일 기준 특징
                
                if features is not None:
                    X_train_data.append(features.tolist())
                    y_train_labels.append(stock_code_to_theme[stock_code])
                else:
                    logging.warning(f"Could not extract features for {stock_code} on {current_date.isoformat()}. Skipping for training.")
        
        current_date += datetime.timedelta(days=1)

    if not X_train_data:
        logging.error("No training data generated. Please check your date ranges and Creon API connection/data.")
        exit()

    X = pd.DataFrame(X_train_data, columns=['momentum_20d', 'volatility_20d'])
    y = pd.Series(y_train_labels)

    # 3. 랜덤 포레스트 모델 학습
    logging.info(f"Training Random Forest model with {len(X)} samples...")
    # 학습/테스트 세트 분리 (선택 사항이지만 모델 성능 검증에 필수)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # 클래스 불균형 고려
    model.fit(X_train, y_train)

    # 모델 성능 평가
    y_pred = model.predict(X_test)
    logging.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # 4. 새로운 종목 분류 시뮬레이션 (200개 종목)
    logging.info("Simulating classification of new stocks...")
    
    # 분류할 종목 목록 (예시로 기존 종목 중 일부와 가상의 새 종목 추가)
    # 실제로는 '기본적 분석으로 필터링한 200개 종목'이 여기에 들어갑니다.
    # 여기서는 기존 테마 종목의 일부를 재활용하여 모의 분류를 진행합니다.
    stocks_to_classify_codes = list(np.random.choice(all_stock_codes, size=min(20, len(all_stock_codes)), replace=False))
    
    # 가상의 새로운 종목 추가 (이름과 코드를 임의로 부여)
    # 실제 Creon API를 사용할 경우, 이 부분은 실제 시장에 존재하는 종목 코드로 대체되어야 합니다.
    # CreonAPIClient의 get_stock_code가 해당 가상 종목을 찾지 못할 수 있습니다.
    # 따라서 이 부분은 실제 시뮬레이션에서 사용하려면 해당 종목의 크레온 코드와 이름이 매핑되어 있어야 합니다.
    for i in range(5):
        new_stock_name = f"가상기업{i+1}"
        new_stock_code = f"A999{i:03d}" # 예시 가상 코드
        # creon_api_client.stock_name_to_code[new_stock_name] = new_stock_code # 실제 사용 시 이 매핑은 API에서 가져오거나 별도 관리
        stocks_to_classify_codes.append(new_stock_code)


    classification_date = datetime.date(2025, 5, 20) # 가장 최신 분류 날짜
    classification_data = []
    classified_stock_info = []

    for stock_code_to_classify in tqdm(stocks_to_classify_codes, desc=f"Extracting features for classification on {classification_date.isoformat()}"):
        fetch_start_date = classification_date - datetime.timedelta(days=35) # 넉넉하게 35일 전부터
        
        # 실제 Creon API를 통해 일봉 데이터 가져오기
        daily_df_history = creon_api_client.get_daily_ohlcv(
            stock_code_to_classify,
            fetch_start_date.strftime('%Y%m%d'),
            classification_date.strftime('%Y%m%d')
        )
        
        features = extract_price_features(daily_df_history, lookback_period=20)
        
        if features is not None:
            classification_data.append(features.tolist())
            
            # 종목 이름을 다시 찾기 (CreonClientWrapper에서)
            stock_name_found = next((name for name, code in creon_api_client.stock_name_to_code.items() if code == stock_code_to_classify), stock_code_to_classify)
            classified_stock_info.append((stock_name_found, stock_code_to_classify))
        else:
            logging.warning(f"Could not extract features for classification for {stock_code_to_classify} on {classification_date.isoformat()}. Skipping.")

    if not classification_data:
        logging.warning("No classification data generated. Check stock codes and dates.")
    else:
        X_classify = pd.DataFrame(classification_data, columns=['momentum_20d', 'volatility_20d'])
        predicted_themes = model.predict(X_classify)
        
        logging.info("\n=== Classification Results for New Stocks ===")
        for i, (stock_name, stock_code) in enumerate(classified_stock_info):
            logging.info(f"Stock: {stock_name} ({stock_code}) -> Predicted Theme: {predicted_themes[i]}")

        # 특징 중요도 시각화 (모델의 학습 결과를 해석)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            logging.info("\n=== Feature Importances ===")
            logging.info(feature_importance)
