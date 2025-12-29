"""
위험 판매자 예측 데이터 자동 생성 스크립트

kys_Data_For_Mailing.ipynb의 머신러닝 파이프라인을 스크립트로 변환
대시보드에서 버튼 클릭으로 실행 가능
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import os

warnings.filterwarnings('ignore')


def generate_risk_report(
    input_path='kys/Olist_DataSet/merged_olist.csv',
    output_path='kys/Result/risk_report_result.csv'
):
    """
    위험 판매자 예측 데이터 생성
    
    Args:
        input_path: 입력 데이터 경로
        output_path: 출력 CSV 경로
        
    Returns:
        dict: {
            'success': bool,
            'total_risk_sellers': int,
            'red_zone': int,
            'yellow_zone': int,
            'message': str,
            'duration_seconds': float
        }
    """
    start_time = datetime.now()
    
    try:
        # ===== 1. 데이터 로드 및 전처리 =====
        if not os.path.exists(input_path):
            return {
                'success': False,
                'message': f'입력 파일을 찾을 수 없습니다: {input_path}',
                'total_risk_sellers': 0,
                'red_zone': 0,
                'yellow_zone': 0,
                'duration_seconds': 0
            }
        
        df = pd.read_csv(input_path)
        
        # 필요한 컬럼만 선택
        cols_needed = [
            'order_id', 'seller_id', 'order_purchase_timestamp',
            'review_score', 'has_text_review',
            'seller_delay_days', 'seller_processing_days',
            'processing_days_diff',
            'is_logistics_fault'
        ]
        
        df = df[cols_needed].copy()
        
        # 날짜 변환
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
        
        # 물류사 과실 제외
        df = df[df['is_logistics_fault'] == False].copy()
        
        # ===== 2. 파생 변수 생성 =====
        df['is_processing_delayed'] = (df['processing_days_diff'] > 0).astype(int)
        df['is_seller_delayed'] = (df['seller_delay_days'] > 0).astype(int)
        df['is_negative_review'] = (df['review_score'] <= 3).astype(int)
        df['is_critical_complaint'] = ((df['review_score'] <= 3) & (df['has_text_review'] == True)).astype(int)
        
        # ===== 3. 월별 판매자 통계 생성 =====
        monthly_stats = df.groupby(['seller_id', 'year_month']).agg({
            'is_processing_delayed': 'mean',
            'is_seller_delayed': 'mean',
            'is_negative_review': 'mean',
            'is_critical_complaint': 'mean',
            'seller_processing_days': ['mean', 'std'],
            'order_id': 'count'
        }).reset_index()
        
        monthly_stats.columns = [
            'seller_id', 'year_month',
            'processing_delay_rate',
            'seller_delay_rate',
            'negative_review_rate',
            'critical_complaint_rate',
            'processing_days_mean', 'processing_days_std',
            'order_count'
        ]
        
        monthly_stats['processing_days_std'] = monthly_stats['processing_days_std'].fillna(0)
        
        # ===== 4. 점수 계산 및 Target 생성 =====
        def linear_score(rate_value, min_val=0.0, max_val=1.0):
            clipped = np.clip(rate_value, min_val, max_val)
            score = (clipped - min_val) / (max_val - min_val) * 100
            return score
        
        def calculate_weighted_score_linear(processing_rate, delay_rate, negative_rate):
            score_processing = linear_score(processing_rate)
            score_delay = linear_score(delay_rate)
            score_negative = linear_score(negative_rate)
            
            weighted_score = (
                score_processing * 0.15 +
                score_delay * 0.50 +
                score_negative * 0.35
            )
            return weighted_score
        
        monthly_stats['weighted_score_linear_75'] = calculate_weighted_score_linear(
            monthly_stats['processing_delay_rate'],
            monthly_stats['seller_delay_rate'],
            monthly_stats['negative_review_rate']
        )
        
        monthly_stats['is_seller_of_note_linear_75'] = monthly_stats['weighted_score_linear_75'] >= 50
        
        # ===== 5. 피처 엔지니어링 =====
        monthly_stats_enhanced = monthly_stats.sort_values(['seller_id', 'year_month']).copy()
        
        # 이동 평균 피처
        cols_for_rolling = ['processing_delay_rate', 'seller_delay_rate', 'negative_review_rate']
        for col in cols_for_rolling:
            monthly_stats_enhanced[f'{col}_rolling_2'] = monthly_stats_enhanced.groupby('seller_id')[col].rolling(2, min_periods=1).mean().values
            monthly_stats_enhanced[f'{col}_rolling_3'] = monthly_stats_enhanced.groupby('seller_id')[col].rolling(3, min_periods=1).mean().values
        
        # 변화율 피처
        for col in ['processing_delay_rate', 'seller_delay_rate', 'negative_review_rate', 'order_count']:
            prev = monthly_stats_enhanced.groupby('seller_id')[col].shift(1)
            monthly_stats_enhanced[f'{col}_change'] = (monthly_stats_enhanced[col] - prev).fillna(0)
        
        # 상호작용 피처
        monthly_stats_enhanced['delay_negative_interaction'] = monthly_stats_enhanced['seller_delay_rate'] * monthly_stats_enhanced['negative_review_rate']
        monthly_stats_enhanced['processing_seller_delay_interaction'] = monthly_stats_enhanced['processing_delay_rate'] * monthly_stats_enhanced['seller_delay_rate']
        monthly_stats_enhanced['total_risk_score'] = monthly_stats_enhanced['processing_delay_rate'] + monthly_stats_enhanced['seller_delay_rate'] + monthly_stats_enhanced['negative_review_rate']
        monthly_stats_enhanced['avg_delay_rate'] = (monthly_stats_enhanced['processing_delay_rate'] + monthly_stats_enhanced['seller_delay_rate']) / 2
        
        # 시간 피처
        monthly_stats_enhanced['year_month_str'] = monthly_stats_enhanced['year_month'].astype(str) + '-01'
        monthly_stats_enhanced['year_month_dt'] = pd.to_datetime(monthly_stats_enhanced['year_month_str'])
        monthly_stats_enhanced['month'] = monthly_stats_enhanced['year_month_dt'].dt.month
        monthly_stats_enhanced['quarter'] = monthly_stats_enhanced['year_month_dt'].dt.quarter
        monthly_stats_enhanced['is_month_start'] = (monthly_stats_enhanced['month'] <= 6).astype(int)
        monthly_stats_enhanced['seller_tenure_months'] = monthly_stats_enhanced.groupby('seller_id').cumcount() + 1
        
        # ===== 6. Target 및 Lag 피처 생성 =====
        monthly_stats_enhanced['target_is_seller_of_note_linear_75'] = monthly_stats_enhanced.groupby('seller_id')['is_seller_of_note_linear_75'].shift(-1)
        
        # Lag 피처
        enhanced_feature_cols = [
            'processing_delay_rate', 'seller_delay_rate', 'negative_review_rate',
            'critical_complaint_rate', 'processing_days_mean', 'processing_days_std', 'order_count',
            'processing_delay_rate_rolling_2', 'processing_delay_rate_rolling_3',
            'seller_delay_rate_rolling_2', 'seller_delay_rate_rolling_3',
            'negative_review_rate_rolling_2', 'negative_review_rate_rolling_3',
            'processing_delay_rate_change', 'seller_delay_rate_change',
            'negative_review_rate_change', 'order_count_change',
            'delay_negative_interaction', 'processing_seller_delay_interaction',
            'total_risk_score', 'avg_delay_rate',
            'month', 'quarter', 'is_month_start', 'seller_tenure_months'
        ]
        
        for col in enhanced_feature_cols:
            if col not in ['month', 'quarter', 'is_month_start', 'seller_tenure_months']:
                monthly_stats_enhanced[f'prev_{col}'] = monthly_stats_enhanced.groupby('seller_id')[col].shift(1)
            else:
                monthly_stats_enhanced[f'prev_{col}'] = monthly_stats_enhanced[col]
        
        # ===== 7. 최종 데이터셋 생성 =====
        final_enhanced_feature_cols = [f'prev_{col}' for col in enhanced_feature_cols]
        final_cols_enhanced = ['seller_id', 'year_month'] + final_enhanced_feature_cols + ['target_is_seller_of_note_linear_75']
        
        df_final_enhanced = monthly_stats_enhanced[final_cols_enhanced].dropna().copy()
        df_final_enhanced['target_is_seller_of_note_linear_75'] = df_final_enhanced['target_is_seller_of_note_linear_75'].astype(int)
        
        # ===== 8. Train/Test 분할 =====
        X_enhanced = df_final_enhanced[final_enhanced_feature_cols]
        y_enhanced = df_final_enhanced['target_is_seller_of_note_linear_75']
        
        split_idx = int(len(X_enhanced) * 0.8)
        X_train_enh = X_enhanced.iloc[:split_idx]
        X_test_enh = X_enhanced.iloc[split_idx:]
        y_train_enh = y_enhanced.iloc[:split_idx]
        y_test_enh = y_enhanced.iloc[split_idx:]
        
        # ===== 9. RandomForest 모델 학습 =====
        model_rf_enh = RandomForestClassifier(
            class_weight='balanced',
            max_depth=10,
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        
        model_rf_enh.fit(X_train_enh, y_train_enh)
        
        # ===== 10. 예측 수행 =====
        y_pred_proba_rf = model_rf_enh.predict_proba(X_test_enh)[:, 1]
        y_pred_rf_025 = (y_pred_proba_rf >= 0.25).astype(int)
        
        # ===== 11. 메일링용 데이터프레임 생성 =====
        test_indices = y_test_enh.index
        prediction_df = df_final_enhanced.loc[test_indices, ['seller_id', 'year_month']].copy()
        prediction_df['predicted'] = y_pred_rf_025
        prediction_df['y_pred_proba'] = y_pred_proba_rf
        
        # predicted=1만 필터링
        risk_sellers = prediction_df[prediction_df['predicted'] == 1].copy()
        
        # 우선순위 분류
        def assign_priority(prob):
            if prob >= 0.8:
                return 'RED'
            else:
                return 'YELLOW'
        
        risk_sellers['priority'] = risk_sellers['y_pred_proba'].apply(assign_priority)
        risk_sellers = risk_sellers.sort_values('y_pred_proba', ascending=False)
        
        # ===== 12. 위험사유 매핑 =====
        risk_names = {
            'processing_delay': '처리지연율',
            'seller_delay': '출고지연율',
            'negative_review': '부정리뷰율',
            'processing_delay_trend': '처리지연율 추세',
            'seller_delay_trend': '출고지연율 추세'
        }
        
        risk_reasons = []
        for idx in risk_sellers.index:
            features = X_test_enh.loc[idx]
            
            processing_delay = features['prev_processing_delay_rate']
            seller_delay = features['prev_seller_delay_rate']
            negative_review = features['prev_negative_review_rate']
            processing_delay_change = features['prev_processing_delay_rate_change']
            seller_delay_change = features['prev_seller_delay_rate_change']
            
            reasons = []
            if processing_delay >= 0.5:
                reasons.append(f"{risk_names['processing_delay']} 높음({processing_delay*100:.0f}%)")
            if seller_delay >= 0.5:
                reasons.append(f"{risk_names['seller_delay']} 높음({seller_delay*100:.0f}%)")
            if negative_review >= 0.3:
                reasons.append(f"{risk_names['negative_review']} 높음({negative_review*100:.0f}%)")
            if processing_delay_change > 0.2:
                reasons.append(f"{risk_names['processing_delay_trend']} 높음({processing_delay_change*100:.0f}%)")
            if seller_delay_change > 0.2:
                reasons.append(f"{risk_names['seller_delay_trend']} 높음({seller_delay_change*100:.0f}%)")
            
            risk_reasons.append(" | ".join(reasons) if reasons else "지연/리뷰 문제")
        
        risk_sellers['주요_위험사유'] = risk_reasons
        
        # ===== 13. CSV 저장 및 DataFrame 반환 =====
        risk_sellers = risk_sellers[['seller_id', 'year_month', 'y_pred_proba', 'priority', '주요_위험사유']]
        
        # CSV 저장 시도 (Streamlit Cloud에서는 실패할 수 있음)
        try:
            risk_sellers.to_csv(output_path, index=False, encoding='utf-8-sig')
            csv_saved = True
        except (PermissionError, OSError) as e:
            # Streamlit Cloud 등 쓰기 권한이 없는 환경
            csv_saved = False
        
        # 실행 시간 계산
        duration = (datetime.now() - start_time).total_seconds()
        
        # 통계 집계
        red_zone_count = (risk_sellers['priority'] == 'RED').sum()
        yellow_zone_count = (risk_sellers['priority'] == 'YELLOW').sum()
        
        return {
            'success': True,
            'total_risk_sellers': len(risk_sellers),
            'red_zone': int(red_zone_count),
            'yellow_zone': int(yellow_zone_count),
            'message': f'성공적으로 생성되었습니다. (RED: {red_zone_count}명, YELLOW: {yellow_zone_count}명)',
            'duration_seconds': round(duration, 2),
            'dataframe': risk_sellers,  # DataFrame 추가
            'csv_saved': csv_saved  # CSV 저장 여부
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'오류 발생: {str(e)}',
            'total_risk_sellers': 0,
            'red_zone': 0,
            'yellow_zone': 0,
            'duration_seconds': (datetime.now() - start_time).total_seconds()
        }


if __name__ == '__main__':
    # 독립 실행 테스트
    print("=" * 80)
    print("위험 판매자 데이터 생성 시작")
    print("=" * 80)
    
    result = generate_risk_report()
    
    print(f"\n실행 결과:")
    print(f"  - 성공 여부: {result['success']}")
    print(f"  - 총 위험 판매자: {result['total_risk_sellers']}명")
    print(f"  - RED ZONE: {result['red_zone']}명")
    print(f"  - YELLOW ZONE: {result['yellow_zone']}명")
    print(f"  - 메시지: {result['message']}")
    print(f"  - 실행 시간: {result['duration_seconds']}초")

