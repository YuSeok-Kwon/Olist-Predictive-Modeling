import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
import sys
from pathlib import Path

# 같은 디렉토리의 스크립트 임포트
sys.path.append(str(Path(__file__).parent))
from generate_risk_data import generate_risk_report

# =========================================================
# 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="Olist 비즈니스 대시보드", 
    page_icon="📊", 
    layout="wide"
)

# =========================================================
# 데이터 로딩 함수
# =========================================================
@st.cache_data
def load_ml_data():
    """ML_olist.csv 로딩 및 기본 전처리"""
    try:
        df = pd.read_csv('kys/Olist_DataSet/ML_olist.csv')
        # 날짜 컬럼 변환
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
        df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
        return df
    except FileNotFoundError:
        st.error("ML_olist.csv 파일을 찾을 수 없습니다!")
        return pd.DataFrame()

def load_risk_data():
    """위험 판매자 예측 결과 로딩 (세션 스테이트 활용)"""
    # 세션 스테이트에 있으면 그것 사용
    if 'risk_data' in st.session_state and st.session_state.risk_data is not None:
        return st.session_state.risk_data
    
    # 없으면 파일에서 로드
    try:
        df = pd.read_csv('kys/Result/risk_report_result.csv')
        st.session_state.risk_data = df
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def aggregate_seller_data(df):
    """판매자별 집계 데이터 생성"""
    seller_agg = df.groupby('seller_id').agg({
        'order_id': 'count',
        'seller_processing_days': 'mean',
        'seller_delay_days': 'mean',
        'review_score': 'mean',
        'is_Seller_of_Note': 'first',
        'is_logistics_fault': lambda x: (x == True).sum() / len(x) * 100
    }).reset_index()
    
    seller_agg.columns = ['seller_id', 'order_count', 'avg_processing_days', 
                          'avg_delay_days', 'avg_review_score', 'is_seller_of_note',
                          'logistics_fault_rate']
    return seller_agg

# 데이터 로딩
df_ml = load_ml_data()
df_risk = load_risk_data()

# =========================================================
# 사이드바 - 필터 설정
# =========================================================
st.sidebar.header("🛠️ 필터 설정")

if not df_ml.empty:
    # 날짜 범위 필터
    min_date = df_ml['order_purchase_timestamp'].min().date()
    max_date = df_ml['order_purchase_timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "날짜 범위",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # 카테고리 필터
    categories = ['전체'] + sorted(df_ml['product_category_name_english'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("카테고리", categories)
    
    # 위험 판매자 Threshold (위험 모니터링용)
    st.sidebar.markdown("---")
    st.sidebar.subheader("위험 판매자 기준")
    risk_threshold = st.sidebar.slider("위험 확률 기준", 0.0, 1.0, 0.30, 0.05)
    
    # =========================================================
    # 데이터 갱신 섹션
    # =========================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 데이터 갱신")
    
    # 마지막 갱신 시간 표시
    risk_file_path = 'kys/Result/risk_report_result.csv'
    if os.path.exists(risk_file_path):
        last_modified = datetime.fromtimestamp(os.path.getmtime(risk_file_path))
        st.sidebar.info(f"마지막 갱신: {last_modified.strftime('%Y-%m-%d %H:%M')}")
        
        # 24시간 경과 여부 체크
        hours_passed = (datetime.now() - last_modified).total_seconds() / 3600
        if hours_passed > 24:
            st.sidebar.warning("⚠️ 데이터가 24시간 이상 경과했습니다")
        else:
            remaining_hours = 24 - hours_passed
            st.sidebar.success(f"✅ 데이터 최신 상태 (갱신까지 {remaining_hours:.1f}시간)")
    else:
        st.sidebar.warning("⚠️ 위험 판매자 데이터가 없습니다")
    
    # 갱신 버튼
    if st.sidebar.button("🔄 위험 판매자 데이터 갱신", type="primary", use_container_width=True):
        with st.spinner("데이터 생성 중... (약 30초 소요)"):
            # generate_risk_data.py 실행
            result = generate_risk_report()
            
            if result['success']:
                # 세션 스테이트에 DataFrame 저장 (Streamlit Cloud 대응)
                st.session_state.risk_data = result['dataframe']
                
                csv_status = "CSV 저장됨" if result.get('csv_saved', False) else "세션에만 저장됨"
                
                st.sidebar.success(
                    f"갱신 완료!\n"
                    f"- 총 위험 판매자: {result['total_risk_sellers']}명\n"
                    f"- RED ZONE: {result['red_zone']}명\n"
                    f"- YELLOW ZONE: {result['yellow_zone']}명\n"
                    f"- 소요 시간: {result['duration_seconds']}초\n"
                    f"- {csv_status}"
                )
                # 캐시 클리어 및 페이지 새로고침
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.error(f"❌ 갱신 실패\n{result['message']}")
    
    st.sidebar.caption("💡 일일 1회 갱신을 권장합니다")
    
    # 필터 적용
    if len(date_range) == 2:
        df_filtered = df_ml[
            (df_ml['order_purchase_timestamp'].dt.date >= date_range[0]) &
            (df_ml['order_purchase_timestamp'].dt.date <= date_range[1])
        ].copy()
    else:
        df_filtered = df_ml.copy()
    
    if selected_category != '전체':
        df_filtered = df_filtered[df_filtered['product_category_name_english'] == selected_category]
else:
    df_filtered = df_ml

# =========================================================
# 메인 헤더
# =========================================================
st.title("📊 Olist 비즈니스 통합 대시보드")
st.markdown("### 전체 비즈니스 현황에서 위험 요인까지 한눈에 확인")
st.markdown("---")

if df_ml.empty:
    st.error("데이터를 불러올 수 없습니다. ML_olist.csv 파일을 확인해주세요!")
    st.stop()

# =========================================================
# Section 1 & 2: 비즈니스 개요 + 판매자 성과 (상단 Z자)
# =========================================================
st.markdown("## 📈 전체 비즈니스 현황")

# =========================================================
# 상황판 요약 배너 - 4개 신호등 스타일
# =========================================================
# 목표 기준값 설정 (필요시 조정 가능)
TARGET_DELAY_RATE = 20.0  # 배송 지연율 목표 (%)
TARGET_NEGATIVE_RATE = 10.0  # 부정 리뷰율 목표 (%)
TARGET_SELLER_RISK_RATE = 10.0  # 유의 판매자 비율 목표 (%)
TARGET_LOGISTICS_FAULT = 50.0  # 물류 문제율 목표 (%)

# 지표 계산
delay_rate = (len(df_filtered[df_filtered['seller_delay_days'] > 0]) / len(df_filtered) * 100)
negative_rate = (len(df_filtered[df_filtered['review_score'] <= 2]) / len(df_filtered) * 100)
total_sellers = df_filtered['seller_id'].nunique()
risk_sellers = df_filtered[df_filtered['is_Seller_of_Note'] == True]['seller_id'].nunique()
risk_ratio = (risk_sellers / total_sellers * 100) if total_sellers > 0 else 0
logistics_fault_rate = (df_filtered['is_logistics_fault'].sum() / len(df_filtered) * 100)

banner_col1, banner_col2, banner_col3, banner_col4 = st.columns(4)

with banner_col1:
    if delay_rate > TARGET_DELAY_RATE * 2:
        st.error("🔴 배송 지연 심각")
    elif delay_rate > TARGET_DELAY_RATE * 1.25:
        st.warning("🟡 배송 지연 주의")
    else:
        st.success("🟢 배송 정상")

with banner_col2:
    if negative_rate > TARGET_NEGATIVE_RATE * 1.5:
        st.error("🔴 고객 불만 증가")
    elif negative_rate > TARGET_NEGATIVE_RATE:
        st.warning("🟡 리뷰 관리 필요")
    else:
        st.success("🟢 고객 만족 양호")

with banner_col3:
    if risk_ratio > TARGET_SELLER_RISK_RATE * 1.5:
        st.error("🔴 위험 판매자 多")
    elif risk_ratio > TARGET_SELLER_RISK_RATE:
        st.warning("🟡 판매자 관리 필요")
    else:
        st.success("🟢 판매자 품질 양호")

with banner_col4:
    if logistics_fault_rate > TARGET_LOGISTICS_FAULT * 1.2:
        st.error("🔴 물류사 개선 시급")
    elif logistics_fault_rate > TARGET_LOGISTICS_FAULT * 0.8:
        st.warning("🟡 물류사 점검 필요")
    else:
        st.success("🟢 물류 운영 정상")

st.markdown("---")

col1, col2 = st.columns(2)

# ===== 좌측: 비즈니스 개요 =====
with col1:
    st.markdown("### 📊 비즈니스 개요")
    
    # === 핵심 지표 4개 (2x2) ===
    kpi_col1, kpi_col2 = st.columns(2)
    
    with kpi_col1:
        # 주문 건수
        st.metric("총 주문 건수", f"{len(df_filtered):,}건")
        
        # 배송 지연율 (목표 대비)
        delay_delta = delay_rate - TARGET_DELAY_RATE
        st.metric(
            "배송 지연 발생률", 
            f"{delay_rate:.1f}%",
            delta=f"{delay_delta:+.1f}% (목표 대비)",
            delta_color="inverse"
        )
    
    with kpi_col2:
        # 평균 리뷰 평점 (목표 대비)
        avg_score = df_filtered['review_score'].mean()
        score_target = 4.0
        score_delta = avg_score - score_target
        st.metric(
            "평균 리뷰 평점", 
            f"{avg_score:.2f}점",
            delta=f"{score_delta:+.2f} (목표 대비)",
            delta_color="normal"
        )
        
        # 부정 리뷰율 (목표 대비)
        negative_delta = negative_rate - TARGET_NEGATIVE_RATE
        st.metric(
            "부정 리뷰율", 
            f"{negative_rate:.1f}%",
            delta=f"{negative_delta:+.1f}% (목표 대비)",
            delta_color="inverse"
        )
    
    # === 위험 신호 알림 박스 ===
    st.markdown("---")
    st.markdown("#### 🚨 주의 필요 항목")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        # 유의 판매자 현황
        risk_delta = risk_ratio - TARGET_SELLER_RISK_RATE
        
        if risk_ratio > TARGET_SELLER_RISK_RATE * 1.5:
            st.error(f"⚠️ **유의 판매자**: {risk_sellers}명 ({risk_ratio:.1f}%)")
            st.caption(f"목표 대비 {risk_delta:+.1f}% 초과")
        elif risk_ratio > TARGET_SELLER_RISK_RATE:
            st.warning(f"⚠️ **유의 판매자**: {risk_sellers}명 ({risk_ratio:.1f}%)")
            st.caption(f"목표 대비 {risk_delta:+.1f}% 초과")
        else:
            st.success(f"✅ **유의 판매자**: {risk_sellers}명 ({risk_ratio:.1f}%)")
            st.caption(f"목표 {TARGET_SELLER_RISK_RATE:.0f}% 이하 달성")
    
    with alert_col2:
        # 물류사 과실 현황
        logistics_delta = logistics_fault_rate - TARGET_LOGISTICS_FAULT
        
        if logistics_fault_rate > TARGET_LOGISTICS_FAULT * 1.2:
            st.error(f"🚚 **물류 문제율**: {logistics_fault_rate:.1f}%")
            st.caption(f"목표 대비 {logistics_delta:+.1f}% 초과")
        elif logistics_fault_rate > TARGET_LOGISTICS_FAULT * 0.8:
            st.warning(f"🚚 **물류 문제율**: {logistics_fault_rate:.1f}%")
            st.caption(f"목표 {TARGET_LOGISTICS_FAULT:.0f}% 부근")
        else:
            st.success(f"🚚 **물류 문제율**: {logistics_fault_rate:.1f}%")
            st.caption(f"목표 이하 달성")
    
    # === 추세 그래프 ===
    st.markdown("---")
    monthly_orders = df_filtered.groupby('year_month').size().reset_index(name='주문수')
    fig_trend = px.line(monthly_orders, x='year_month', y='주문수', 
                        title='📅 월별 주문 추이',
                        markers=True)
    fig_trend.update_layout(height=250, xaxis_title="", yaxis_title="주문 건수")
    st.plotly_chart(fig_trend, use_container_width=True)

# ===== 우측: 판매자 성과 분석 =====
with col2:
    st.markdown("### 🚚 판매자 성과 분석")
    
    # KPI 지표
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        avg_processing = df_filtered['seller_processing_days'].mean()
        st.metric("평균 처리 시간", f"{avg_processing:.1f}일")
    with kpi_col2:
        avg_delay = df_filtered['seller_delay_days'].mean()
        st.metric("평균 지연", f"{avg_delay:.1f}일")
    with kpi_col3:
        logistics_fault_rate = (df_filtered['is_logistics_fault'].sum() / len(df_filtered) * 100)
        st.metric("물류사 과실률", f"{logistics_fault_rate:.1f}%")
    
    # 판매자 처리 시간 분포
    fig_processing = px.histogram(df_filtered, x='seller_processing_days',
                                  title='⏱️ 판매자 처리 시간 분포',
                                  nbins=30)
    fig_processing.update_layout(height=300, xaxis_title="처리 시간 (일)", yaxis_title="주문 건수")
    fig_processing.add_vline(x=avg_processing, line_dash="dash", line_color="red", 
                            annotation_text=f"평균: {avg_processing:.1f}일")
    st.plotly_chart(fig_processing, use_container_width=True)
    
    # 배송 지연 vs 정상 배송
    delay_status = pd.DataFrame({
        '상태': ['정상 배송', '지연 발생'],
        '건수': [
            len(df_filtered[df_filtered['seller_delay_days'] <= 0]),
            len(df_filtered[df_filtered['seller_delay_days'] > 0])
        ]
    })
    fig_delay = px.bar(delay_status, x='상태', y='건수', 
                       title='📦 배송 상태 현황',
                       color='상태',
                       color_discrete_map={'정상 배송': 'green', '지연 발생': 'red'})
    fig_delay.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_delay, use_container_width=True)

st.markdown("---")

# =========================================================
# Section 3 & 4: 리뷰 분석 + 위험 판매자 (중단 Z자)
# =========================================================
st.markdown("## 💬 고객 만족도 및 위험 관리")

col3, col4 = st.columns(2)

# ===== 좌측: 리뷰 분석 =====
with col3:
    st.markdown("### ⭐ 리뷰 분석")
    
    # KPI 지표
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        text_review_rate = (df_filtered['has_text_review'].sum() / len(df_filtered) * 100)
        st.metric("텍스트 리뷰율", f"{text_review_rate:.1f}%")
    with kpi_col2:
        negative_rate = (len(df_filtered[df_filtered['review_score'] <= 2]) / len(df_filtered) * 100)
        st.metric("부정 리뷰율", f"{negative_rate:.1f}%")
    with kpi_col3:
        positive_rate = (len(df_filtered[df_filtered['review_score'] >= 4]) / len(df_filtered) * 100)
        st.metric("긍정 리뷰율", f"{positive_rate:.1f}%")
    
    # 리뷰 점수 분포
    review_dist = df_filtered['review_score'].value_counts().sort_index()
    fig_review = px.bar(x=review_dist.index, y=review_dist.values,
                        title='📊 리뷰 점수 분포',
                        labels={'x': '리뷰 점수', 'y': '건수'},
                        color=review_dist.index,
                        color_continuous_scale=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    fig_review.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_review, use_container_width=True)
    
    
    # 카테고리별 평균 리뷰 점수 (상위/하위 각 5개)
    category_review = df_filtered.groupby('product_category_name_english')['review_score'].agg(['mean', 'count'])
    category_review = category_review[category_review['count'] >= 10].sort_values('mean')
    
    top_bottom = pd.concat([category_review.head(5), category_review.tail(5)])
    fig_cat_review = px.bar(top_bottom.reset_index(), 
                            x='mean', 
                            y='product_category_name_english',
                            orientation='h',
                            title='🏷️ 카테고리별 평균 리뷰 점수 (상위/하위)',
                            labels={'mean': '평균 점수', 'product_category_name_english': ''},
                            color='mean',
                            color_continuous_scale='RdYlGn')
    fig_cat_review.update_layout(height=350)
    st.plotly_chart(fig_cat_review, use_container_width=True)

# ===== 우측: 위험 판매자 모니터링 =====
with col4:
    st.markdown("### 🚨 위험 판매자 조기 경보")
    
    if not df_risk.empty:
        # 위험 판매자 필터링
        risky_sellers = df_risk[df_risk['y_pred_proba'] >= risk_threshold]
        safe_sellers = df_risk[df_risk['y_pred_proba'] < risk_threshold]
        
        # KPI 지표
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.metric("감지된 위험 판매자", f"{len(risky_sellers)}명")
        with kpi_col2:
            st.metric("현재 Threshold", f"{risk_threshold:.2f}")
        with kpi_col3:
            if not risky_sellers.empty:
                st.metric("평균 위험 확률", f"{risky_sellers['y_pred_proba'].mean():.2f}")
            else:
                st.metric("평균 위험 확률", "N/A")
        
        # 위험도 분포
        fig_risk_hist = px.histogram(df_risk, x="y_pred_proba", nbins=20, 
                                     title="📊 전체 판매자 위험 확률 분포")
        fig_risk_hist.add_vline(x=risk_threshold, line_dash="dash", line_color="red", 
                               annotation_text="Threshold")
        fig_risk_hist.update_layout(height=300, xaxis_title="위험 확률", yaxis_title="판매자 수")
        st.plotly_chart(fig_risk_hist, use_container_width=True)
        
        # 상위 위험 판매자 TOP 10
        if not risky_sellers.empty:
            top_10 = risky_sellers.sort_values('y_pred_proba', ascending=False).head(10)
            fig_top_risk = px.bar(top_10, x='y_pred_proba', y='seller_id', 
                                 orientation='h',
                                 title="🚨 최상위 위험군 (Top 10)", 
                                 color='y_pred_proba', 
                                 color_continuous_scale='Reds')
            fig_top_risk.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_risk, use_container_width=True)
        else:
            st.info("현재 기준을 넘는 위험 판매자가 없습니다. ✅")
        
st.markdown("---")

# =========================================================
# Section 5: 최종 인사이트 (하단 전체 폭)
# =========================================================
st.markdown("## 🎯 최종 인사이트 및 권장사항")

tab1, tab2, tab3 = st.tabs(["💡 주요 발견사항", "📋 비즈니스 개선 권장사항", "⚠️ 위험 요인 및 대응 방안"])

with tab1:
    st.markdown("### 💡 데이터 분석 핵심 인사이트")
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown("#### 1. 배송 지연 분석")
        st.info("""
        **주요 발견:**
        - 배송 지연의 **90% 이상이 판매자 책임**
        - 판매자 처리 시간은 지역과 무관하게 평균 **2일** 일정
        - 지역 간 배송 시간 차이는 주로 **물류 배송 단계**에서 발생
        """)
        
        # 물류 vs 판매자 과실 비교
        fault_comparison = pd.DataFrame({
            '책임': ['물류사 과실', '판매자 책임'],
            '비율': [
                (df_filtered['is_logistics_fault'] == True).sum() / len(df_filtered) * 100,
                (df_filtered['is_logistics_fault'] == False).sum() / len(df_filtered) * 100
            ]
        })
        fig_fault = px.pie(fault_comparison, values='비율', names='책임',
                          title='📦 배송 문제 책임 소재',
                          color='책임',
                          color_discrete_map={'물류사 과실': 'orange', '판매자 책임': 'red'})
        st.plotly_chart(fig_fault, use_container_width=True)
    
    with col_insight2:
        st.markdown("#### 2. 카테고리 특성 분석")
        st.info("""
        **주요 발견:**
        - `office_furniture`, `baby` 카테고리에서 **처리 지연율 높음** (+27.4%, +6.3%)
        - 특수 포장/처리가 필요한 카테고리에서 지연 빈발
        - 카테고리별 맞춤형 배송 정책 필요성 대두
        """)
        
        # 카테고리별 평균 처리 시간 (상위 10개)
        cat_processing = df_filtered.groupby('product_category_name_english').agg({
            'seller_processing_days': 'mean',
            'order_id': 'count'
        }).reset_index()
        cat_processing = cat_processing[cat_processing['order_id'] >= 20].sort_values('seller_processing_days', ascending=False).head(10)
        
        fig_cat_proc = px.bar(cat_processing, 
                             x='seller_processing_days', 
                             y='product_category_name_english',
                             orientation='h',
                             title='⏱️ 처리 시간이 긴 카테고리 TOP 10',
                             labels={'seller_processing_days': '평균 처리 시간 (일)', 
                                    'product_category_name_english': ''},
                             color='seller_processing_days',
                             color_continuous_scale='Reds')
        st.plotly_chart(fig_cat_proc, use_container_width=True)
    
    st.markdown("#### 3. 유의 판매자 패턴")
    col_pattern1, col_pattern2, col_pattern3 = st.columns(3)
    
    with col_pattern1:
        seller_of_note_count = df_filtered[df_filtered['is_Seller_of_Note'] == True]['seller_id'].nunique()
        total_sellers = df_filtered['seller_id'].nunique()
        st.metric("유의 판매자 수", f"{seller_of_note_count}명", 
                 delta=f"{seller_of_note_count/total_sellers*100:.1f}%")
    
    with col_pattern2:
        if seller_of_note_count > 0:
            avg_score_note = df_filtered[df_filtered['is_Seller_of_Note'] == True]['review_score'].mean()
            avg_score_normal = df_filtered[df_filtered['is_Seller_of_Note'] == False]['review_score'].mean()
            st.metric("유의 판매자 평균 평점", f"{avg_score_note:.2f}점",
                     delta=f"{avg_score_note - avg_score_normal:.2f}점", delta_color="inverse")
        else:
            st.metric("유의 판매자 평균 평점", "N/A")
    
    with col_pattern3:
        if seller_of_note_count > 0:
            avg_delay_note = df_filtered[df_filtered['is_Seller_of_Note'] == True]['seller_processing_days'].mean()
            avg_delay_normal = df_filtered[df_filtered['is_Seller_of_Note'] == False]['seller_processing_days'].mean()
            st.metric("유의 판매자 평균 처리시간", f"{avg_delay_note:.1f}일",
                     delta=f"+{avg_delay_note - avg_delay_normal:.1f}일", delta_color="inverse")
        else:
            st.metric("유의 판매자 평균 처리시간", "N/A")

with tab2:
    st.markdown("### 📋 비즈니스 개선을 위한 실행 가능한 권장사항")
    
    st.markdown("#### 🎓 1. 판매자 교육 프로그램")
    st.success("""
    **목표:** 특정 카테고리 판매자의 처리 효율성 향상
    
    **실행 방안:**
    - `office_furniture`, `baby`, `pet_shop` 카테고리 판매자 대상 **특별 온보딩 과정** 운영
    - 포장 및 배송 준비 Best Practice 가이드 제공
    - 처리 시간 단축에 성공한 판매자 **인센티브 프로그램** 도입
    
    **예상 효과:**
    - 평균 처리 시간 **20-30% 단축**
    - 카테고리별 특화 노하우 축적
    """)
    
    st.markdown("#### 🚚 2. 물류 파트너십 재검토")
    st.success("""
    **목표:** 물류사 기인 배송 지연 최소화 (현재 전체 지연의 90%)
    
    **실행 방안:**
    - 지역별 물류사 성과 평가 및 **SLA(Service Level Agreement)** 강화
    - 주요 노선에 복수 물류 파트너 확보로 **위험 분산**
    - 실시간 배송 추적 시스템 고도화
    - 물류사별 인센티브/페널티 제도 도입
    
    **예상 효과:**
    - 배송 지연률 **50% 감소**
    - 고객 만족도 **15-20% 향상**
    """)
    
    st.markdown("#### 🆕 3. 초기 판매자 온보딩 강화")
    st.success("""
    **목표:** 신규 판매자의 조기 정착 및 품질 기준 준수
    
    **실행 방안:**
    - 첫 30일간 **집중 모니터링 기간** 설정
    - 판매자 등급제 도입 (Starter → Bronze → Silver → Gold)
    - 초기 3개월간 배송 품질 평가 후 **등급 조정**
    - 우수 판매자 **수수료 할인** 혜택 제공
    
    **예상 효과:**
    - 유의 판매자 발생률 **30% 감소**
    - 플랫폼 전체 품질 지표 향상
    """)
    
    st.markdown("#### 📊 4. 데이터 기반 의사결정 체계 구축")
    st.success("""
    **목표:** 실시간 모니터링 및 선제적 대응
    
    **실행 방안:**
    - **이 대시보드를 활용한 주간 리뷰** 정례화
    - 위험 판매자 조기 경보 시스템 알림 자동화
    - 카테고리별 KPI 목표 설정 및 추적
    - 분기별 전략 회의에서 인사이트 활용
    
    **예상 효과:**
    - 문제 상황 **조기 발견 및 대응**
    - CS 비용 **20-30% 절감**
    """)

with tab3:
    st.markdown("### ⚠️ 위험 요인 종합 분석 및 대응 전략")
    
    st.markdown("#### 📌 유의 판매자(Seller of Note) 정의")
    st.warning("""
    **유의 판매자란?**
    
    다음 3가지 조건을 **모두 충족**하는 판매자를 의미합니다:
    1. **처리 지연율** 높음 (상위 판매자 기준 75%, 중간 판매자 기준 90% 초과)
    2. **출고 기한 위반율** 높음 (동일 기준)
    3. **불만족 리뷰 비율** 높음 (동일 기준)
    
    플랫폼 운영 관점에서 **집중 관리가 필요한 판매자**로 분류됩니다.
    """)
    
    st.markdown("#### 🎯 조기 경보 시스템 활용 가이드")
    
    col_guide1, col_guide2 = st.columns(2)
    
    with col_guide1:
        st.info("""
        **Threshold 설정 가이드:**
        
        - **0.30 (기본)**: 균형잡힌 감지 - 적당한 수의 위험 판매자 포착
        - **0.20 (민감)**: 잠재적 위험까지 포함 - 예방적 관리
        - **0.50 (엄격)**: 고위험군만 집중 - 즉각 조치 필요 대상
        
        *사이드바에서 조정하여 실시간 확인 가능*
        """)
    
    with col_guide2:
        st.info("""
        **우선순위별 대응:**
        
        - **RED (0.6 이상)**: 즉각 제재/경고 발송
        - **ORANGE (0.4-0.6)**: 집중 모니터링 및 개선 권고
        - **YELLOW (0.3-0.4)**: 관찰 대상 등록
        
        *우선순위는 자동으로 할당됩니다*
        """)
    
    st.markdown("#### 🛡️ 단계별 관리 전략")
    
    st.markdown("""
    **1단계: 즉각 제재 및 개선 요구 (HIGH RISK)**
    - 대상: 위험 확률 0.6 이상 (RED)
    - 조치:
      - 즉시 **경고 메일 발송** (주요 위험 사유 명시)
      - **2주 내 개선 계획서 제출** 요구
      - 신규 상품 등록 **일시 제한**
      - 1주일 후 **재평가** 실시
    
    **2단계: 집중 모니터링 (MEDIUM RISK)**
    - 대상: 위험 확률 0.4-0.6 (ORANGE)
    - 조치:
      - **개선 권고 안내** 발송
      - 월 1회 **성과 리포트** 제공
      - 교육 프로그램 참여 권유
      - 월별 추이 모니터링
    
    **3단계: 관찰 대상 (LOW-MEDIUM RISK)**
    - 대상: 위험 확률 0.3-0.4 (YELLOW)
    - 조치:
      - **관찰 대상 등록**
      - 분기별 재평가
      - 자율적 개선 유도
    """)
    
    st.markdown("#### 💰 기대 효과")
    
    col_effect1, col_effect2, col_effect3 = st.columns(3)
    
    with col_effect1:
        st.metric("CS 비용 절감", "20-30%", delta="예상 절감률")
    
    with col_effect2:
        st.metric("고객 만족도 향상", "15-20%", delta="예상 향상률")
    
    with col_effect3:
        st.metric("플랫폼 신뢰도", "+25%", delta="예상 증가")

st.markdown("---")
st.markdown("**📍 데이터 기준:** 2016-2018년 Olist 주문 데이터 (68,468건) | **마지막 업데이트:** " + datetime.now().strftime("%Y-%m-%d"))
