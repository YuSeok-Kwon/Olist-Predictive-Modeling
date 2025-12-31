import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from dotenv import load_dotenv
import os
import schedule
import time
import datetime

# 0. .env 파일 로드
load_dotenv()

# 1. 4단계 Priority 분류 함수
def assign_priority_4tier(prob):
    """
    위험 확률에 따라 4단계로 분류
    - RED: 0.8 이상 (즉각 제재 및 개선 요구)
    - ORANGE: 0.4 ~ 0.79 (집중 모니터링)
    - YELLOW: 0.3 ~ 0.39 (관찰 대상)
    - GREEN: 0.3 미만 (안전 판매자)
    """
    if prob >= 0.8:
        return 'RED'
    elif prob >= 0.4:
        return 'ORANGE'
    elif prob >= 0.3:
        return 'YELLOW'
    else:
        return 'GREEN'

# 2. 메일 전송 함수 (4단계 Priority 기반)
def send_risk_report(df, threshold=0.0, receiver_email=None):
    """
    4단계 위험도 체계로 판매자 리포트 메일 발송
    threshold=0.0: 전체 판매자 포함 (GREEN 포함)
    """
    # 이메일 주소가 없으면 환경변수에서 가져오기
    if receiver_email is None:
        receiver_email = os.getenv('GMAIL_USER')

    sender_email = os.getenv('GMAIL_USER')
    app_password = os.getenv('GMAIL_PASSWORD')

    # threshold 이상의 판매자만 필터링 (기본값 0.0이면 전체 포함)
    filtered_sellers = df[df['y_pred_proba'] >= threshold].copy()
    
    # 데이터가 없으면 메일 안 보내기
    if len(filtered_sellers) == 0:
        print(f"[{datetime.datetime.now()}] 리포트 대상 판매자 없음.")
        return

    # 4단계 Priority 재분류 (y_pred_proba 기반)
    filtered_sellers['priority_4tier'] = filtered_sellers['y_pred_proba'].apply(assign_priority_4tier)
    
    # 각 Zone별 분류
    red_zone = filtered_sellers[filtered_sellers['priority_4tier'] == 'RED'].copy()
    orange_zone = filtered_sellers[filtered_sellers['priority_4tier'] == 'ORANGE'].copy()
    yellow_zone = filtered_sellers[filtered_sellers['priority_4tier'] == 'YELLOW'].copy()
    green_zone = filtered_sellers[filtered_sellers['priority_4tier'] == 'GREEN'].copy()
    
    # 각 Zone별 표시 인원 제한
    # RED: 전체 표시, ORANGE: 상위 10명, YELLOW: 상위 5명, GREEN: 상위 5명
    orange_zone_top = orange_zone.head(10)
    yellow_zone_top = yellow_zone.head(5)
    green_zone_top = green_zone.head(5)
    
    today_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # ===== HTML 섹션 생성 =====
    
    # RED ZONE HTML
    red_html = ""
    if len(red_zone) > 0:
        red_html = f"""
        <h3 style="color: #d32f2f;">RED ZONE - 즉각 제재 및 개선 요구 ({len(red_zone)}명)</h3>
        <p style="color: #d32f2f;"><strong>위험 확률 ≥ 0.8</strong></p>
        <ul style="color: #d32f2f; font-size: 14px;">
            <li>즉시 <strong>경고 메일 발송</strong></li>
            <li><strong>2주 내 개선 계획서 제출</strong> 요구</li>
            <li>신규 상품 등록 <strong>일시 제한</strong></li>
            <li>1주일 후 <strong>재평가</strong> 실시</li>
        </ul>
        {red_zone.to_html(index=False, border=1, classes='red-zone')}
        <br>
        """
    
    # ORANGE ZONE HTML (집중 모니터링)
    orange_html = ""
    if len(orange_zone) > 0:
        orange_html = f"""
        <h3 style="color: #f57c00;">ORANGE ZONE - 집중 모니터링 (전체 {len(orange_zone)}명 중 상위 10명)</h3>
        <p style="color: #f57c00;"><strong>위험 확률 0.4 ~ 0.79</strong></p>
        <ul style="color: #f57c00; font-size: 14px;">
            <li><strong>개선 권고 안내</strong> 발송</li>
            <li>월 1회 <strong>성과 리포트</strong> 제공</li>
            <li>교육 프로그램 참여 권유</li>
            <li>월별 추이 모니터링</li>
        </ul>
        {orange_zone_top.to_html(index=False, border=1, classes='orange-zone')}
        <br>
        """
    
    # YELLOW ZONE HTML (관찰 대상)
    yellow_html = ""
    if len(yellow_zone) > 0:
        yellow_html = f"""
        <h3 style="color: #fbc02d;">YELLOW ZONE - 관찰 대상 (전체 {len(yellow_zone)}명 중 상위 5명)</h3>
        <p style="color: #fbc02d;"><strong>위험 확률 0.3 ~ 0.39</strong></p>
        <ul style="color: #fbc02d; font-size: 14px;">
            <li><strong>관찰 대상 등록</strong></li>
            <li>분기별 재평가</li>
            <li>자율적 개선 유도</li>
        </ul>
        {yellow_zone_top.to_html(index=False, border=1, classes='yellow-zone')}
        <br>
        """
    
    # GREEN ZONE HTML (안전 판매자 / 우수 사례)
    green_html = ""
    if len(green_zone) > 0:
        green_html = f"""
        <h3 style="color: #388e3c;">GREEN ZONE - 안전 판매자 / 우수 사례 (전체 {len(green_zone)}명 중 상위 5명)</h3>
        <p style="color: #388e3c;"><strong>위험 확률 &lt; 0.3</strong></p>
        <ul style="color: #388e3c; font-size: 14px;">
            <li>우수 판매자 인센티브 제공</li>
            <li>베스트 프랙티스 사례 공유</li>
            <li>정기적 만족도 조사</li>
        </ul>
        {green_zone_top.to_html(index=False, border=1, classes='green-zone')}
        """
    
    # HTML 본문
    html_body = f"""
    <html>
        <head>
            <style>
                body {{ font-family: 'Malgun Gothic', Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th {{ background-color: #f2f2f2; padding: 8px; text-align: left; font-weight: bold; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .red-zone th {{ background-color: #ffcccc; }}
                .orange-zone th {{ background-color: #ffe0b2; }}
                .yellow-zone th {{ background-color: #fff9c4; }}
                .green-zone th {{ background-color: #c8e6c9; }}
                hr {{ border: 0; height: 1px; background: #ddd; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h2>판매자 위험도 종합 리포트 ({today_date})</h2>
            <p style="font-size: 16px;">
                총 분석 판매자: <b>{len(filtered_sellers)}명</b><br>
                RED: <b>{len(red_zone)}명</b> | 
                ORANGE: <b>{len(orange_zone)}명</b> | 
                YELLOW: <b>{len(yellow_zone)}명</b> | 
                GREEN: <b>{len(green_zone)}명</b>
            </p>
            <hr>
            {red_html}
            {orange_html}
            {yellow_html}
            {green_html}
            <hr>
            <p style="font-size: 12px; color: gray;">
            <strong>단계별 관리 전략</strong><br>
            * <strong>RED ZONE</strong>: 즉각 제재 및 개선 요구 (위험 확률 ≥ 0.8)<br>
            * <strong>ORANGE ZONE</strong>: 집중 모니터링 (위험 확률 0.4-0.79)<br>
            * <strong>YELLOW ZONE</strong>: 관찰 대상 (위험 확률 0.3-0.39)<br>
            * <strong>GREEN ZONE</strong>: 안전 판매자 / 우수 사례 (위험 확률 &lt; 0.3)
            </p>
        </body>
    </html>
    """

    # SMTP 설정
    msg = MIMEMultipart()
    msg['From'] = "Risk Alarm System"
    msg['To'] = receiver_email
    msg['Subject'] = f"[{today_date}] 판매자 위험도 종합 리포트 (RED:{len(red_zone)} | ORANGE:{len(orange_zone)} | YELLOW:{len(yellow_zone)} | GREEN:{len(green_zone)})"
    msg.attach(MIMEText(html_body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"[{today_date}] 메일 전송 완료!")
        print(f"  - 총 분석 판매자: {len(filtered_sellers)}명")
        print(f"  - RED ZONE: {len(red_zone)}명 (즉각 제재)")
        print(f"  - ORANGE ZONE: {len(orange_zone)}명 (집중 모니터링)")
        print(f"  - YELLOW ZONE: {len(yellow_zone)}명 (관찰 대상)")
        print(f"  - GREEN ZONE: {len(green_zone)}명 (안전 판매자)")
    except Exception as e:
        print(f"전송 실패: {e}")

# 2. 스케줄러가 실행할 작업 
def job():
    print(f"\n[스케줄러 실행] {datetime.datetime.now()}")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'risk_report_result.csv')
        
        df = pd.read_csv(csv_path)
        print(f"데이터 로드 성공: {csv_path}")
        print("메일 전송을 시도합니다.")
        
        # 메일 전송 함수 호출 (threshold=0.0: 전체 판매자 포함, GREEN까지)
        send_risk_report(df, threshold=0.0, receiver_email="kyus0919@gmail.com")
        
    except FileNotFoundError:
        print("[오류] 분석 결과 파일(CSV)이 없습니다.")
    except Exception as e:
        print(f"[오류] 작업 중 에러 발생: {e}")

# 3. 스케줄 설정 및 실행
if __name__ == "__main__":
    print("자동 메일링 시스템이 시작되었습니다.")
    print("메일이 매일 오전 09:00에 발송됩니다. (종료: Ctrl + C)")

    # --- 스케줄 설정 ---
    # 매일 아침 9시 실행
    # schedule.every().day.at("09:00").do(job)
    
    # (테스트용) 10초마다 실행
    # schedule.every(10).seconds.do(job)

    # 무한 루프 (프로그램이 꺼지지 않게 함)
    while True:
        schedule.run_pending()
        time.sleep(1)