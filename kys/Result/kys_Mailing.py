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

# 1. 메일 전송 함수 (Priority 기반)
def send_risk_report(df, threshold=0.3, receiver_email=None):
    # 이메일 주소가 없으면 환경변수에서 가져오기
    if receiver_email is None:
        receiver_email = os.getenv('GMAIL_USER')

    sender_email = os.getenv('GMAIL_USER')
    app_password = os.getenv('GMAIL_PASSWORD')

    # (Threshold 필터링)
    risky_sellers = df[df['y_pred_proba'] >= threshold].copy()
    
    # (위험한 사람이 없으면 메일 안 보내기)
    if len(risky_sellers) == 0:
        print(f"[{datetime.datetime.now()}] 위험 판매자 없음.")
        return

    # Priority 기준으로 분류
    red_zone = risky_sellers[risky_sellers['priority'] == 'RED'].copy()
    yellow_zone = risky_sellers[risky_sellers['priority'] == 'YELLOW'].copy()
    
    # RED ZONE: 전체 표시 (즉시 대응 필요)
    # YELLOW ZONE: 상위 10명만 표시 (모니터링 대상)
    yellow_zone_top10 = yellow_zone.head(10)
    
    today_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # HTML 표 만들기
    red_html = ""
    if len(red_zone) > 0:
        red_html = f"""
        <h3 style="color: red;">RED ZONE - 즉시 대응 필요 ({len(red_zone)}명)</h3>
        <p style="color: red;"><strong>위험 확률 0.8 이상 - 즉시 전화 확인 및 모니터링 강화</strong></p>
        {red_zone.to_html(index=False, border=1, classes='red-zone')}
        <br>
        """
    
    yellow_html = ""
    if len(yellow_zone) > 0:
        yellow_html = f"""
        <h3 style="color: orange;">YELLOW ZONE - 관심 리스트 (전체 {len(yellow_zone)}명 중 상위 10명)</h3>
        <p style="color: orange;"><strong>위험 확률 0.25~0.79 - 배송 현황 모니터링</strong></p>
        {yellow_zone_top10.to_html(index=False, border=1, classes='yellow-zone')}
        """
    
    # HTML 본문
    html_body = f"""
    <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th {{ background-color: #f2f2f2; padding: 8px; text-align: left; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .red-zone th {{ background-color: #ffcccc; }}
                .yellow-zone th {{ background-color: #fff4cc; }}
            </style>
        </head>
        <body>
            <h2>조기 경보 리포트 ({today_date})</h2>
            <p>총 위험 판매자: <b>{len(risky_sellers)}명</b> (RED: {len(red_zone)}명, YELLOW: {len(yellow_zone)}명)</p>
            <hr>
            {red_html}
            {yellow_html}
            <hr>
            <p style="font-size: 12px; color: gray;">
            * RED ZONE: 즉시 대응 필요<br>
            * YELLOW ZONE: 지속 모니터링 (배송 지연 발생 시 RED로 상향)
            </p>
        </body>
    </html>
    """

    # SMTP 설정
    msg = MIMEMultipart()
    msg['From'] = "Risk Alarm System"
    msg['To'] = receiver_email
    msg['Subject'] = f"[{today_date}] 위험 판매자 리포트"
    msg.attach(MIMEText(html_body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"[{today_date}] 메일 전송 완료!")
        print(f"  - 총 위험 판매자: {len(risky_sellers)}명")
        print(f"  - 🔴 RED ZONE: {len(red_zone)}명 (즉시 대응)")
        print(f"  - 🟡 YELLOW ZONE: {len(yellow_zone)}명 (모니터링)")
    except Exception as e:
        print(f"전송 실패: {e}")

# 2. 스케줄러가 실행할 작업 
def job():
    print(f"\n[스케줄러 실행] {datetime.datetime.now()}")
    
    try:
        # csv 파일 이름이 맞는지 꼭 확인하세요
        df = pd.read_csv('kys/Result/risk_report_result.csv')
        print("데이터 로드 성공. 메일 전송을 시도합니다.")
        
        # 메일 전송 함수 호출 (threshold=0.25: YELLOW ZONE 기준)
        send_risk_report(df, threshold=0.25, receiver_email="kyus0919@gmail.com")
        
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
    
    # (테스트용) 10초마다 실행 -> 테스트 후엔 주석 처리하고 위 코드를 푸세요!
    schedule.every(10).seconds.do(job)

    # 무한 루프 (프로그램이 꺼지지 않게 함)
    while True:
        schedule.run_pending()
        time.sleep(1)