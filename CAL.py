import requests
import yfinance as yf
from bs4 import BeautifulSoup
from textblob import TextBlob

# -----------------------------
# 텔레그램 전송 함수
# -----------------------------
TELEGRAM_TOKEN = "정수_토큰"
CHAT_ID = "정수_챗아이디"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    requests.post(url, data=data)


# -----------------------------
# 뉴스 가져오기 + 감성 분석
# -----------------------------
def get_news_sentiment(keyword):
    try:
        url = f"https://news.google.com/search?q={keyword}+stock&hl=en-US&gl=US&ceid=US:en"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        headlines = [h.text for h in soup.select("h3")][:10]  # 상위 10개 뉴스

        if not headlines:
            return 0, ["뉴스 없음"]

        sentiment_score = 0
        for h in headlines:
            sentiment_score += TextBlob(h).sentiment.polarity

        return sentiment_score, headlines

    except Exception:
        return 0, ["뉴스 분석 실패"]


# -----------------------------
# 종목 분석 함수
# -----------------------------
def analyze_stock(ticker):
    try:
        data = yf.Ticker(ticker).history(period="2d")

        if len(data) < 2:
            return f"{ticker}: 데이터 부족"

        open_price = data["Open"].iloc[-1]
        close_price = data["Close"].iloc[-1]
        change = (close_price - open_price) / open_price * 100

        sentiment_score, headlines = get_news_sentiment(ticker)

        result = (
            f"📌 {ticker}\n"
            f"가격 변화: {change:.2f}%\n"
            f"감성 점수: {sentiment_score:.2f}\n"
            f"주요 뉴스:\n - " + "\n - ".join(headlines[:3])
        )

        return result

    except Exception as e:
        return f"{ticker}: 분석 중 오류 발생 → {e}"


# -----------------------------
# 메인 실행
# -----------------------------
def main():
    try:
        tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]

        results = []
        for t in tickers:
            results.append(analyze_stock(t))

        final_message = "📊 오늘의 자동 분석 결과\n\n" + "\n\n".join(results)
        send_telegram(final_message)

    except Exception as e:
        send_telegram(f"❌ 시스템 오류 발생: {e}")


if __name__ == "__main__":
    main()
