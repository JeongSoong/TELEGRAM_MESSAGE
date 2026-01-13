import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# 텔레그램 전송
# -----------------------------
def send_telegram(message):
    bot_token = "8386665445:AAG5bEM30o9UzU-9NO9cGM7Lg0K7b1xcbFk"
    chat_id = "6983611450"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(url, data=payload)

# -----------------------------
# 뉴스 감성 분석 (RSS + VADER)
# -----------------------------
def get_news_sentiment():
    try:
        url = "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"
        xml = requests.get(url).text
        soup = BeautifulSoup(xml, "xml")

        items = soup.find_all("item")[:10]
        headlines = [item.title.get_text(strip=True) for item in items]

        if not headlines:
            return 50, ["뉴스 없음"]

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]

        avg_score = sum(scores) / len(scores)  # -1 ~ +1
        score_100 = int((avg_score + 1) * 50)  # 0~100 변환

        return score_100, headlines

    except:
        return 50, ["뉴스 분석 실패"]

# -----------------------------
# 기술적 지표 (MACD, RSI, 볼린저)
# -----------------------------
def compute_indicators(close_series):
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_latest = float(rsi.iloc[-1])

    ema12 = close_series.ewm(span=12).mean()
    ema26 = close_series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_hist = macd - signal

    macd_latest = float(macd.iloc[-1])
    signal_latest = float(signal.iloc[-1])
    hist_latest = float(macd_hist.iloc[-1])

    ma20 = close_series.rolling(20).mean()
    std20 = close_series.rolling(20).std()
    upper = float((ma20 + 2 * std20).iloc[-1])
    lower = float((ma20 - 2 * std20).iloc[-1])
    price = float(close_series.iloc[-1])

    bb_pos = (price - lower) / (upper - lower) * 100

    return rsi_latest, macd_latest, signal_latest, hist_latest, bb_pos, upper, lower

# -----------------------------
# Proxy FGI (서버 100% 호환)
# -----------------------------
def compute_proxy_fgi():
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"][0]
        putcall = yf.Ticker("^PCR").history(period="1d")["Close"][0]
        junk = yf.Ticker("HYG").history(period="1d")["Close"][0]
        gold = yf.Ticker("GC=F").history(period="1d")["Close"][0]
        sp = yf.Ticker("^GSPC").history(period="1d")["Close"][0]

        safe_haven = gold / sp

        # 0~100 점수화
        vix_score = max(0, min(100, 100 - vix * 3))
        putcall_score = max(0, min(100, 150 - putcall * 100))
        junk_score = max(0, min(100, junk / 10))
        safe_score = max(0, min(100, safe_haven * 100))

        proxy_fgi = int((vix_score + putcall_score + junk_score + safe_score) / 4)

        return proxy_fgi

    except:
        return 50  # 중립

# -----------------------------
# 메인 실행
# -----------------------------
def main():
    sp = yf.Ticker("^GSPC").history(period="30d")
    ndx = yf.Ticker("^NDX").history(period="1d")
    vix = yf.Ticker("^VIX").history(period="1d")

    sp_change = float((sp["Close"].iloc[-1] - sp["Open"].iloc[-1]) / sp["Open"].iloc[-1] * 100)
    ndx_change = float((ndx["Close"][0] - ndx["Open"][0]) / ndx["Open"][0] * 100)
    vix_value = float(vix["Close"][0])

    rsi, macd, signal, hist, bb_pos, bb_upper, bb_lower = compute_indicators(sp["Close"])

    sentiment_score, headlines = get_news_sentiment()
    proxy_fgi = compute_proxy_fgi()

    fx = yf.Ticker("USDKRW=X").history(period="1d")["Close"][0]
    tnx = yf.Ticker("^TNX").history(period="1d")["Close"][0]
    oil = yf.Ticker("CL=F").history(period="1d")["Close"][0]

    # -----------------------------
    # 100점 만점 종합 점수
    # -----------------------------
    tech_score = 0
    if rsi >= 80: tech_score += 10
    if bb_pos >= 80: tech_score += 10
    if macd > signal: tech_score += 10
    if vix_value <= 15: tech_score += 10

    news_score = sentiment_score * 0.3
    fgi_score = proxy_fgi * 0.3

    final_score = int(tech_score + news_score + fgi_score)

    if final_score >= 70:
        result = "전량 매도"
    elif final_score >= 40:
        result = "분할 매도"
    else:
        result = "모으기"

    # -----------------------------
    # 포트폴리오 배분
    # -----------------------------
    portfolio = {
        "TECL": 20,
        "SOXL": 25,
        "ETHU": 10,
        "SOLT": 10,
        "INDL": 10,
        "FNGU": 15,
        "WEBL": 10
    }

    avg_change = (sp_change + ndx_change) / 2
    base_amount = 10000 if avg_change >= 0 else 20000

    allocation = "\n".join(
        f"{ticker}: {base_amount * weight / 100:,.0f}원"
        for ticker, weight in portfolio.items()
    )

    # -----------------------------
    # 텔레그램 메시지
    # -----------------------------
    telegram_message = (
        f"[정수 버블 체크]\n"
        f"S&P 변동폭: {sp_change:.2f}%\n"
        f"나스닥 변동폭: {ndx_change:.2f}%\n"
        f"VIX: {vix_value:.2f}\n\n"

        f"RSI(14): {rsi:.2f}\n"
        f"MACD: {macd:.4f} / Signal: {signal:.4f} / Hist: {hist:.4f}\n"
        f"볼린저 위치: {bb_pos:.1f}% (상단 {bb_upper:.2f}, 하단 {bb_lower:.2f})\n\n"

        f"뉴스 감성 점수: {sentiment_score}/100\n"
        f"Proxy FGI: {proxy_fgi}/100\n"
        f"USD/KRW: {fx:,.2f}원\n"
        f"미국 10년물 금리: {tnx:.2f}%\n"
        f"WTI 유가: {oil:.2f}달러\n\n"

        f"총 점수: {final_score}/100\n"
        f"결론: {result}\n\n"

        f"[포트폴리오 배분]\n{allocation}\n\n"
        f"[주요 뉴스]\n - " + "\n - ".join(headlines[:3])
    )

    send_telegram(telegram_message)
    print("텔레그램 전송 완료")

if __name__ == "__main__":
    main()
