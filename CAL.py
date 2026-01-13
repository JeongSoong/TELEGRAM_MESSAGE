import requests
from bs4 import BeautifulSoup
import yfinance as yf
from textblob import TextBlob
import pandas as pd

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
# 뉴스 감성 분석 (100점 환산)
# -----------------------------
def get_sentiment_score():
    try:
        url = "https://news.google.com/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        # 다양한 태그에서 제목 추출
        candidates = soup.select("h3, a span, article h4, article h3")
        headlines = [c.get_text(strip=True) for c in candidates][:10]

        if not headlines:
            return 50, ["뉴스 없음"]

        polarity_sum = sum(TextBlob(h).sentiment.polarity for h in headlines)
        avg_polarity = polarity_sum / len(headlines)

        score_100 = int((avg_polarity + 1) * 50)  # -1→0, 0→50, +1→100

        return score_100, headlines

    except Exception:
        return 50, ["뉴스 분석 실패"]

# -----------------------------
# 기술적 지표 (MACD, RSI, 볼린저밴드)
# -----------------------------
def compute_indicators(close_series: pd.Series):
    # RSI(14)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_latest = float(rsi.iloc[-1])

    # MACD(12,26,9)
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_latest = float(macd_line.iloc[-1])
    signal_latest = float(signal_line.iloc[-1])
    macd_hist = macd_latest - signal_latest

    # 볼린저밴드(20, 2)
    ma20 = close_series.rolling(window=20).mean()
    std20 = close_series.rolling(window=20).std()
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20

    price = float(close_series.iloc[-1])
    upper = float(upper_band.iloc[-1])
    lower = float(lower_band.iloc[-1])

    if upper != lower:
        bb_pos = (price - lower) / (upper - lower) * 100
    else:
        bb_pos = 50.0

    return {
        "rsi": rsi_latest,
        "macd": macd_latest,
        "macd_signal": signal_latest,
        "macd_hist": macd_hist,
        "bb_pos": bb_pos,
        "bb_upper": upper,
        "bb_lower": lower,
    }

# -----------------------------
# 공포탐욕지수(FGI)
# -----------------------------
def get_fgi():
    try:
        url = "https://edition.cnn.com/markets/fear-and-greed"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        value = None
        for tag in soup.find_all(["span", "div"]):
            text = tag.get_text(strip=True)
            if text.isdigit() and 0 <= int(text) <= 100:
                value = int(text)
                break

        if value is None:
            return None, "FGI 파싱 실패"

        if value <= 25: label = "극단적 공포"
        elif value <= 45: label = "공포"
        elif value < 55: label = "중립"
        elif value < 75: label = "탐욕"
        else: label = "극단적 탐욕"

        return value, label

    except Exception:
        return None, "FGI 오류"

# -----------------------------
# 환율 / 금리 / 유가
# -----------------------------
def get_macro_data():
    try:
        fx = yf.Ticker("USDKRW=X").history(period="1d")
        fx_now = float(fx["Close"][0])

        tnx = yf.Ticker("^TNX").history(period="1d")
        tnx_now = float(tnx["Close"][0])

        oil = yf.Ticker("CL=F").history(period="1d")
        oil_now = float(oil["Close"][0])

        return fx_now, tnx_now, oil_now
    except:
        return None, None, None

# -----------------------------
# 시장 데이터 수집
# -----------------------------
def fetch_market_data():
    sp_hist = yf.Ticker("^GSPC").history(period="30d")
    ndx_hist = yf.Ticker("^NDX").history(period="1d")
    vix_hist = yf.Ticker("^VIX").history(period="1d")

    sp_change = float((sp_hist["Close"].iloc[-1] - sp_hist["Open"].iloc[-1]) / sp_hist["Open"].iloc[-1] * 100)
    ndx_change = float((ndx_hist["Close"][0] - ndx_hist["Open"][0]) / ndx_hist["Open"][0] * 100)
    vix_value = float(vix_hist["Close"][0])

    indicators = compute_indicators(sp_hist["Close"])

    sentiment_score, headlines = get_sentiment_score()
    fgi_value, fgi_label = get_fgi()
    fx_now, tnx_now, oil_now = get_macro_data()

    return {
        "sp_change": sp_change,
        "ndx_change": ndx_change,
        "vix_value": vix_value,
        **indicators,
        "sentiment_score": sentiment_score,
        "headlines": headlines,
        "fgi_value": fgi_value,
        "fgi_label": fgi_label,
        "fx_now": fx_now,
        "tnx_now": tnx_now,
        "oil_now": oil_now,
    }

# -----------------------------
# 메인 실행
# -----------------------------
def main():
    data = fetch_market_data()

    sp_change = data["sp_change"]
    ndx_change = data["ndx_change"]
    vix_value = data["vix_value"]
    rsi = data["rsi"]
    macd = data["macd"]
    macd_signal = data["macd_signal"]
    macd_hist = data["macd_hist"]
    bb_pos = data["bb_pos"]
    bb_upper = data["bb_upper"]
    bb_lower = data["bb_lower"]
    sentiment_score = data["sentiment_score"]
    headlines = data["headlines"]
    fgi_value = data["fgi_value"]
    fgi_label = data["fgi_label"]
    fx_now = data["fx_now"]
    tnx_now = data["tnx_now"]
    oil_now = data["oil_now"]

    # -----------------------------
    # 기존 점수 계산
    # -----------------------------
    raw_score = 0

    if rsi >= 90: raw_score += 3
    elif rsi >= 85: raw_score += 2
    elif rsi >= 80: raw_score += 1

    change = max(abs(sp_change), abs(ndx_change))
    if change >= 10: raw_score += 3
    elif change >= 8: raw_score += 2
    elif change >= 5: raw_score += 1

    if vix_value <= 12: raw_score += 3
    elif vix_value <= 14: raw_score += 2
    elif vix_value <= 16: raw_score += 1

    if sentiment_score >= 70: raw_score += 2
    elif sentiment_score >= 55: raw_score += 1

    # -----------------------------
    # 총점수 100점 환산
    # -----------------------------
    final_score = int(raw_score / 15 * 100)

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
        f"MACD: {macd:.4f} / Signal: {macd_signal:.4f} / Hist: {macd_hist:.4f}\n"
        f"볼린저 위치: {bb_pos:.1f}% (상단 {bb_upper:.2f}, 하단 {bb_lower:.2f})\n\n"

        f"뉴스 감성 점수: {sentiment_score}/100\n"
        f"공포탐욕지수(FGI): {fgi_value} ({fgi_label})\n"
        f"USD/KRW: {fx_now:,.2f}원\n"
        f"미국 10년물 금리: {tnx_now:.2f}%\n"
        f"WTI 유가: {oil_now:.2f}달러\n\n"

        f"총 점수: {final_score}/100\n"
        f"결론: {result}\n\n"

        f"[포트폴리오 배분]\n{allocation}\n\n"
        f"[주요 뉴스]\n - " + "\n - ".join(headlines[:3])
    )

    send_telegram(telegram_message)
    print("텔레그램 전송 완료")

if __name__ == "__main__":
    main()
