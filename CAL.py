import requests
from bs4 import BeautifulSoup
import yfinance as yf
from textblob import TextBlob
import pandas as pd

def send_telegram(message):
    bot_token = "8386665445:AAG5bEM30o9UzU-9NO9cGM7Lg0K7b1xcbFk"
    chat_id = "6983611450"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(url, data=payload)

# -----------------------------
# 뉴스 기반 감성 분석 (100점 환산)
# -----------------------------
def get_sentiment_score():
    try:
        url = "https://news.google.com/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        headlines = [h.text for h in soup.find_all("h3")][:10]

        if not headlines:
            return 50, ["뉴스 없음"]  # 중립 점수

        polarity_sum = 0
        for h in headlines:
            polarity_sum += TextBlob(h).sentiment.polarity

        avg_polarity = polarity_sum / len(headlines)  # -1 ~ +1
        score_100 = int((avg_polarity + 1) * 50)      # -1→0, 0→50, +1→100

        return score_100, headlines

    except Exception:
        return 50, ["뉴스 분석 실패"]

# -----------------------------
# 기술적 지표 계산 (MACD, RSI, 볼린저)
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
    mid = float(ma20.iloc[-1])

    # 밴드 내 위치(%)
    if upper != lower:
        bb_pos = (price - lower) / (upper - lower) * 100
    else:
        bb_pos = 50.0

    return {
        "rsi": rsi_latest,
        "macd": macd_latest,
        "macd_signal": signal_latest,
        "macd_hist": macd_hist,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_mid": mid,
        "bb_pos": bb_pos,
        "price": price,
    }

# -----------------------------
# 공포탐욕지수(FGI) 크롤링
# -----------------------------
def get_fgi():
    try:
        url = "https://edition.cnn.com/markets/fear-and-greed"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        # 페이지 구조 바뀔 수 있음 → 최대한 일반적으로 파싱
        # 숫자 크게 표시된 곳 찾아서 첫 번째 숫자만 사용
        value = None
        for tag in soup.find_all(["span", "div"]):
            text = tag.get_text(strip=True)
            if text.isdigit() and 0 <= int(text) <= 100:
                value = int(text)
                break

        if value is None:
            return None, "FGI 파싱 실패"

        if value <= 25:
            label = "극단적 공포"
        elif value <= 45:
            label = "공포"
        elif value < 55:
            label = "중립"
        elif value < 75:
            label = "탐욕"
        else:
            label = "극단적 탐욕"

        return value, label

    except Exception:
        return None, "FGI 크롤링 오류"

# -----------------------------
# 환율 / 금리 / 유가
# -----------------------------
def get_macro_data():
    try:
        # USD/KRW
        fx = yf.Ticker("USDKRW=X").history(period="1d")
        fx_now = float(fx["Close"][0])

        # 미국 10년물 금리 (^TNX는 % 단위)
        tnx = yf.Ticker("^TNX").history(period="1d")
        tnx_now = float(tnx["Close"][0])

        # WTI 유가 (CL=F)
        oil = yf.Ticker("CL=F").history(period="1d")
        oil_now = float(oil["Close"][0])

        return fx_now, tnx_now, oil_now
    except Exception:
        return None, None, None

# -----------------------------
# 시장 데이터 수집 + 지표
# -----------------------------
def fetch_market_data():
    # S&P, 나스닥, VIX
    sp_hist = yf.Ticker("^GSPC").history(period="30d")
    ndx_hist = yf.Ticker("^NDX").history(period="1d")
    vix_hist = yf.Ticker("^VIX").history(period="1d")

    sp_change = float((sp_hist["Close"].iloc[-1] - sp_hist["Open"].iloc[-1]) / sp_hist["Open"].iloc[-1] * 100)
    ndx_change = float((ndx_hist["Close"][0] - ndx_hist["Open"][0]) / ndx_hist["Open"][0] * 100)
    vix_value = float(vix_hist["Close"][0])

    # 기술적 지표 (S&P 기준)
    indicators = compute_indicators(sp_hist["Close"])
    rsi = indicators["rsi"]
    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    macd_hist = indicators["macd_hist"]
    bb_pos = indicators["bb_pos"]
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]

    # 뉴스 감성
    sentiment_score, headlines = get_sentiment_score()

    # FGI
    fgi_value, fgi_label = get_fgi()

    # 환율 / 금리 / 유가
    fx_now, tnx_now, oil_now = get_macro_data()

    return {
        "sp_change": sp_change,
        "ndx_change": ndx_change,
        "vix_value": vix_value,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_pos": bb_pos,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
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

    change = max(abs(sp_change), abs(ndx_change))

    score = 0
    # RSI
    if rsi >= 90: score += 3
    elif rsi >= 85: score += 2
    elif rsi >= 80: score += 1

    # 지수 변동폭
    if change >= 10: score += 3
    elif change >= 8: score += 2
    elif change >= 5: score += 1

    # VIX
    if vix_value <= 12: score += 3
    elif vix_value <= 14: score += 2
    elif vix_value <= 16: score += 1

    # 뉴스 감성 (100점 기준)
    if sentiment_score >= 70: score += 2
    elif sentiment_score >= 55: score += 1

    if score >= 10:
        result = "전량 매도"
    elif score >= 5:
        result = "분할 매도"
    else:
        result = "모으기"

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

    allocation_lines = []
    for ticker, weight in portfolio.items():
        amount = base_amount * weight / 100
        allocation_lines.append(f"{ticker}: {amount:,.0f}원")
    allocation = "\n".join(allocation_lines)

    # FGI 텍스트
    if fgi_value is None:
        fgi_text = f"{fgi_label}"
    else:
        fgi_text = f"{fgi_value} ({fgi_label})"

    # 매크로 텍스트
    fx_text = f"{fx_now:,.2f}원" if fx_now is not None else "데이터 없음"
    tnx_text = f"{tnx_now:.2f}%" if tnx_now is not None else "데이터 없음"
    oil_text = f"{oil_now:.2f}달러" if oil_now is not None else "데이터 없음"

    telegram_message = (
        f"[정수 버블 체크]\n"
        f"S&P 변동폭: {sp_change:.2f}%\n"
        f"나스닥 변동폭: {ndx_change:.2f}%\n"
        f"VIX: {vix_value:.2f}\n"
        f"RSI(14): {rsi:.2f}\n"
        f"MACD: {macd:.4f} / Signal: {macd_signal:.4f} / Hist: {macd_hist:.4f}\n"
        f"볼린저 위치: 상단 {bb_upper:.2f}, 하단 {bb_lower:.2f}, 밴드 내 위치 {bb_pos:.1f}%\n"
        f"뉴스 감성 점수: {sentiment_score}/100\n"
        f"공포탐욕지수(FGI): {fgi_text}\n"
        f"USD/KRW: {fx_text}\n"
        f"미국 10년물: {tnx_text}\n"
        f"WTI 유가: {oil_text}\n"
        f"총 점수: {score}\n"
        f"결론: {result}\n\n"
        f"지수 평균 변동률: {avg_change:.4f}%\n"
        f"기준금액: {base_amount:,}원\n\n"
        f"[포트폴리오 배분]\n{allocation}\n\n"
        f"[주요 뉴스]\n - " + "\n - ".join(headlines[:3])
    )

    send_telegram(telegram_message)
    print("텔레그램 전송 완료")

if __name__ == "__main__":
    main()
