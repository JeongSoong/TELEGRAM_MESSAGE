import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from textblob import TextBlob

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
# Google News RSS 기반 감성 분석
# -----------------------------
def get_sentiment_score():
    try:
        url = "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"
        xml = requests.get(url).text
        soup = BeautifulSoup(xml, "xml")

        items = soup.find_all("item")[:10]
        headlines = [item.title.get_text(strip=True) for item in items]

        if not headlines:
            return 50, ["뉴스 없음"]

        polarity_sum = sum(TextBlob(h).sentiment.polarity for h in headlines)
        avg_polarity = polarity_sum / len(headlines)
        score_100 = int((avg_polarity + 1) * 50)

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
# 정교한 Proxy FGI (7개 요소 기반)
# -----------------------------
def compute_proxy_fgi():
    try:
        # 1) VIX
        vix = yf.Ticker("^VIX").history(period="10d")["Close"]
        vix_now = vix.iloc[-1]
        vix_change = (vix_now - vix.iloc[0]) / vix.iloc[0] * 100
        vix_score = max(0, min(100, 100 - vix_now * 3))

        # 2) Put/Call Ratio
        putcall = yf.Ticker("^PCR").history(period="1d")["Close"][0]
        putcall_score = max(0, min(100, 150 - putcall * 100))

        # 3) Junk Bond Demand
        junk = yf.Ticker("HYG").history(period="30d")["Close"]
        junk_now = junk.iloc[-1]
        junk_change = (junk_now - junk.iloc[0]) / junk.iloc[0] * 100
        junk_score = max(0, min(100, 50 + junk_change * 5))

        # 4) Safe Haven Demand (Gold vs S&P)
        gold = yf.Ticker("GC=F").history(period="1d")["Close"][0]
        sp = yf.Ticker("^GSPC").history(period="1d")["Close"][0]
        safe_ratio = gold / sp
        safe_score = max(0, min(100, 100 - safe_ratio * 100))

        # 5) Momentum (125일 모멘텀)
        sp125 = yf.Ticker("^GSPC").history(period="125d")["Close"]
        momentum = (sp125.iloc[-1] - sp125.iloc[0]) / sp125.iloc[0] * 100
        momentum_score = max(0, min(100, 50 + momentum))

        # 6) Breadth (상승 종목 비율)
        adv = yf.Ticker("^ADVN").history(period="1d")["Close"][0]
        dec = yf.Ticker("^DECL").history(period="1d")["Close"][0]
        breadth_ratio = adv / (adv + dec)
        breadth_score = int(breadth_ratio * 100)

        # 7) Volatility Change
        vol_score = max(0, min(100, 100 - abs(vix_change) * 2))

        # 최종 Proxy FGI
        proxy_fgi = int((vix_score + putcall_score + junk_score + safe_score +
                         momentum_score + breadth_score + vol_score) / 7)

        return proxy_fgi

    except:
        return 50  # 중립

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

    rsi, macd, signal, hist, bb_pos, bb_upper, bb_lower = compute_indicators(sp_hist["Close"])

    sentiment_score, headlines = get_sentiment_score()
    proxy_fgi = compute_proxy_fgi()
    fx_now, tnx_now, oil_now = get_macro_data()

    return {
        "sp_change": sp_change,
        "ndx_change": ndx_change,
        "vix_value": vix_value,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": hist,
        "bb_pos": bb_pos,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "sentiment_score": sentiment_score,
        "headlines": headlines,
        "proxy_fgi": proxy_fgi,
        "fx_now": fx_now,
        "tnx_now": tnx_now,
        "oil_now": oil_now,
    }

# -----------------------------
# 환율 / 금리 / 유가
# -----------------------------
def get_macro_data():
    try:
        fx = yf.Ticker("USDKRW=X").history(period="1d")["Close"][0]
        tnx = yf.Ticker("^TNX").history(period="1d")["Close"][0]
        oil = yf.Ticker("CL=F").history(period="1d")["Close"][0]
        return fx, tnx, oil
    except:
        return None, None, None

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
    proxy_fgi = data["proxy_fgi"]
    fx_now = data["fx_now"]
    tnx_now = data["tnx_now"]
    oil_now = data["oil_now"]

    # -----------------------------
    # 100점 만점 종합 점수
    # -----------------------------
    tech_score = 0
    if rsi >= 80: tech_score += 10
    if bb_pos >= 80: tech_score += 10
    if macd > macd_signal: tech_score += 10
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
        f"MACD: {macd:.4f} / Signal: {macd_signal:.4f} / Hist: {macd_hist:.4f}\n"
        f"볼린저 위치: {bb_pos:.1f}% (상단 {bb_upper:.2f}, 하단 {bb_lower:.2f})\n\n"

        f"뉴스 감성 점수: {sentiment_score}/100\n"
        f"Proxy FGI: {proxy_fgi}/100\n"
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
