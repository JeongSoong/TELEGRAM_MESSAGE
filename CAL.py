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
# 기술적 지표 (RSI, MACD, 볼린저, Stoch, CCI, WilliamsR, ATR,乖離율, 52주 고점 등)
# -----------------------------
def compute_indicators(df: pd.DataFrame):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_latest = float(rsi.iloc[-1])

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    macd_latest = float(macd.iloc[-1])
    signal_latest = float(signal.iloc[-1])
    hist_latest = float(macd_hist.iloc[-1])

    # 볼린저밴드(20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = float((ma20 + 2 * std20).iloc[-1])
    lower = float((ma20 - 2 * std20).iloc[-1])
    price = float(close.iloc[-1])
    bb_pos = (price - lower) / (upper - lower) * 100 if upper != lower else 50

    # Stochastic Slow (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = (close - low14) / (high14 - low14) * 100
    stoch_d = stoch_k.rolling(3).mean()
    stoch_k_latest = float(stoch_k.iloc[-1])
    stoch_d_latest = float(stoch_d.iloc[-1])

    # CCI (20)
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mean_dev = (tp - sma_tp).abs().rolling(20).mean()
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    cci_latest = float(cci.iloc[-1])

    # Williams %R (14)
    highest14 = high.rolling(14).max()
    lowest14 = low.rolling(14).min()
    williams_r = -100 * (highest14 - close) / (highest14 - lowest14)
    williams_r_latest = float(williams_r.iloc[-1])

    # ATR(14) 및 ATR 비율
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_latest = float(atr.iloc[-1])
    atr_ratio_latest = atr_latest / price if price != 0 else 0

    # 20MA乖離율
    ma20_latest = float(ma20.iloc[-1])
    ma_deviation_pct = (price - ma20_latest) / ma20_latest * 100 if ma20_latest != 0 else 0

    return {
        "rsi": rsi_latest,
        "macd": macd_latest,
        "macd_signal": signal_latest,
        "macd_hist": hist_latest,
        "bb_pos": bb_pos,
        "bb_upper": upper,
        "bb_lower": lower,
        "stoch_k": stoch_k_latest,
        "stoch_d": stoch_d_latest,
        "cci": cci_latest,
        "williams_r": williams_r_latest,
        "atr_ratio": atr_ratio_latest,
        "ma_deviation_pct": ma_deviation_pct,
        "price": price,
    }

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

        proxy_fgi = int((vix_score + putcall_score + junk_score + safe_score +
                         momentum_score + breadth_score + vol_score) / 7)

        return proxy_fgi

    except:
        return 50  # 중립

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
# 시장 데이터 수집
# -----------------------------
def fetch_market_data():
    # 252일로 52주 고점 계산 가능하게
    sp_hist = yf.Ticker("^GSPC").history(period="252d")
    ndx_hist = yf.Ticker("^NDX").history(period="1d")
    vix_hist = yf.Ticker("^VIX").history(period="1d")

    # 일간 변동률 (가장 최근 날)
    sp_today = sp_hist.iloc[-1]
    sp_yday = sp_hist.iloc[-2]
    sp_change = float((sp_today["Close"] - sp_today["Open"]) / sp_today["Open"] * 100)

    ndx_change = float((ndx_hist["Close"][0] - ndx_hist["Open"][0]) / ndx_hist["Open"][0] * 100)
    vix_value = float(vix_hist["Close"][0])

    # 기술 지표 계산 (마지막 60일 정도만 사용해도 충분하지만 전체 close 사용)
    indicators = compute_indicators(sp_hist[["Open", "High", "Low", "Close"]])

    sentiment_score, headlines = get_sentiment_score()
    proxy_fgi = compute_proxy_fgi()
    fx_now, tnx_now, oil_now = get_macro_data()

    # 52주 고점
    high_52w = float(sp_hist["High"].max())

    return {
        "sp_change": sp_change,
        "ndx_change": ndx_change,
        "vix_value": vix_value,
        "high_52w": high_52w,
        **indicators,
        "sentiment_score": sentiment_score,
        "headlines": headlines,
        "proxy_fgi": proxy_fgi,
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
    high_52w = data["high_52w"]

    rsi = data["rsi"]
    macd = data["macd"]
    macd_signal = data["macd_signal"]
    macd_hist = data["macd_hist"]
    bb_pos = data["bb_pos"]
    bb_upper = data["bb_upper"]
    bb_lower = data["bb_lower"]
    stoch_k = data["stoch_k"]
    stoch_d = data["stoch_d"]
    cci = data["cci"]
    williams_r = data["williams_r"]
    atr_ratio = data["atr_ratio"]
    ma_deviation_pct = data["ma_deviation_pct"]
    price = data["price"]

    sentiment_score = data["sentiment_score"]
    headlines = data["headlines"]
    proxy_fgi = data["proxy_fgi"]
    fx_now = data["fx_now"]
    tnx_now = data["tnx_now"]
    oil_now = data["oil_now"]

    # -----------------------------
    # 기술 점수 (10개 지표 × 10점 = 100점)
    # -----------------------------
    tech_score_raw = 0

    # 1) RSI 과열
    if rsi >= 80:
        tech_score_raw += 10

    # 2) 볼린저 상단 근접
    if bb_pos >= 80:
        tech_score_raw += 10

    # 3) MACD > 시그널 (상승 추세)
    if macd > macd_signal:
        tech_score_raw += 10

    # 4) VIX 낮음 (안도/과열 구간)
    if vix_value <= 15:
        tech_score_raw += 10

    # 5) Stochastic Slow 과열
    if stoch_k >= 80 and stoch_d >= 80:
        tech_score_raw += 10

    # 6) CCI 과열
    if cci >= 100:
        tech_score_raw += 10

    # 7) Williams %R 과열
    if williams_r >= -20:
        tech_score_raw += 10

    # 8) ATR 변동성 낮음
    if atr_ratio <= 0.015:
        tech_score_raw += 10

    # 9) 20MA乖離율 (5% 이상 위)
    if ma_deviation_pct >= 5:
        tech_score_raw += 10

    # 10) 52주 고점 대비 95% 이상
    if high_52w > 0 and price >= high_52w * 0.95:
        tech_score_raw += 10

    # tech_score_raw (0~100)을 40% 비중으로 축소
    tech_score = tech_score_raw * 0.4

    # -----------------------------
    # 최종 100점 만점 종합 점수
    # -----------------------------
    news_score = sentiment_score * 0.3
    fgi_score = proxy_fgi * 0.3

    final_score = int(tech_score + news_score + fgi_score)

    # -----------------------------
    # 행동 결정 + 매수 금액 계산
    # -----------------------------
    avg_change = (sp_change + ndx_change) / 2

    if final_score >= 86:
        result = "전량 매도"
        buy_amount = 0

    elif final_score >= 70:
        result = "분할 매도"
        buy_amount = 0

    else:
        result = "모으기"
        # 0~39점 구간: 점수 낮을수록 많이 매수 (1만~3만)
        buy_amount = int(10000 + ((69 - final_score) / 69) * 20000)
        # 전날 S&P+나스닥 평균이 음수면 무조건 1만원
        if avg_change < 0:
            buy_amount = 10000

    # -----------------------------
    # 포트폴리오별 매수 금액 계산
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

    portfolio_lines = []
    for ticker, weight in portfolio.items():
        amount = int(buy_amount * weight / 100)
        portfolio_lines.append(f"{ticker}: {amount:,}원")

    portfolio_text = "\n".join(portfolio_lines)

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
        f"볼린저 위치: {bb_pos:.1f}% (상단 {bb_upper:.2f}, 하단 {bb_lower:.2f})\n"
        f"Stoch Slow %K/%D: {stoch_k:.2f} / {stoch_d:.2f}\n"
        f"CCI(20): {cci:.2f}\n"
        f"Williams %R: {williams_r:.2f}\n"
        f"ATR 비율: {atr_ratio*100:.2f}%\n"
        f"20MA乖離율: {ma_deviation_pct:.2f}%\n"
        f"52주 고점 대비: {price/high_52w*100:.2f}%\n\n" if high_52w > 0 else "\n"
        f"뉴스 감성 점수: {sentiment_score}/100\n"
        f"Proxy FGI: {proxy_fgi}/100\n"
        f"USD/KRW: {fx_now:,.2f}원\n"
        f"미국 10년물 금리: {tnx_now:.2f}%\n"
        f"WTI 유가: {oil_now:.2f}달러\n\n"
        f"기술 점수(원점수): {tech_score_raw}/100\n"
        f"총 점수: {final_score}/100\n"
        f"결론: {result}\n"
        f"매수 금액: {buy_amount:,}원\n\n"
        f"[포트폴리오 매수 금액]\n{portfolio_text}\n\n"
        f"[주요 뉴스]\n - " + "\n - ".join(headlines[:3])
    )

    send_telegram(telegram_message)
    print("텔레그램 전송 완료")

if __name__ == "__main__":
    main()
