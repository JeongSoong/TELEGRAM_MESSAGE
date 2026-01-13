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
# 기술적 지표 계산
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

    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_latest = float(atr.iloc[-1])
    atr_ratio_latest = atr_latest / price if price != 0 else 0

    # 20MA 괴리율
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
# 지표별 코멘트 생성
# -----------------------------
def indicator_comments(data, high_52w):

    rsi = data["rsi"]
    bb_pos = data["bb_pos"]
    stoch_k = data["stoch_k"]
    stoch_d = data["stoch_d"]
    cci = data["cci"]
    williams_r = data["williams_r"]
    atr_ratio = data["atr_ratio"]
    ma_dev = data["ma_deviation_pct"]
    price = data["price"]

    # RSI
    if rsi >= 80: rsi_c = "과열 (매우 높음)"
    elif rsi >= 70: rsi_c = "과열 신호"
    elif rsi >= 50: rsi_c = "중립"
    else: rsi_c = "저평가"

    # Bollinger
    if bb_pos >= 90: bb_c = "상단 돌파 (강한 과열)"
    elif bb_pos >= 80: bb_c = "상단 근접 (과열)"
    elif bb_pos >= 50: bb_c = "중립"
    else: bb_c = "하단 근접 (저평가)"

    # Stochastic
    if stoch_k >= 90 and stoch_d >= 90: stoch_c = "극과열"
    elif stoch_k >= 80 and stoch_d >= 80: stoch_c = "과열"
    elif stoch_k >= 50: stoch_c = "중립"
    else: stoch_c = "저평가"

    # CCI
    if cci >= 100: cci_c = "과열"
    elif cci <= -100: cci_c = "저평가"
    else: cci_c = "중립"

    # Williams %R
    if williams_r >= -10: wr_c = "극과열"
    elif williams_r >= -20: wr_c = "과열"
    elif williams_r >= -80: wr_c = "중립"
    else: wr_c = "저평가"

    # ATR
    if atr_ratio <= 0.01: atr_c = "변동성 매우 낮음 (과열 패턴)"
    elif atr_ratio <= 0.02: atr_c = "변동성 낮음"
    else: atr_c = "변동성 높음"

    # MA Deviation
    if ma_dev >= 5: ma_c = "이평선 대비 과열"
    elif ma_dev >= 2: ma_c = "상승 추세"
    else: ma_c = "중립"

    # 52주 고점 대비
    if high_52w > 0:
        ratio = price / high_52w * 100
        if ratio >= 98: high52_c = "52주 고점 근접 (과열)"
        elif ratio >= 90: high52_c = "고점권"
        else: high52_c = "중립"
    else:
        high52_c = "데이터 없음"

    return {
        "rsi_c": rsi_c,
        "bb_c": bb_c,
        "stoch_c": stoch_c,
        "cci_c": cci_c,
        "wr_c": wr_c,
        "atr_c": atr_c,
        "ma_c": ma_c,
        "high52_c": high52_c
    }

# -----------------------------
# Proxy FGI
# -----------------------------
def compute_proxy_fgi():
    try:
        vix = yf.Ticker("^VIX").history(period="10d")["Close"]
        if len(vix) < 2:
            return 50
        vix_now = float(vix.iloc[-1])
        vix_change = (vix_now - float(vix.iloc[0])) / float(vix.iloc[0]) * 100
        vix_score = max(0, min(100, 100 - vix_now * 3))

        junk = yf.Ticker("HYG").history(period="30d")["Close"]
        if len(junk) < 2:
            return 50
        junk_now = float(junk.iloc[-1])
        junk_change = (junk_now - float(junk.iloc[0])) / float(junk.iloc[0]) * 100
        junk_score = max(0, min(100, 50 + junk_change * 5))

        gold_hist = yf.Ticker("GC=F").history(period="1d")["Close"]
        sp_hist = yf.Ticker("^GSPC").history(period="1d")["Close"]
        if len(gold_hist) == 0 or len(sp_hist) == 0:
            return 50
        gold = float(gold_hist.iloc[-1])
        sp = float(sp_hist.iloc[-1])
        safe_ratio = gold / sp if sp != 0 else 1
        safe_score = max(0, min(100, 100 - safe_ratio * 100))

        sp125 = yf.Ticker("^GSPC").history(period="125d")["Close"]
        if len(sp125) < 2:
            return 50
        momentum = (float(sp125.iloc[-1]) - float(sp125.iloc[0])) / float(sp125.iloc[0]) * 100
        momentum_score = max(0, min(100, 50 + momentum))

        adv_hist = yf.Ticker("^ADVN").history(period="1d")["Close"]
        dec_hist = yf.Ticker("^DECL").history(period="1d")["Close"]
        if len(adv_hist) == 0 or len(dec_hist) == 0:
            breadth_score = 50
        else:
            adv = float(adv_hist.iloc[-1])
            dec = float(dec_hist.iloc[-1])
            breadth_ratio = adv / (adv + dec) if (adv + dec) != 0 else 0.5
            breadth_score = int(breadth_ratio * 100)

        vol_score = max(0, min(100, 100 - abs(vix_change) * 2))

        proxy_fgi = int((vix_score + junk_score + safe_score +
                         momentum_score + breadth_score + vol_score) / 6)

        return proxy_fgi

    except:
        return 50

# -----------------------------
# 환율 / 금리 / 유가
# -----------------------------
def get_macro_data():
    try:
        fx_hist = yf.Ticker("USDKRW=X").history(period="1d")["Close"]
        tnx_hist = yf.Ticker("^TNX").history(period="1d")["Close"]
        oil_hist = yf.Ticker("CL=F").history(period="1d")["Close"]

        fx = float(fx_hist.iloc[-1]) if len(fx_hist) > 0 else None
        tnx = float(tnx_hist.iloc[-1]) if len(tnx_hist) > 0 else None
        oil = float(oil_hist.iloc[-1]) if len(oil_hist) > 0 else None

        return fx, tnx, oil
    except:
        return None, None, None

# -----------------------------
# 시장 데이터 수집
# -----------------------------
def fetch_market_data():
    sp_all = yf.Ticker("^GSPC").history(period="252d")
    sp_hist = sp_all.iloc[-60:]
    ndx_hist = yf.Ticker("^NDX").history(period="1d")
    vix_hist = yf.Ticker("^VIX").history(period="1d")

    sp_today = sp_all.iloc[-1]
    sp_change = float((sp_today["Close"] - sp_today["Open"]) / sp_today["Open"] * 100)

    ndx_close = ndx_hist["Close"]
    ndx_open = ndx_hist["Open"]
    ndx_change = float((float(ndx_close.iloc[-1]) - float(ndx_open.iloc[-1])) / float(ndx_open.iloc[-1]) * 100)

    vix_close = vix_hist["Close"]
    vix_value = float(vix_close.iloc[-1])

    indicators = compute_indicators(sp_hist[["Open", "High", "Low", "Close"]])

    sentiment_score, headlines = get_sentiment_score()
    proxy_fgi = compute_proxy_fgi()
    fx_now, tnx_now, oil_now = get_macro_data()

    high_52w = float(sp_all["High"].max()) if len(sp_all) > 0 else 0

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

    sentiment_score = data["sentiment_score"]
    headlines = data["headlines"]
    proxy_fgi = data["proxy_fgi"]
    fx_now = data["fx_now"]
    tnx_now = data["tnx_now"]
    oil_now = data["oil_now"]

    # 기술 지표 코멘트 생성
    comments = indicator_comments(data, high_52w)

    # 기술 점수 계산
    tech_score_raw = 0
    if data["rsi"] >= 80: tech_score_raw += 10
    if data["bb_pos"] >= 80: tech_score_raw += 10
    if data["macd"] > data["macd_signal"]: tech_score_raw += 10
    if vix_value <= 15: tech_score_raw += 10
    if data["stoch_k"] >= 80 and data["stoch_d"] >= 80: tech_score_raw += 10
    if data["cci"] >= 100: tech_score_raw += 10
    if data["williams_r"] >= -20: tech_score_raw += 10
    if data["atr_ratio"] <= 0.015: tech_score_raw += 10
    if data["ma_deviation_pct"] >= 5: tech_score_raw += 10
    if high_52w > 0 and data["price"] >= high_52w * 0.95: tech_score_raw += 10

    tech_score = tech_score_raw * 0.4

    # 최종 점수
    final_score = int(tech_score + sentiment_score * 0.3 + proxy_fgi * 0.3)

    # 행동 결정
    avg_change = (sp_change + ndx_change) / 2

    if final_score >= 90:
        result = "전량 매도"
        buy_amount = 0
    elif final_score >= 75:
        result = "분할 매도"
        buy_amount = 0
    else:
        result = "모으기"
        buy_amount = int(10000 + ((74 - final_score) / 74) * 20000)
        if avg_change > 0:
            buy_amount = 10000

    # 포트폴리오 배분
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

    # 52주 고점 문구
    if high_52w > 0:
        high_52w_line = f"52주 고점 대비: {data['price'] / high_52w * 100:.2f}% → {comments['high52_c']}\n"
    else:
        high_52w_line = ""

    # 텔레그램 메시지
    telegram_message =
