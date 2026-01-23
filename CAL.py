import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import re
import json

# -----------------------------
# 설정
# -----------------------------
DEBUG = False
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")  # 안 써도 됨, 남겨둠

# -----------------------------
# 1. 텔레그램 전송
# -----------------------------
def send_telegram(message):
    bot_token = "8386665445:AAG5bEM30o9UzU-9NO9cGM7Lg0K7b1xcbFk"
    chat_id = "6983611450"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"텔레그램 전송 에러: {e}")

# -----------------------------
# 2. 날짜 계산 (D-Day)
# -----------------------------
def get_dday(target_date_str="2026-06-15"):
    today = datetime.now().date()
    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    diff = (target - today).days
    return diff

# -----------------------------
# 3. 기술적 지표 계산
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
    rsi_prev = float(rsi.iloc[-2])

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    macd_latest = float(macd.iloc[-1])
    macd_prev = float(macd.iloc[-2])
    signal_latest = float(signal.iloc[-1])
    signal_prev = float(signal.iloc[-2])
    hist_latest = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-2])

    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = float((ma20 + 2 * std20).iloc[-1])
    lower = float((ma20 - 2 * std20).iloc[-1])
    
    price = float(close.iloc[-1])
    price_prev = float(close.iloc[-2])

    bb_pos = (price - lower) / (upper - lower) * 100 if upper != lower else 50
    bb_pos_prev = (price_prev - float((ma20 - 2 * std20).iloc[-2])) / (float((ma20 + 2 * std20).iloc[-2]) - float((ma20 - 2 * std20).iloc[-2])) * 100 if upper != lower else 50

    # Stochastic Slow
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = (close - low14) / (high14 - low14) * 100
    stoch_d = stoch_k.rolling(3).mean()

    stoch_k_latest = float(stoch_k.iloc[-1])
    stoch_k_prev = float(stoch_k.iloc[-2])
    stoch_d_latest = float(stoch_d.iloc[-1])
    stoch_d_prev = float(stoch_d.iloc[-2])

    # CCI
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mean_dev = (tp - sma_tp).abs().rolling(20).mean()
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    cci_latest = float(cci.iloc[-1])
    cci_prev = float(cci.iloc[-2])

    # Williams %R
    highest14 = high.rolling(14).max()
    lowest14 = low.rolling(14).min()
    williams_r = -100 * (highest14 - close) / (highest14 - lowest14)
    williams_r_latest = float(williams_r.iloc[-1])
    williams_r_prev = float(williams_r.iloc[-2])

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_latest = float(atr.iloc[-1])
    atr_prev = float(atr.iloc[-2])
    atr_ratio_latest = atr_latest / price if price != 0 else 0
    atr_ratio_prev = atr_prev / price_prev if price_prev != 0 else 0

    # 20MA 괴리율
    ma20_latest = float(ma20.iloc[-1])
    ma20_prev = float(ma20.iloc[-2])
    ma_dev = (price - ma20_latest) / ma20_latest * 100 if ma20_latest != 0 else 0
    ma_dev_prev = (price_prev - ma20_prev) / ma20_prev * 100 if ma20_prev != 0 else 0

    return {
        "rsi": rsi_latest, "rsi_prev": rsi_prev,
        "macd": macd_latest, "macd_prev": macd_prev,
        "macd_signal": signal_latest, "macd_signal_prev": signal_prev,
        "macd_hist": hist_latest, "macd_hist_prev": hist_prev,
        "bb_pos": bb_pos, "bb_pos_prev": bb_pos_prev,
        "bb_upper": upper, "bb_lower": lower,
        "stoch_k": stoch_k_latest, "stoch_k_prev": stoch_k_prev,
        "stoch_d": stoch_d_latest, "stoch_d_prev": stoch_d_prev,
        "cci": cci_latest, "cci_prev": cci_prev,
        "williams_r": williams_r_latest, "williams_r_prev": williams_r_prev,
        "atr_ratio": atr_ratio_latest, "atr_ratio_prev": atr_ratio_prev,
        "ma_deviation_pct": ma_dev, "ma_deviation_pct_prev": ma_dev_prev,
        "price": price, "price_prev": price_prev
    }

def format_change(curr, prev, digits=2):
    try:
        delta = curr - prev
    except:
        return "변화 없음"
    if prev == 0:
        return f"{delta:+.{digits}f}"
    pct = delta / abs(prev) * 100
    return f"{delta:+.{digits}f} ({pct:+.{digits}f}%)"

# -----------------------------
# 4. Proxy FGI & Breadth (대체 지표 계산)
# -----------------------------
def compute_proxy_fgi(indicators, vix_value):
    score = 0
    rsi = indicators.get("rsi", 50)
    if rsi < 30: score += 15
    elif rsi < 45: score += 30
    elif rsi < 55: score += 50
    elif rsi < 70: score += 70
    else: score += 90

    if vix_value > 30: score += 10
    elif vix_value > 20: score += 30
    elif vix_value > 15: score += 50
    elif vix_value > 12: score += 75
    else: score += 90

    dev = indicators.get("ma_deviation_pct", 0)
    if dev < -5: score += 10
    elif dev < -1: score += 30
    elif dev < 1: score += 50
    elif dev < 5: score += 70
    else: score += 90

    final_proxy = int(score / 3)
    return max(0, min(100, final_proxy))

# -----------------------------
# 4-2. Breadth PRO MAX 세부 점수 함수
# -----------------------------
def score_index(sp_change, ndx_change):
    avg_idx = (sp_change + ndx_change) / 2
    avg_idx = max(-3, min(3, avg_idx))
    base = 50 + (avg_idx / 3) * 30
    return base

def score_vix_level(vix):
    if vix <= 12: return +15
    if vix <= 15: return +8
    if vix <= 18: return +3
    if vix <= 22: return 0
    if vix <= 28: return -10
    return -20

def score_vix_change(vix_change_pct):
    if vix_change_pct <= -10: return +12
    if vix_change_pct <= -5: return +6
    if vix_change_pct <= 0: return +2
    if vix_change_pct <= 5: return -4
    return -10

def score_volatility(atr_ratio):
    if atr_ratio <= 0.010: return +8
    if atr_ratio <= 0.015: return +4
    if atr_ratio <= 0.025: return 0
    if atr_ratio <= 0.035: return -6
    return -12

def score_combo(sp_change, ndx_change, vix, vix_change_pct):
    avg_idx = (sp_change + ndx_change) / 2
    if avg_idx > 0.5 and vix_change_pct > 3:
        return -8
    if -0.3 <= avg_idx <= 0.3 and vix_change_pct <= -8:
        return +10
    return 0

def compute_proxy_breadth_promax(sp_change, ndx_change, vix, vix_prev, atr_ratio):
    # 1) 지수 기반
    base = score_index(sp_change, ndx_change)

    # 2) VIX 절대 수준
    vix_level = score_vix_level(vix)

    # 3) VIX 변화율
    vix_change_pct = ((vix - vix_prev) / vix_prev) * 100 if vix_prev else 0
    vix_change = score_vix_change(vix_change_pct)

    # 4) ATR 변동성
    vol = score_volatility(atr_ratio)

    # 5) 조합 패턴
    combo = score_combo(sp_change, ndx_change, vix, vix_change_pct)

    # 합산
    raw = base + vix_level + vix_change + vol + combo

    # 0~100 클리핑
    final = max(5, min(95, int(raw)))
    return final


def blend_real_breadth(proxy_val, real_val):
    return int(proxy_val * 0.7 + real_val * 0.3)

# -----------------------------
# 5-1. FGI 안정화 버전
# -----------------------------
def fetch_fgi_stable(indicators, vix_value):
    urls = [
        "https://api.alternative.me/fng/?limit=1&format=json",
        "https://alternative.me/fng/?limit=1&format=json",
        "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }

    for url in urls:
        for attempt in range(3):
            try:
                res = requests.get(url, headers=headers, timeout=6)
                res.raise_for_status()
                data = res.json()

                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    v = data["data"][0].get("value")
                    if v is not None:
                        if DEBUG:
                            print(f"FGI from {url}: {v}")
                        return int(v), False

                if "fgi" in data and "now" in data["fgi"] and "value" in data["fgi"]["now"]:
                    v = data["fgi"]["now"]["value"]
                    if DEBUG:
                        print(f"FGI from RapidAPI: {v}")
                    return int(v), False

            except Exception as e:
                if DEBUG:
                    print(f"FGI fetch error ({url}, attempt {attempt+1}): {e}")
                time.sleep(attempt)

    if DEBUG:
        print("FGI all sources failed → using Proxy FGI")
    return compute_proxy_fgi(indicators, vix_value), True

# -----------------------------
# 5-2. Breadth PRO MAX (Proxy 메인 + 원본 덤)
# -----------------------------
def fetch_breadth_final(sp_change, ndx_change, vix, vix_prev, atr_ratio):
    proxy_val = compute_proxy_breadth_promax(sp_change, ndx_change, vix, vix_prev, atr_ratio)
    real_val = None
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        url = "https://stockcharts.com/h-sc/ui?s=$NYAD"
        res = requests.get(url, headers=headers, timeout=8)
        res.raise_for_status()
        html = res.text

        adv = re.search(r"Advances[^0-9]*([0-9,]+)", html)
        dec = re.search(r"Declines[^0-9]*([0-9,]+)", html)

        if adv and dec:
            adv_n = float(adv.group(1).replace(",", ""))
            dec_n = float(dec.group(1).replace(",", ""))
            if adv_n + dec_n > 0:
                real_val = int((adv_n / (adv_n + dec_n)) * 100)
                if DEBUG:
                    print(f"[Breadth] StockCharts real breadth={real_val}")
    except Exception as e:
        if DEBUG:
            print(f"[Breadth] StockCharts 실패(무시): {e}")

    if real_val is not None:
        blended = blend_real_breadth(proxy_val, real_val)
        if DEBUG:
            print(f"[Breadth] Proxy={proxy_val}, Real={real_val}, Blended={blended}")
        return blended, False

    if DEBUG:
        print(f"[Breadth] Real 없음 → Proxy만 사용: {proxy_val}")
    return proxy_val, True

# -----------------------------
# 5-3. FGI + Breadth 통합
# -----------------------------
def get_fgi_and_breadth(indicators, vix_value, vix_prev, sp_change, ndx_change):
    fgi_value, is_proxy_fgi = fetch_fgi_stable(indicators, vix_value)
    breadth_raw, is_proxy_breadth = fetch_breadth_final(
        sp_change=sp_change,
        ndx_change=ndx_change,
        vix=vix_value,
        vix_prev=vix_prev,
        atr_ratio=indicators.get("atr_ratio", 0)
    )
    return fgi_value, breadth_raw, is_proxy_fgi, is_proxy_breadth

# -----------------------------
# 6. 매크로 데이터 (환율/금리/유가)
# -----------------------------
def get_macro_data():
    try:
        fx_hist = yf.Ticker("USDKRW=X").history(period="5d")["Close"]
        tnx_hist = yf.Ticker("^TNX").history(period="5d")["Close"]
        oil_hist = yf.Ticker("CL=F").history(period="5d")["Close"]

        fx = float(fx_hist.mean()) if len(fx_hist) > 0 else None
        tnx = float(tnx_hist.mean()) if len(tnx_hist) > 0 else None
        oil = float(oil_hist.mean()) if len(oil_hist) > 0 else None

        return fx, tnx, oil
    except Exception:
        return None, None, None

def compute_macro_score(fx_now, tnx_now, oil_now):
    macro_score = 50

    if fx_now is not None:
        if fx_now < 1320: macro_score += 20
        elif fx_now < 1380: macro_score += 10
        elif fx_now < 1420: macro_score += 0
        elif fx_now < 1460: macro_score -= 10
        elif fx_now < 1500: macro_score -= 20
        else: macro_score -= 30

    if tnx_now is not None:
        if tnx_now < 3.5: macro_score += 20
        elif tnx_now < 4.0: macro_score += 10
        elif tnx_now < 4.3: macro_score += 0
        elif tnx_now < 4.6: macro_score -= 15
        elif tnx_now < 4.9: macro_score -= 25
        else: macro_score -= 35

    if oil_now is not None:
        if oil_now < 55: macro_score += 25
        elif oil_now < 65: macro_score += 15
        elif oil_now < 75: macro_score += 0
        elif oil_now < 85: macro_score -= 10
        elif oil_now < 95: macro_score -= 20
        else: macro_score -= 35

    return max(0, min(100, macro_score))

# -----------------------------
# 7. 변동성 안정성 점수
# -----------------------------
def compute_volatility_stability(vix_value, atr_ratio):
    if vix_value is None or atr_ratio is None:
        return 50
    score = 50
    if vix_value < 13: score += 30
    elif vix_value < 17: score += 10
    elif vix_value > 25: score -= 20

    if atr_ratio < 0.01: score += 10
    elif atr_ratio > 0.03: score -= 10

    return int(max(0, min(100, score)))

# -----------------------------
# 8. 포트폴리오 수익률 및 데이터 수집
# -----------------------------
def get_ticker_returns(tickers):
    returns = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="2d")["Close"]
            if len(hist) >= 2:
                today = float(hist.iloc[-1])
                prev = float(hist.iloc[-2])
                pct = (today - prev) / prev * 100 if prev != 0 else 0.0
                returns[t] = pct
            else:
                returns[t] = 0.0
        except Exception:
            returns[t] = 0.0
    return returns

def allocation_multiplier_from_return(pct):
    # 급락 구간
    if pct <= -6.0: return 1.60
    if pct <= -4.0: return 1.45
    if pct <= -2.0: return 1.30
    if pct <= -1.0: return 1.20
    if pct <= -0.3: return 1.10

    # 중립 구간
    if pct < 0.3: return 1.00

    # 상승 구간
    if pct < 1.0: return 0.90
    if pct < 2.0: return 0.80
    if pct < 4.0: return 0.65

    # 과열 구간
    return 0.50

def fetch_market_data():
    sp_all = yf.Ticker("^GSPC").history(period="252d")
    sp_hist = sp_all.iloc[-60:]
    ndx_all = yf.Ticker("^NDX").history(period="252d")
    vix_hist = yf.Ticker("^VIX").history(period="2d")

    sp_yesterday = sp_all.iloc[-2]
    sp_today = sp_all.iloc[-1]
    sp_change = float((sp_today["Close"] - sp_yesterday["Close"]) / sp_yesterday["Close"] * 100)

    ndx_yesterday = ndx_all.iloc[-2]
    ndx_today = ndx_all.iloc[-1]
    ndx_change = float((ndx_today["Close"] - ndx_yesterday["Close"]) / ndx_yesterday["Close"] * 100)

    vix_close = vix_hist["Close"]
    vix_value = float(vix_close.iloc[-1])
    vix_prev = float(vix_close.iloc[-2]) if len(vix_close) >= 2 else vix_value

    indicators = compute_indicators(sp_hist[["Open", "High", "Low", "Close"]])

    fgi_val, breadth_val, is_proxy_fgi, is_proxy_breadth = get_fgi_and_breadth(
        indicators, vix_value, vix_prev, sp_change, ndx_change
    )

    fx_now, tnx_now, oil_now = get_macro_data()

    high_52w = float(sp_all["High"].max()) if len(sp_all) > 0 else 0
    ma50 = float(sp_all["Close"].rolling(50).mean().iloc[-1])
    ma200 = float(sp_all["Close"].rolling(200).mean().iloc[-1]) if len(sp_all) >= 200 else None

    return {
        "sp_change": sp_change,
        "ndx_change": ndx_change,
        "vix_value": vix_value,
        "vix_prev": vix_prev,
        "high_52w": high_52w,
        "ma50": ma50,
        "ma200": ma200,
        **indicators,
        "real_fgi": fgi_val,
        "breadth_score": breadth_val,
        "is_proxy_fgi": is_proxy_fgi,
        "is_proxy_breadth": is_proxy_breadth,
        "fx_now": fx_now,
        "tnx_now": tnx_now,
        "oil_now": oil_now,
    }

# -----------------------------
# 9. 상세 코멘트 생성
# -----------------------------
def indicator_comments(data, high_52w, vix_value, vix_prev):
    comments = {}

    comments["vix_c"] = (
        "극저변동성" if vix_value <= 12 else
        "낮은 변동성" if vix_value <= 15 else
        "정상 변동성" if vix_value <= 20 else
        "변동성 증가" if vix_value <= 25 else
        "공포 구간"
    )
    comments["vix_change_c"] = format_change(vix_value, vix_prev)

    comments["macd_level_c"] = "상승 추세" if data.get("macd", 0) > 0 else "하락 추세"
    comments["macd_signal_c"] = "상승 모멘텀" if data.get("macd", 0) > data.get("macd_signal", 0) else "하락 모멘텀"
    comments["macd_hist_c"] = "모멘텀 강함" if abs(data.get("macd_hist", 0)) >= 5 else "모멘텀 약함"
    comments["macd_change_c"] = format_change(data.get("macd", 0), data.get("macd_prev", 0), 4)
    comments["macd_signal_change_c"] = format_change(data.get("macd_signal", 0), data.get("macd_signal_prev", 0), 4)
    comments["macd_hist_change_c"] = format_change(data.get("macd_hist", 0), data.get("macd_hist_prev", 0), 4)

    comments["rsi_c"] = "과열" if data.get("rsi", 50) >= 70 else "중립"
    comments["rsi_change_c"] = format_change(data.get("rsi", 50), data.get("rsi_prev", 50))

    comments["bb_c"] = "과열" if data.get("bb_pos", 50) >= 80 else "중립"
    comments["bb_change_c"] = format_change(data.get("bb_pos", 50), data.get("bb_pos_prev", 50))

    comments["stoch_c"] = "과열" if data.get("stoch_k", 50) >= 80 else "중립"
    comments["stoch_k_change_c"] = format_change(data.get("stoch_k", 50), data.get("stoch_k_prev", 50))
    comments["stoch_d_change_c"] = format_change(data.get("stoch_d", 50), data.get("stoch_d_prev", 50))

    comments["cci_c"] = "과열" if data.get("cci", 0) >= 100 else "중립"
    comments["cci_change_c"] = format_change(data.get("cci", 0), data.get("cci_prev", 0))

    comments["wr_c"] = "극과열" if data.get("williams_r", 0) >= -10 else "중립"
    comments["wr_change_c"] = format_change(data.get("williams_r", 0), data.get("williams_r_prev", 0))

    comments["atr_c"] = "변동성 낮음" if data.get("atr_ratio", 0) <= 0.015 else "변동성 높음"
    comments["atr_change_c"] = format_change(data.get("atr_ratio", 0), data.get("atr_ratio_prev", 0), 4)

    comments["ma_c"] = "과열" if data.get("ma_deviation_pct", 0) >= 5 else "중립"
    comments["ma_change_c"] = format_change(data.get("ma_deviation_pct", 0), data.get("ma_deviation_pct_prev", 0))

    if high_52w > 0:
        ratio = data.get("price", 0) / high_52w * 100
        ratio_prev = data.get("price_prev", 0) / high_52w * 100
        comments["high52_c"] = "고점 근접" if ratio >= 98 else "중립"
        comments["high52_change_c"] = format_change(ratio, ratio_prev)
    else:
        comments["high52_c"] = "데이터 없음"
        comments["high52_change_c"] = "변화 없음"

    return comments

# -----------------------------
# 10. 메인 실행
# -----------------------------
def main():
    print("데이터 수집 시작...")
    try:
        data = fetch_market_data()
    except Exception as e:
        print(f"데이터 수집 중 치명적 오류: {e}")
        send_telegram(f"❌ 봇 실행 실패: {e}")
        return

    dday = get_dday()

    sp_change = data.get("sp_change", 0)
    ndx_change = data.get("ndx_change", 0)
    vix_value = data.get("vix_value", 0)
    vix_prev = data.get("vix_prev", 0)
    high_52w = data.get("high_52w", 0)
    ma50 = data.get("ma50", 0)
    ma200 = data.get("ma200", None)

    fgi_val = data.get("real_fgi", 50)
    breadth_raw = data.get("breadth_score", 50)
    
    is_proxy_fgi = data.get("is_proxy_fgi", False)
    is_proxy_breadth = data.get("is_proxy_breadth", False)

    fx_now = data.get("fx_now", None)
    tnx_now = data.get("tnx_now", None)
    oil_now = data.get("oil_now", None)

    macro_score = compute_macro_score(fx_now, tnx_now, oil_now)

    if breadth_raw >= 70:
        breadth_score = 95
        breadth_label = "과열"
    elif breadth_raw >= 60:
        breadth_score = 80
        breadth_label = "강세"
    elif breadth_raw >= 40:
        breadth_score = 50
        breadth_label = "중립"
    elif breadth_raw >= 30:
        breadth_score = 30
        breadth_label = "약세"
    else:
        breadth_score = 10
        breadth_label = "위험"

    comments = indicator_comments(data, high_52w, vix_value, vix_prev)

    tech_score_raw = 0
    if data.get("rsi",0) >= 80: tech_score_raw += 10
    if data.get("bb_pos",0) >= 80: tech_score_raw += 10
    if data.get("macd",0) > data.get("macd_signal",0): tech_score_raw += 10
    if vix_value <= 15: tech_score_raw += 10
    if data.get("stoch_k",0) >= 80 and data.get("stoch_d",0) >= 80: tech_score_raw += 10
    if data.get("cci",0) >= 100: tech_score_raw += 10
    if data.get("williams_r",0) >= -20: tech_score_raw += 10
    if data.get("atr_ratio",0) <= 0.015: tech_score_raw += 10
    if data.get("ma_deviation_pct",0) >= 5: tech_score_raw += 10
    if high_52w > 0 and data.get("price",0) >= high_52w * 0.95: tech_score_raw += 10

    if data.get("price",0) > ma50: tech_score_raw += 5
    if ma200 is not None and data.get("price",0) > ma200: tech_score_raw += 5

    tech_score_raw = min(100, max(0, tech_score_raw))
    tech_score = tech_score_raw * 0.35

    vol_stability = compute_volatility_stability(vix_value, data.get("atr_ratio",0))

    final_score = int(
        tech_score +
        (fgi_val * 0.25) +
        (macro_score * 0.20) +
        (breadth_score * 0.10) +
        (vol_stability * 0.10)
    )

    if final_score >= 85:
        summary = "과열 구간에 근접 → 리스크 관리 최우선"
    elif final_score >= 70:
        summary = "상당한 과열 신호 → 매도/비중축소 고려"
    elif final_score >= 55:
        summary = "중립~살짝 과열 → 관망 또는 소량 조절"
    elif final_score >= 40:
        summary = "중립~저평가 구간 → 분할 매수 고려"
    else:
        summary = "공포·저평가 구간 → 공격적 매수 구간 후보"

    avg_change = (sp_change + ndx_change) / 2
    if final_score >= 90:
        result = "전량 매도"
        buy_amount = 0
    elif final_score >= 75:
        result = "분할 매도"
        buy_amount = 0
    elif final_score >= 50:
        result = "모으기"
        buy_amount = int(10000 + ((74 - final_score) / 74) * 25000)
        if avg_change > 0:
            buy_amount = 10000
    else:
        result = "모으기 (적극)"
        buy_amount = max(0, int(15000 + ((49 - final_score) / 74) * 30000))

    alert_lines = []
    if is_proxy_fgi:
        alert_lines.append("⚠️ FGI 원본 실패 → Proxy FGI(추정) 사용")
    if is_proxy_breadth:
        alert_lines.append("⚠️ Breadth 원본 실패 → Proxy Breadth(추정) 사용")
    alert_msg = "\n".join(alert_lines) + "\n\n" if alert_lines else ""

    portfolio = {
        "SOXL": 20, "TNA": 20, "TECL": 10, "ETHU": 10,
        "SOLT": 10, "INDL": 10, "FNGU": 10, "CURE": 10,
    }
    tickers = list(portfolio.keys())
    ticker_returns = get_ticker_returns(tickers)

    base_amounts = {t: buy_amount * w / 100 for t, w in portfolio.items()}
    adjusted_amounts = {}
    for t, base in base_amounts.items():
        pct = ticker_returns.get(t, 0.0)
        mult = allocation_multiplier_from_return(pct)
        adjusted_amounts[t] = base * mult

    total_adjusted = sum(adjusted_amounts.values()) if adjusted_amounts else 0
    scale = buy_amount / total_adjusted if (total_adjusted > 0 and buy_amount > 0) else 0.0

    portfolio_lines = []
    for t, adj in adjusted_amounts.items():
        final_amt = int(adj * scale)
        pct = ticker_returns.get(t, 0.0)
        mult = allocation_multiplier_from_return(pct)
        portfolio_lines.append(f"{t}: {final_amt:,}원 (today {pct:+.2f}%, mult {mult})")
    portfolio_text = "\n".join(portfolio_lines)

    bb_pos_display = f"{data.get('bb_pos', 50):.1f}"
    bb_upper = data.get('bb_upper', 0)
    bb_lower = data.get('bb_lower', 0)

    fgi_display_name = "Proxy FGI (추정)" if is_proxy_fgi else "Alternative.me FGI"

    telegram_message = f"""{alert_msg}📊 [정수 버블 체크 - {fgi_display_name}]

📌 요약
- {summary}

📈 지수 변동 (전일 종가 대비)
- S&P500: {sp_change:.2f}%
- NASDAQ: {ndx_change:.2f}%
- VIX: {vix_value:.2f}
  → {comments.get('vix_c', '-')}
  → 전일 대비 {comments.get('vix_change_c','-')}

🔍 기술적 지표 요약

🔸 MACD
- MACD / Signal / Hist: {data.get('macd',0):.4f} / {data.get('macd_signal',0):.4f} / {data.get('macd_hist',0):.4f}
- 해석: {comments.get('macd_level_c','-')} / {comments.get('macd_signal_c','-')} / {comments.get('macd_hist_c','-')}
- 변화:
  • MACD {comments.get('macd_change_c','-')}
  • Signal {comments.get('macd_signal_change_c','-')}
  • Hist {comments.get('macd_hist_change_c','-')}

🔸 RSI(14)
- {data.get('rsi',0):.2f} → {comments.get('rsi_c','-')}
- 변화: {comments.get('rsi_change_c','-')}

🔸 Bollinger Band
- 위치: {bb_pos_display}% (상단 {bb_upper:.2f}, 하단 {bb_lower:.2f})
- 해석: {comments.get('bb_c','-')}
- 변화: {comments.get('bb_change_c','-')}

🔸 Stochastic Slow
- %K / %D: {data.get('stoch_k',0):.2f} / {data.get('stoch_d',0):.2f}
- 해석: {comments.get('stoch_c','-')}
- 변화:
  • K {comments.get('stoch_k_change_c','-')}
  • D {comments.get('stoch_d_change_c','-')}

🔸 CCI(20)
- {data.get('cci',0):.2f} → {comments.get('cci_c','-')}
- 변화: {comments.get('cci_change_c','-')}

🔸 Williams %R
- {data.get('williams_r',0):.2f} → {comments.get('wr_c','-')}
- 변화: {comments.get('wr_change_c','-')}

🔸 ATR 비율
- {data.get('atr_ratio',0)*100:.2f}% → {comments.get('atr_c','-')}
- 변화: {comments.get('atr_change_c','-')}

🔸 20MA 괴리율
- {data.get('ma_deviation_pct',0):.2f}% → {comments.get('ma_c','-')}
- 변화: {comments.get('ma_change_c','-')}

🔸 52주 고점 대비
- { (data.get('price',0) / high_52w * 100) if high_52w>0 else 0:.2f}% → {comments.get('high52_c','-')}
- 변화: {comments.get('high52_change_c','-')}

🧮 점수 산출
- 기술 점수(35%): {tech_score_raw}/100
- {fgi_display_name}(25%): {fgi_val}/100 🔥
- 매크로 점수(20%): {macro_score}/100
- Breadth 점수(10%): {breadth_score}/100 ({breadth_label})
- 변동성 안정성(10%): {vol_stability}/100
- 총 점수: {final_score}/100

🧭 결론
- 75점↑ 매도 / 90점↑ 전량 매도
- 현재: {result}
- 매수 금액: {buy_amount:,}원

💼 포트폴리오 매수 금액
{portfolio_text}

📅 D-Day: 2026-06-15 (D-{dday})
"""

    send_telegram(telegram_message)
    print("텔레그램 전송 완료")
    if DEBUG:
        print(f"DEBUG: ProxyFGI={is_proxy_fgi}, FGI={fgi_val}, ProxyBreadth={is_proxy_breadth}, Final={final_score}")

if __name__ == "__main__":
    main()
