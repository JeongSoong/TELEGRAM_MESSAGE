import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from datetime import datetime
import json
import time

# -----------------------------
# 1. 텔레그램 설정 (사용자님 설정값 유지)
# -----------------------------
def send_telegram(message):
    bot_token = "8386665445:AAG5bEM30o9UzU-9NO9cGM7Lg0K7b1xcbFk"
    chat_id = "6983611450"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
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
# 3. 기술적 지표 계산 (상세 로직 유지)
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
    
    # 데이터가 모자랄 경우를 대비해 예외처리 없이 그대로 진행 (원본 흐름)
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
    upper_prev = float((ma20 + 2 * std20).iloc[-2])
    lower_prev = float((ma20 - 2 * std20).iloc[-2])

    price = float(close.iloc[-1])
    price_prev = float(close.iloc[-2])

    # 밴드 위치 계산
    bb_pos = (price - lower) / (upper - lower) * 100 if upper != lower else 50
    bb_pos_prev = (price_prev - lower_prev) / (upper_prev - lower_prev) * 100 if upper_prev != lower_prev else 50

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
# 4. [수정됨] 진짜 CNN Fear & Greed Index + Breadth 계산
# -----------------------------
def get_real_cnn_fgi_and_breadth():
    # 1) CNN FGI 가져오기
    fgi_value = 50 # 기본값
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        url = "https://production.dataviz.cnn.io/index/fearandgreed/static/history"
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            fgi_value = int(data['market_rating_indicator']['rating_value'])
            print(f"CNN FGI 수집 성공: {fgi_value}")
        else:
            print(f"CNN FGI 수집 실패 (Status: {res.status_code})")
    except Exception as e:
        print(f"CNN FGI 에러: {e}")

    # 2) Market Breadth 계산 (원래 proxy 함수에 있던 기능 분리)
    breadth_raw = 50
    try:
        adv_hist = yf.Ticker("^ADVN").history(period="1d")["Close"]
        dec_hist = yf.Ticker("^DECL").history(period="1d")["Close"]
        
        if not adv_hist.empty and not dec_hist.empty:
            adv = float(adv_hist.iloc[-1])
            dec = float(dec_hist.iloc[-1])
            if (adv + dec) > 0:
                breadth_ratio = adv / (adv + dec)
                breadth_raw = int(breadth_ratio * 100)
    except Exception as e:
        print(f"Breadth 계산 에러: {e}")

    return fgi_value, breadth_raw

# -----------------------------
# 5. 매크로 데이터 (환율/금리/유가)
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
    macro_score = 50  # 기본값

    # 환율
    if fx_now is not None:
        if fx_now < 1320: macro_score += 20
        elif fx_now < 1380: macro_score += 10
        elif fx_now < 1420: macro_score += 0
        elif fx_now < 1460: macro_score -= 10
        elif fx_now < 1500: macro_score -= 20
        else: macro_score -= 30

    # 금리
    if tnx_now is not None:
        if tnx_now < 3.5: macro_score += 20
        elif tnx_now < 4.0: macro_score += 10
        elif tnx_now < 4.3: macro_score += 0
        elif tnx_now < 4.6: macro_score -= 15
        elif tnx_now < 4.9: macro_score -= 25
        else: macro_score -= 35

    # 유가
    if oil_now is not None:
        if oil_now < 55: macro_score += 25
        elif oil_now < 65: macro_score += 15
        elif oil_now < 75: macro_score += 0
        elif oil_now < 85: macro_score -= 10
        elif oil_now < 95: macro_score -= 20
        else: macro_score -= 35

    return max(0, min(100, macro_score))

# -----------------------------
# 6. 변동성 안정성 점수
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
# 7. 포트폴리오 로직 (수익률 기반 배수)
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
    if pct <= -3.0: return 1.30
    if pct <= -1.0: return 1.20
    if pct < 0.0: return 1.10
    if pct < 2.0: return 1.00
    if pct < 5.0: return 0.80
    return 0.50

# -----------------------------
# 8. 데이터 수집 통합 함수
# -----------------------------
def fetch_market_data():
    sp_all = yf.Ticker("^GSPC").history(period="252d")
    sp_hist = sp_all.iloc[-60:]
    ndx_all = yf.Ticker("^NDX").history(period="252d")
    vix_hist = yf.Ticker("^VIX").history(period="2d")

    # 변동률 계산
    sp_yesterday = sp_all.iloc[-2]
    sp_today = sp_all.iloc[-1]
    sp_change = float((sp_today["Close"] - sp_yesterday["Close"]) / sp_yesterday["Close"] * 100)

    ndx_yesterday = ndx_all.iloc[-2]
    ndx_today = ndx_all.iloc[-1]
    ndx_change = float((ndx_today["Close"] - ndx_yesterday["Close"]) / ndx_yesterday["Close"] * 100)

    vix_close = vix_hist["Close"]
    vix_value = float(vix_close.iloc[-1])
    vix_prev = float(vix_close.iloc[-2]) if len(vix_close) >= 2 else vix_value

    # 지표 계산
    indicators = compute_indicators(sp_hist[["Open", "High", "Low", "Close"]])

    # [중요] Proxy 대신 진짜 CNN FGI + Breadth 사용
    cnn_fgi, breadth_raw = get_real_cnn_fgi_and_breadth()
    
    # 매크로
    fx_now, tnx_now, oil_now = get_macro_data()

    # 52주 고점 등
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
        "real_fgi": cnn_fgi,         # 이름 변경: proxy_fgi -> real_fgi
        "breadth_score": breadth_raw,
        "fx_now": fx_now,
        "tnx_now": tnx_now,
        "oil_now": oil_now,
    }

# -----------------------------
# 9. 상세 코멘트 생성 (사용자 원본 로직 유지)
# -----------------------------
def indicator_comments(data, high_52w, vix_value, vix_prev):
    comments = {}

    # VIX
    comments["vix_c"] = (
        "극저변동성" if vix_value <= 12 else
        "낮은 변동성" if vix_value <= 15 else
        "정상 변동성" if vix_value <= 20 else
        "변동성 증가" if vix_value <= 25 else
        "공포 구간"
    )
    comments["vix_change_c"] = format_change(vix_value, vix_prev)

    # MACD
    comments["macd_level_c"] = "상승 추세" if data["macd"] > 0 else "하락 추세"
    comments["macd_signal_c"] = "상승 모멘텀" if data["macd"] > data["macd_signal"] else "하락 모멘텀"
    comments["macd_hist_c"] = "모멘텀 강함" if abs(data["macd_hist"]) >= 5 else "모멘텀 약함"
    comments["macd_change_c"] = format_change(data["macd"], data["macd_prev"], 4)
    comments["macd_signal_change_c"] = format_change(data["macd_signal"], data["macd_signal_prev"], 4)
    comments["macd_hist_change_c"] = format_change(data["macd_hist"], data["macd_hist_prev"], 4)

    # RSI
    comments["rsi_c"] = "과열" if data["rsi"] >= 70 else "중립"
    comments["rsi_change_c"] = format_change(data["rsi"], data["rsi_prev"])

    # Bollinger Band
    comments["bb_c"] = "과열" if data["bb_pos"] >= 80 else "중립"
    comments["bb_change_c"] = format_change(data["bb_pos"], data["bb_pos_prev"])

    # Stochastic
    comments["stoch_c"] = "과열" if data["stoch_k"] >= 80 else "중립"
    comments["stoch_k_change_c"] = format_change(data["stoch_k"], data["stoch_k_prev"])
    comments["stoch_d_change_c"] = format_change(data["stoch_d"], data["stoch_d_prev"])

    # CCI
    comments["cci_c"] = "과열" if data["cci"] >= 100 else "중립"
    comments["cci_change_c"] = format_change(data["cci"], data["cci_prev"])

    # Williams %R
    comments["wr_c"] = "극과열" if data["williams_r"] >= -10 else "중립"
    comments["wr_change_c"] = format_change(data["williams_r"], data["williams_r_prev"])

    # ATR
    comments["atr_c"] = "변동성 낮음" if data["atr_ratio"] <= 0.015 else "변동성 높음"
    comments["atr_change_c"] = format_change(data["atr_ratio"], data["atr_ratio_prev"], 4)

    # MA Deviation
    comments["ma_c"] = "과열" if data["ma_deviation_pct"] >= 5 else "중립"
    comments["ma_change_c"] = format_change(data["ma_deviation_pct"], data["ma_deviation_pct_prev"])

    # 52w High
    if high_52w > 0:
        ratio = data["price"] / high_52w * 100
        ratio_prev = data["price_prev"] / high_52w * 100
        comments["high52_c"] = "고점 근접" if ratio >= 98 else "중립"
        comments["high52_change_c"] = format_change(ratio, ratio_prev)
    else:
        comments["high52_c"] = "데이터 없음"
        comments["high52_change_c"] = "변화 없음"

    return comments

# -----------------------------
# 10. 메인 실행 함수
# -----------------------------
def main():
    print("데이터 수집 시작...")
    data = fetch_market_data()
    dday = get_dday()

    sp_change = data["sp_change"]
    ndx_change = data["ndx_change"]
    vix_value = data["vix_value"]
    vix_prev = data["vix_prev"]
    high_52w = data["high_52w"]
    ma50 = data["ma50"]
    ma200 = data["ma200"]

    # [중요] 진짜 CNN FGI 값 사용
    real_fgi = data["real_fgi"]
    breadth_raw = data["breadth_score"]
    
    fx_now = data["fx_now"]
    tnx_now = data["tnx_now"]
    oil_now = data["oil_now"]

    # Macro score
    macro_score = compute_macro_score(fx_now, tnx_now, oil_now)

    # Breadth Score Logic
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

    # 코멘트 생성
    comments = indicator_comments(data, high_52w, vix_value, vix_prev)

    # 기술 점수 (Tech Score) 계산
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

    # 추세 점수 추가
    if data["price"] > ma50: tech_score_raw += 5
    if ma200 is not None and data["price"] > ma200: tech_score_raw += 5

    tech_score_raw = min(100, max(0, tech_score_raw))
    tech_score = tech_score_raw * 0.4  # 가중치 40%

    # 변동성 안정성
    vol_stability = compute_volatility_stability(vix_value, data["atr_ratio"])

    # -----------------------------
    # [최종 점수] CNN FGI (30%) 적용
    # -----------------------------
    final_score = int(
        tech_score +
        (real_fgi * 0.30) +     # 기존 proxy_fgi 대신 real_fgi 사용
        (macro_score * 0.15) +
        (breadth_score * 0.10) +
        (vol_stability * 0.05)
    )

    # 요약 문구
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

    # 매수/매도 행동 결정
    avg_change = (sp_change + ndx_change) / 2

    if final_score >= 90:
        result = "전량 매도"
        buy_amount = 0
    elif final_score >= 75:
        result = "분할 매도"
        buy_amount = 0
    elif final_score >= 50:
        result = "모으기"
        buy_amount = int(10000 + ((74 - final_score) / 74) * 20000)
        if avg_change > 0:
            buy_amount = 10000
    else:
        result = "모으기 (적극)"
        buy_amount = int(10000 + ((49 - final_score) / 74) * 25000)

    # -----------------------------
    # 포트폴리오 배분
    # -----------------------------
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
    scale = buy_amount / total_adjusted if (total_adjusted > 0 and buy_amount > 0) else 1.0

    portfolio_lines = []
    for t, adj in adjusted_amounts.items():
        final_amt = int(adj * scale)
        pct = ticker_returns.get(t, 0.0)
        mult = allocation_multiplier_from_return(pct)
        portfolio_lines.append(f"{t}: {final_amt:,}원 (today {pct:+.2f}%, mult {mult})")

    portfolio_text = "\n".join(portfolio_lines)

    # 52주 고점 텍스트
    if high_52w > 0:
        ratio_now = data["price"] / high_52w * 100
        high52_line = (
            f"- {ratio_now:.2f}% → {comments['high52_c']}\n"
            f"- 변화: {comments['high52_change_c']}\n"
        )
    else:
        high52_line = "- 데이터 없음\n"

    # -----------------------------
    # 텔레그램 메시지 생성 (상세 버전)
    # -----------------------------
    telegram_message = f"""
📊 [정수 버블 체크 - Real CNN FGI]

📌 요약
- {summary}

📈 지수 변동 (전일 종가 대비)
- S&P500: {sp_change:.2f}%
- NASDAQ: {ndx_change:.2f}%
- VIX: {vix_value:.2f}
  → {comments['vix_c']}
  → 전일 대비 {comments['vix_change_c']}

🔍 기술적 지표 요약

🔸 MACD
- MACD / Signal / Hist: {data['macd']:.4f} / {data['macd_signal']:.4f} / {data['macd_hist']:.4f}
- 해석: {comments['macd_level_c']} / {comments['macd_signal_c']} / {comments['macd_hist_c']}
- 변화:
  • MACD {comments['macd_change_c']}
  • Signal {comments['macd_signal_change_c']}
  • Hist {comments['macd_hist_change_c']}

🔸 RSI(14)
- {data['rsi']:.2f} → {comments['rsi_c']}
- 변화: {comments['rsi_change_c']}

🔸 Bollinger Band
- 위치: {data['bb_pos']:.1f}% (상단 {data['bb_upper']:.2f}, 하단 {data['bb_lower']:.2f})
- 해석: {comments['bb_c']}
- 변화: {comments['bb_change_c']}

🔸 Stochastic Slow
- %K / %D: {data['stoch_k']:.2f} / {data['stoch_d']:.2f}
- 해석: {comments['stoch_c']}
- 변화:
  • K {comments['stoch_k_change_c']}
  • D {comments['stoch_d_change_c']}

🔸 CCI(20)
- {data['cci']:.2f} → {comments['cci_c']}
- 변화: {comments['cci_change_c']}

🔸 Williams %R
- {data['williams_r']:.2f} → {comments['wr_c']}
- 변화: {comments['wr_change_c']}

🔸 ATR 비율
- {data['atr_ratio']*100:.2f}% → {comments['atr_c']}
- 변화: {comments['atr_change_c']}

🔸 20MA 괴리율
- {data['ma_deviation_pct']:.2f}% → {comments['ma_c']}
- 변화: {comments['ma_change_c']}

🔸 52주 고점 대비
{high52_line}

🧮 점수 산출
- 기술 점수(40%): {tech_score_raw}/100
- Real CNN FGI(30%): {real_fgi}/100 🔥
- 매크로 점수(15%): {macro_score}/100
- Breadth 점수(10%): {breadth_score}/100 ({breadth_label})
- 변동성 안정성(5%): {vol_stability}/100
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
    print(f"최종 점수: {final_score}, FGI: {real_fgi}")

if __name__ == "__main__":
    main()
