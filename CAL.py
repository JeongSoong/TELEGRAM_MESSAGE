"""
레버리지 ETF 분할매수 시그널 봇 v3.0 (ULTRA)
=============================================
v3.0 주요 업그레이드:
1. 종목별 개별 지표 분석 (RSI, BB, 모멘텀)
2. 듀얼 모멘텀 전략 (절대 + 상대 모멘텀)
3. 섹터 강도 분석 (섹터 ETF 기반)
4. 종목별 매수/매도 개별 판단
5. 섹터 로테이션 반영
"""

import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import math
from typing import Dict, Tuple, Optional, List

# =============================================================================
# 설정
# =============================================================================
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# 텔레그램 설정
TELEGRAM_BOT_TOKEN = "8386665445:AAG5bEM30o9UzU-9NO9cGM7Lg0K7b1xcbFk"
TELEGRAM_CHAT_ID = "6983611450"

# 투자 설정 (레버리지 ETF + 월 30~50만원 추가 입금 기준)
# 월 추가입금 감안하여 좀 더 공격적으로!
BASE_DAILY_BUDGET = 15000   # 기본 일일 매수 예산
MAX_DAILY_BUDGET = 70000    # 최대 일일 매수 예산 (급락 시 공격적)
MIN_DAILY_BUDGET = 0        # 최소 - 기회 아니면 안 삼

# 총 투자금 설정
TOTAL_CAPITAL = 1000000     # 초기 투자금 100만원
MONTHLY_ADDITION = 400000   # 월 평균 추가 입금 40만원
CASH_RESERVE_RATIO = 0.25   # 최소 현금 보유 비율 25%

# D-Day 설정
TARGET_DATE = "2026-06-15"

# =============================================================================
# 포트폴리오 설정 (섹터 ETF 매핑 추가)
# =============================================================================
PORTFOLIO = {
    "SOXL": {
        "weight": 20, 
        "sector": "semiconductor",
        "leverage": 3, 
        "desc": "반도체 3x",
        "underlying": "SOXX",      # 기초 ETF (1x)
        "sector_etf": "XLK",       # 섹터 ETF
    },
    "TECL": {
        "weight": 10, 
        "sector": "tech",
        "leverage": 3, 
        "desc": "기술 3x",
        "underlying": "XLK",
        "sector_etf": "XLK",
    },
    "FNGU": {
        "weight": 10, 
        "sector": "tech",
        "leverage": 3, 
        "desc": "FANG+ 3x",
        "underlying": "QQQ",
        "sector_etf": "XLK",
    },
    "TNA": {
        "weight": 10, 
        "sector": "smallcap",
        "leverage": 3, 
        "desc": "소형주 3x",
        "underlying": "IWM",
        "sector_etf": "IWM",
    },
    "CURE": {
        "weight": 10, 
        "sector": "healthcare",
        "leverage": 3, 
        "desc": "헬스케어 3x",
        "underlying": "XLV",
        "sector_etf": "XLV",
    },
    "INDL": {
        "weight": 10, 
        "sector": "india",
        "leverage": 3, 
        "desc": "인도 3x",
        "underlying": "INDA",
        "sector_etf": "EEM",
    },
    "ETHU": {
        "weight": 10, 
        "sector": "crypto",
        "leverage": 2, 
        "desc": "이더리움 2x",
        "underlying": "ETH-USD",
        "sector_etf": "CRYPTO",
    },
    "SOLT": {
        "weight": 10, 
        "sector": "crypto",
        "leverage": 2, 
        "desc": "솔라나 2x",
        "underlying": "SOL-USD",
        "sector_etf": "CRYPTO",
    },
}

# 섹터 ETF 목록 (강도 분석용)
SECTOR_ETFS = {
    "XLK": "기술",
    "XLV": "헬스케어",
    "XLF": "금융",
    "XLE": "에너지",
    "XLI": "산업재",
    "XLY": "경기소비재",
    "XLP": "필수소비재",
    "XLU": "유틸리티",
    "XLRE": "부동산",
    "XLB": "소재",
    "XLC": "통신",
    "SOXX": "반도체",
    "IWM": "소형주",
    "EEM": "신흥국",
}

# =============================================================================
# 유틸리티 함수
# =============================================================================
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default

def get_dday(target_date_str: str = TARGET_DATE) -> int:
    today = datetime.now().date()
    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    return (target - today).days

def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, data=payload, timeout=10)
        return res.status_code == 200
    except Exception as e:
        print(f"텔레그램 에러: {e}")
        return False

# =============================================================================
# 1. 종목별 기술적 지표 계산
# =============================================================================
def compute_ticker_indicators(ticker: str, period: str = "120d") -> Optional[Dict]:
    """
    개별 종목의 기술적 지표 계산
    """
    try:
        df = yf.Ticker(ticker).history(period=period)
        if len(df) < 50:
            return None
        
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        
        result = {"ticker": ticker}
        
        # 가격 정보
        result["price"] = float(close.iloc[-1])
        result["price_prev"] = float(close.iloc[-2])
        result["daily_return"] = safe_div(result["price"] - result["price_prev"], result["price_prev"]) * 100
        
        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        result["rsi"] = float(rsi.iloc[-1])
        
        # RSI 상태 판단
        if result["rsi"] >= 80:
            result["rsi_status"] = "극과열"
            result["rsi_score"] = 10
        elif result["rsi"] >= 70:
            result["rsi_status"] = "과열"
            result["rsi_score"] = 25
        elif result["rsi"] >= 60:
            result["rsi_status"] = "강세"
            result["rsi_score"] = 45
        elif result["rsi"] >= 40:
            result["rsi_status"] = "중립"
            result["rsi_score"] = 50
        elif result["rsi"] >= 30:
            result["rsi_status"] = "약세"
            result["rsi_score"] = 65
        elif result["rsi"] >= 20:
            result["rsi_status"] = "과매도"
            result["rsi_score"] = 80
        else:
            result["rsi_status"] = "극과매도"
            result["rsi_score"] = 95
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        
        bb_width = float(upper.iloc[-1]) - float(lower.iloc[-1])
        result["bb_position"] = safe_div(result["price"] - float(lower.iloc[-1]), bb_width, 0.5) * 100
        
        # BB 상태 판단
        if result["bb_position"] >= 95:
            result["bb_status"] = "상단돌파"
            result["bb_score"] = 10
        elif result["bb_position"] >= 80:
            result["bb_status"] = "상단근접"
            result["bb_score"] = 25
        elif result["bb_position"] >= 60:
            result["bb_status"] = "상단"
            result["bb_score"] = 40
        elif result["bb_position"] >= 40:
            result["bb_status"] = "중립"
            result["bb_score"] = 50
        elif result["bb_position"] >= 20:
            result["bb_status"] = "하단"
            result["bb_score"] = 65
        elif result["bb_position"] >= 5:
            result["bb_status"] = "하단근접"
            result["bb_score"] = 80
        else:
            result["bb_status"] = "하단돌파"
            result["bb_score"] = 95
        
        # 이동평균
        result["ma20"] = float(ma20.iloc[-1])
        result["ma50"] = float(close.rolling(50).mean().iloc[-1])
        
        if len(close) >= 200:
            result["ma200"] = float(close.rolling(200).mean().iloc[-1])
        else:
            result["ma200"] = None
        
        # 추세 상태
        if result["ma200"]:
            above_200ma = result["price"] > result["ma200"]
            golden_cross = result["ma50"] > result["ma200"]
            
            if above_200ma and golden_cross:
                result["trend"] = "강한상승"
                result["trend_score"] = 80
            elif above_200ma:
                result["trend"] = "상승"
                result["trend_score"] = 65
            elif golden_cross:
                result["trend"] = "전환중"
                result["trend_score"] = 45
            else:
                result["trend"] = "하락"
                result["trend_score"] = 20
        else:
            result["trend"] = "판단불가"
            result["trend_score"] = 50
        
        # ========================================
        # 듀얼 모멘텀 계산 (핵심 추가!)
        # ========================================
        
        # 절대 모멘텀 (1개월, 3개월, 6개월 수익률)
        if len(close) >= 126:  # 6개월
            mom_1m = safe_div(result["price"] - float(close.iloc[-21]), float(close.iloc[-21])) * 100
            mom_3m = safe_div(result["price"] - float(close.iloc[-63]), float(close.iloc[-63])) * 100
            mom_6m = safe_div(result["price"] - float(close.iloc[-126]), float(close.iloc[-126])) * 100
        elif len(close) >= 63:  # 3개월
            mom_1m = safe_div(result["price"] - float(close.iloc[-21]), float(close.iloc[-21])) * 100
            mom_3m = safe_div(result["price"] - float(close.iloc[-63]), float(close.iloc[-63])) * 100
            mom_6m = mom_3m
        elif len(close) >= 21:  # 1개월
            mom_1m = safe_div(result["price"] - float(close.iloc[-21]), float(close.iloc[-21])) * 100
            mom_3m = mom_1m
            mom_6m = mom_1m
        else:
            mom_1m = mom_3m = mom_6m = 0
        
        result["momentum_1m"] = mom_1m
        result["momentum_3m"] = mom_3m
        result["momentum_6m"] = mom_6m
        
        # 가중 모멘텀 점수 (최근에 가중치)
        weighted_mom = mom_1m * 0.5 + mom_3m * 0.3 + mom_6m * 0.2
        result["momentum_weighted"] = weighted_mom
        
        # 절대 모멘텀 신호 (양수면 상승 추세)
        result["abs_momentum_positive"] = weighted_mom > 0
        
        # 모멘텀 점수 (강한 상승도 매수 기회!)
        if weighted_mom >= 30:
            result["momentum_score"] = 75  # 강한 상승 = 추세 추종 매수
            result["momentum_status"] = "강한상승추세"
        elif weighted_mom >= 15:
            result["momentum_score"] = 70
            result["momentum_status"] = "상승추세"
        elif weighted_mom >= 5:
            result["momentum_score"] = 60
            result["momentum_status"] = "약상승"
        elif weighted_mom >= -5:
            result["momentum_score"] = 50
            result["momentum_status"] = "횡보"
        elif weighted_mom >= -15:
            result["momentum_score"] = 40
            result["momentum_status"] = "약하락"
        elif weighted_mom >= -30:
            result["momentum_score"] = 55  # 하락은 역추세 매수 기회
            result["momentum_status"] = "하락추세"
        else:
            result["momentum_score"] = 65  # 급락은 강한 역추세 매수 기회
            result["momentum_status"] = "급락"
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        
        result["macd"] = float(macd.iloc[-1])
        result["macd_signal"] = float(signal.iloc[-1])
        result["macd_hist"] = float(macd_hist.iloc[-1])
        result["macd_hist_prev"] = float(macd_hist.iloc[-2])
        result["macd_rising"] = result["macd_hist"] > result["macd_hist_prev"]
        
        # 거래량 (있으면)
        if "Volume" in df.columns and float(df["Volume"].sum()) > 0:
            volume = df["Volume"]
            vol_ma20 = volume.rolling(20).mean()
            result["volume_ratio"] = safe_div(float(volume.iloc[-1]), float(vol_ma20.iloc[-1]), 1.0)
        else:
            result["volume_ratio"] = 1.0
        
        # ========================================
        # 종목별 종합 점수 계산
        # ========================================
        
        # 역추세 점수 (RSI, BB 기반) - 40%
        contrarian_score = (result["rsi_score"] * 0.5 + result["bb_score"] * 0.5)
        
        # 모멘텀 점수 - 35%
        momentum_score = result["momentum_score"]
        
        # 추세 점수 - 25%
        trend_score = result["trend_score"]
        
        # 종합 (역추세 + 모멘텀 듀얼 전략)
        result["total_score"] = int(
            contrarian_score * 0.40 +
            momentum_score * 0.35 +
            trend_score * 0.25
        )
        
        # 매수/매도 판단
        if result["total_score"] >= 70:
            result["action"] = "STRONG_BUY"
            result["action_label"] = "🟢 강력매수"
        elif result["total_score"] >= 55:
            result["action"] = "BUY"
            result["action_label"] = "🟢 매수"
        elif result["total_score"] >= 45:
            result["action"] = "HOLD"
            result["action_label"] = "🟡 관망"
        elif result["total_score"] >= 35:
            result["action"] = "REDUCE"
            result["action_label"] = "🟠 축소"
        else:
            result["action"] = "SELL"
            result["action_label"] = "🔴 매도"
        
        return result
        
    except Exception as e:
        if DEBUG:
            print(f"{ticker} 지표 계산 에러: {e}")
        return None

# =============================================================================
# 2. 섹터 강도 분석
# =============================================================================
def compute_sector_strength() -> Dict:
    """
    섹터별 강도 분석 (상대 모멘텀)
    """
    result = {
        "sector_scores": {},
        "sector_ranks": {},
        "top_sectors": [],
        "bottom_sectors": [],
    }
    
    spy_data = yf.Ticker("SPY").history(period="65d")
    if len(spy_data) < 21:
        return result
    
    spy_close = spy_data["Close"]
    spy_mom_1m = safe_div(float(spy_close.iloc[-1]) - float(spy_close.iloc[-21]), float(spy_close.iloc[-21])) * 100
    
    sector_moms = {}
    
    for etf, name in SECTOR_ETFS.items():
        try:
            data = yf.Ticker(etf).history(period="65d")
            if len(data) >= 21:
                close = data["Close"]
                mom_1m = safe_div(float(close.iloc[-1]) - float(close.iloc[-21]), float(close.iloc[-21])) * 100
                
                # 상대 모멘텀 (SPY 대비)
                relative_mom = mom_1m - spy_mom_1m
                
                sector_moms[etf] = {
                    "name": name,
                    "momentum_1m": mom_1m,
                    "relative_momentum": relative_mom,
                    "price": float(close.iloc[-1]),
                }
                
                # 강도 점수 (0-100)
                if relative_mom >= 10:
                    score = 90
                elif relative_mom >= 5:
                    score = 75
                elif relative_mom >= 2:
                    score = 60
                elif relative_mom >= -2:
                    score = 50
                elif relative_mom >= -5:
                    score = 40
                elif relative_mom >= -10:
                    score = 25
                else:
                    score = 10
                
                sector_moms[etf]["strength_score"] = score
                
        except Exception as e:
            if DEBUG:
                print(f"섹터 {etf} 에러: {e}")
    
    # 순위 계산
    sorted_sectors = sorted(sector_moms.items(), key=lambda x: x[1]["relative_momentum"], reverse=True)
    
    for rank, (etf, data) in enumerate(sorted_sectors, 1):
        sector_moms[etf]["rank"] = rank
    
    result["sector_scores"] = sector_moms
    result["top_sectors"] = [s[0] for s in sorted_sectors[:3]]
    result["bottom_sectors"] = [s[0] for s in sorted_sectors[-3:]]
    result["spy_momentum"] = spy_mom_1m
    
    return result

# =============================================================================
# 3. 암호화폐 지표
# =============================================================================
def get_crypto_indicators() -> Dict:
    """암호화폐 전용 지표"""
    result = {
        "btc_change": 0, "eth_change": 0, "sol_change": 0,
        "btc_mom_1m": 0, "eth_mom_1m": 0, "sol_mom_1m": 0,
        "crypto_score": 50,
    }
    
    try:
        for symbol, key in [("BTC-USD", "btc"), ("ETH-USD", "eth"), ("SOL-USD", "sol")]:
            data = yf.Ticker(symbol).history(period="30d")
            if len(data) >= 2:
                close = data["Close"]
                result[f"{key}_change"] = safe_div(
                    float(close.iloc[-1]) - float(close.iloc[-2]), 
                    float(close.iloc[-2])
                ) * 100
                
                if len(data) >= 21:
                    result[f"{key}_mom_1m"] = safe_div(
                        float(close.iloc[-1]) - float(close.iloc[-21]),
                        float(close.iloc[-21])
                    ) * 100
        
        # 가중 평균
        avg_change = result["btc_change"] * 0.5 + result["eth_change"] * 0.3 + result["sol_change"] * 0.2
        avg_mom = result["btc_mom_1m"] * 0.5 + result["eth_mom_1m"] * 0.3 + result["sol_mom_1m"] * 0.2
        
        # 점수 계산 (역추세 + 모멘텀)
        if avg_change <= -10:
            daily_score = 90
        elif avg_change <= -5:
            daily_score = 75
        elif avg_change <= -2:
            daily_score = 60
        elif avg_change <= 2:
            daily_score = 50
        elif avg_change <= 5:
            daily_score = 40
        else:
            daily_score = 25
        
        if avg_mom >= 20:
            mom_score = 70
        elif avg_mom >= 10:
            mom_score = 60
        elif avg_mom >= 0:
            mom_score = 50
        elif avg_mom >= -10:
            mom_score = 55
        else:
            mom_score = 65
        
        result["crypto_score"] = int(daily_score * 0.6 + mom_score * 0.4)
        
    except Exception as e:
        if DEBUG:
            print(f"암호화폐 에러: {e}")
    
    return result

# =============================================================================
# 4. 시장 전체 지표 (VIX, FGI 등)
# =============================================================================
def get_market_indicators() -> Dict:
    """시장 전체 지표"""
    result = {}
    
    # S&P 500
    try:
        sp = yf.Ticker("^GSPC").history(period="250d")
        if len(sp) >= 2:
            close = sp["Close"]
            result["sp_price"] = float(close.iloc[-1])
            result["sp_change"] = safe_div(
                float(close.iloc[-1]) - float(close.iloc[-2]),
                float(close.iloc[-2])
            ) * 100
            
            if len(sp) >= 200:
                result["sp_ma200"] = float(close.rolling(200).mean().iloc[-1])
                result["sp_above_200ma"] = result["sp_price"] > result["sp_ma200"]
            else:
                result["sp_above_200ma"] = True
    except:
        result["sp_change"] = 0
        result["sp_above_200ma"] = True
    
    # NASDAQ
    try:
        ndx = yf.Ticker("^NDX").history(period="5d")
        if len(ndx) >= 2:
            result["ndx_change"] = safe_div(
                float(ndx["Close"].iloc[-1]) - float(ndx["Close"].iloc[-2]),
                float(ndx["Close"].iloc[-2])
            ) * 100
    except:
        result["ndx_change"] = 0
    
    # VIX
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        if len(vix) >= 2:
            result["vix"] = float(vix["Close"].iloc[-1])
            result["vix_prev"] = float(vix["Close"].iloc[-2])
            result["vix_change"] = safe_div(
                result["vix"] - result["vix_prev"],
                result["vix_prev"]
            ) * 100
        else:
            result["vix"] = 20
            result["vix_change"] = 0
    except:
        result["vix"] = 20
        result["vix_change"] = 0
    
    # VIX 점수
    vix = result.get("vix", 20)
    if vix <= 12:
        result["vix_score"] = 60  # 너무 낮으면 자만 주의
    elif vix <= 15:
        result["vix_score"] = 70
    elif vix <= 20:
        result["vix_score"] = 60
    elif vix <= 25:
        result["vix_score"] = 50
    elif vix <= 30:
        result["vix_score"] = 55  # 공포 = 역추세 매수 기회
    elif vix <= 40:
        result["vix_score"] = 65
    else:
        result["vix_score"] = 75  # 극단적 공포 = 강한 매수 기회
    
    # FGI
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = res.json()
        if "data" in data and len(data["data"]) > 0:
            result["fgi"] = int(data["data"][0].get("value", 50))
        else:
            result["fgi"] = 50
    except:
        result["fgi"] = 50
    
    # FGI 역추세 점수
    fgi = result.get("fgi", 50)
    result["fgi_contrarian"] = 100 - fgi
    
    # 매크로
    try:
        fx = yf.Ticker("USDKRW=X").history(period="5d")["Close"]
        result["fx"] = float(fx.iloc[-1]) if len(fx) > 0 else 1400
        
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
        result["tnx"] = float(tnx.iloc[-1]) if len(tnx) > 0 else 4.0
        
        oil = yf.Ticker("CL=F").history(period="5d")["Close"]
        result["oil"] = float(oil.iloc[-1]) if len(oil) > 0 else 70
    except:
        result["fx"] = 1400
        result["tnx"] = 4.0
        result["oil"] = 70
    
    return result

# =============================================================================
# 5. 종목별 매수/매도 금액 계산 (개별 판단!)
# =============================================================================
def compute_ticker_allocation(
    ticker: str,
    ticker_data: Dict,
    sector_data: Dict,
    market_data: Dict,
    crypto_data: Dict,
    base_weight: float,
    total_budget: int,
) -> Dict:
    """
    종목별 매수/매도 금액 계산 (개별 판단)
    - 종목별로 BUY / SELL / HOLD 따로 판단
    - 매수 금액, 매도 비율 개별 계산
    """
    config = PORTFOLIO[ticker]
    sector = config["sector"]
    sector_etf = config.get("sector_etf", "")
    
    result = {
        "ticker": ticker,
        "base_weight": base_weight,
        "base_amount": int(total_budget * base_weight),
    }
    
    # 1. 종목 점수 (이미 계산됨)
    ticker_score = ticker_data.get("total_score", 50) if ticker_data else 50
    result["ticker_score"] = ticker_score
    
    # 2. 섹터 강도 점수
    if sector == "crypto":
        sector_score = crypto_data.get("crypto_score", 50)
    elif sector_etf and sector_etf in sector_data.get("sector_scores", {}):
        sector_score = sector_data["sector_scores"][sector_etf].get("strength_score", 50)
    else:
        sector_score = 50
    result["sector_score"] = sector_score
    
    # 3. 시장 점수
    market_score = (
        market_data.get("vix_score", 50) * 0.4 +
        market_data.get("fgi_contrarian", 50) * 0.4 +
        (60 if market_data.get("sp_above_200ma", True) else 30) * 0.2
    )
    result["market_score"] = int(market_score)
    
    # 4. 종합 점수 (종목 50%, 섹터 25%, 시장 25%)
    final_score = int(
        ticker_score * 0.50 +
        sector_score * 0.25 +
        market_score * 0.25
    )
    result["final_score"] = final_score
    
    # 5. 일간 수익률
    daily_return = ticker_data.get("daily_return", 0) if ticker_data else 0
    result["daily_return"] = daily_return
    
    # 6. RSI
    rsi = ticker_data.get("rsi", 50) if ticker_data else 50
    result["rsi"] = rsi
    
    # 7. 모멘텀
    momentum = ticker_data.get("momentum_weighted", 0) if ticker_data else 0
    result["momentum"] = momentum
    
    # =========================================
    # 종목별 개별 매수/매도 판단 (핵심!)
    # =========================================
    
    # 매도 조건 체크
    sell_reasons = []
    
    # RSI 극과열
    if rsi >= 85:
        sell_reasons.append(f"RSI 극과열({rsi:.0f})")
        result["action"] = "SELL"
        result["sell_pct"] = 70  # 70% 매도
    elif rsi >= 78:
        sell_reasons.append(f"RSI 과열({rsi:.0f})")
        result["action"] = "SELL"
        result["sell_pct"] = 50  # 50% 매도
    elif rsi >= 72:
        sell_reasons.append(f"RSI 높음({rsi:.0f})")
        result["action"] = "SELL"
        result["sell_pct"] = 30  # 30% 매도
    
    # 종목 점수 매우 낮음
    if final_score <= 25:
        sell_reasons.append(f"점수 매우 낮음({final_score})")
        result["action"] = "SELL"
        result["sell_pct"] = max(result.get("sell_pct", 0), 50)
    elif final_score <= 35:
        sell_reasons.append(f"점수 낮음({final_score})")
        result["action"] = "SELL"
        result["sell_pct"] = max(result.get("sell_pct", 0), 30)
    
    # 급등 (당일 +8% 이상)
    if daily_return >= 8:
        sell_reasons.append(f"급등({daily_return:+.1f}%)")
        result["action"] = "SELL"
        result["sell_pct"] = max(result.get("sell_pct", 0), 30)
    
    # 매도 판정
    if sell_reasons:
        result["action"] = "SELL"
        result["action_label"] = f"🔴 매도 {result['sell_pct']}%"
        result["reason"] = ", ".join(sell_reasons)
        result["buy_amount"] = 0
        return result
    
    # =========================================
    # 관망 조건 체크
    # =========================================
    hold_reasons = []
    
    # RSI 애매한 구간
    if 65 <= rsi < 72:
        hold_reasons.append(f"RSI 경계({rsi:.0f})")
    
    # 점수 애매
    if 35 < final_score <= 45:
        hold_reasons.append(f"점수 애매({final_score})")
    
    # 급등 (당일 +4% 이상)
    if 4 <= daily_return < 8:
        hold_reasons.append(f"상승({daily_return:+.1f}%)")
    
    if hold_reasons and final_score <= 50:
        result["action"] = "HOLD"
        result["action_label"] = "🟡 관망"
        result["reason"] = ", ".join(hold_reasons)
        result["buy_amount"] = 0
        result["sell_pct"] = 0
        return result
    
    # =========================================
    # 매수 조건 (점수 높을수록 많이)
    # =========================================
    
    # 매수 배율 계산
    if final_score >= 80:
        multiplier = 2.0
        action_label = "🟢🟢 강력매수"
    elif final_score >= 70:
        multiplier = 1.6
        action_label = "🟢 적극매수"
    elif final_score >= 60:
        multiplier = 1.3
        action_label = "🟢 매수"
    elif final_score >= 50:
        multiplier = 1.0
        action_label = "🟢 일반매수"
    elif final_score >= 45:
        multiplier = 0.7
        action_label = "🟡 소량매수"
    else:
        multiplier = 0.0
        action_label = "🟡 관망"
    
    # 일간 수익률 보정 (역추세)
    if daily_return <= -8:
        multiplier *= 1.8
        action_label += " (급락!)"
    elif daily_return <= -5:
        multiplier *= 1.5
        action_label += " (하락)"
    elif daily_return <= -3:
        multiplier *= 1.3
    elif daily_return <= -1.5:
        multiplier *= 1.15
    elif daily_return >= 3:
        multiplier *= 0.6
    elif daily_return >= 1.5:
        multiplier *= 0.8
    
    # RSI 보정 (과매도면 추가)
    if rsi <= 25:
        multiplier *= 1.4
        action_label += " (과매도!)"
    elif rsi <= 35:
        multiplier *= 1.2
    
    result["multiplier"] = round(multiplier, 2)
    
    # 최종 매수 금액
    buy_amount = int(result["base_amount"] * multiplier)
    result["buy_amount"] = max(0, buy_amount)
    result["sell_pct"] = 0
    
    if buy_amount > 0:
        result["action"] = "BUY"
        result["action_label"] = action_label
        result["reason"] = f"점수 {final_score}, RSI {rsi:.0f}"
    else:
        result["action"] = "HOLD"
        result["action_label"] = "🟡 관망"
        result["reason"] = "매수 조건 미충족"
    
    return result

# =============================================================================
# 6. 전체 포트폴리오 계산 (매수/매도 분리)
# =============================================================================
def compute_portfolio_allocation(
    ticker_indicators: Dict[str, Dict],
    sector_data: Dict,
    market_data: Dict,
    crypto_data: Dict,
    total_budget: int,
) -> Dict[str, Dict]:
    """
    전체 포트폴리오 배분 계산
    - 매수 종목: 금액 배분
    - 매도 종목: 매도 비율 표시
    """
    allocations = {}
    buy_tickers = {}
    sell_tickers = {}
    hold_tickers = {}
    
    # 1단계: 각 종목별 판단
    for ticker, config in PORTFOLIO.items():
        ticker_data = ticker_indicators.get(ticker)
        base_weight = config["weight"] / 100
        
        alloc = compute_ticker_allocation(
            ticker=ticker,
            ticker_data=ticker_data,
            sector_data=sector_data,
            market_data=market_data,
            crypto_data=crypto_data,
            base_weight=base_weight,
            total_budget=total_budget,
        )
        
        allocations[ticker] = alloc
        
        if alloc["action"] == "BUY":
            buy_tickers[ticker] = alloc
        elif alloc["action"] == "SELL":
            sell_tickers[ticker] = alloc
        else:
            hold_tickers[ticker] = alloc
    
    # 2단계: 매수 종목만 정규화 (총액 맞추기)
    if buy_tickers:
        total_buy_amount = sum(a["buy_amount"] for a in buy_tickers.values())
        
        if total_buy_amount > 0:
            scale = total_budget / total_buy_amount
            for ticker in buy_tickers:
                raw = allocations[ticker]["buy_amount"] * scale
                allocations[ticker]["final_amount"] = int(round(raw / 1000) * 1000)
        
        # 반올림 오차 보정
        total_final = sum(allocations[t]["final_amount"] for t in buy_tickers)
        diff = total_budget - total_final
        
        if diff != 0 and buy_tickers:
            max_ticker = max(buy_tickers, key=lambda t: allocations[t]["final_amount"])
            allocations[max_ticker]["final_amount"] += diff
    
    # 매도/관망 종목은 final_amount = 0
    for ticker in sell_tickers:
        allocations[ticker]["final_amount"] = 0
    for ticker in hold_tickers:
        allocations[ticker]["final_amount"] = 0
    
    return allocations

# =============================================================================
# 7. 전체 매매 결정 (레버리지 ETF 최적화)
# =============================================================================
def compute_overall_decision(
    market_data: Dict,
    sector_data: Dict,
    ticker_indicators: Dict[str, Dict],
) -> Tuple[str, int, str]:
    """
    전체 매매 결정 (레버리지 ETF 특화)
    
    레버리지 ETF 핵심 전략:
    - 급락 시 → 공격적 매수 (MAX)
    - 평상시 → 소량 또는 관망
    - 과열 시 → 매도
    - 횡보 시 → 관망 (변동성 손실 방지)
    """
    # 시장 점수
    vix = market_data.get("vix", 20)
    vix_change = market_data.get("vix_change", 0)
    fgi = market_data.get("fgi", 50)
    fgi_contrarian = market_data.get("fgi_contrarian", 50)
    sp_above_200ma = market_data.get("sp_above_200ma", True)
    sp_change = market_data.get("sp_change", 0)
    ndx_change = market_data.get("ndx_change", 0)
    avg_change = (sp_change + ndx_change) / 2
    
    # 종목 평균 점수
    ticker_scores = [t.get("total_score", 50) for t in ticker_indicators.values() if t]
    avg_ticker_score = sum(ticker_scores) / len(ticker_scores) if ticker_scores else 50
    
    # =========================================
    # 레버리지 ETF 특화 조건
    # =========================================
    
    # 조건 1: 하락 추세 (200MA 아래) → 매도
    if not sp_above_200ma:
        if avg_ticker_score < 40:
            return "SELL", 100, "🔴 하락 추세 + 약세 - 전량 매도 권고"
        else:
            return "SELL", 50, "🟠 하락 추세 - 절반 매도 권고"
    
    # 조건 2: 극단적 급락 (VIX 스파이크 + 지수 급락)
    if vix >= 30 and avg_change <= -3:
        return "BUY", MAX_DAILY_BUDGET, "🟢🟢 급락장 - 최대 매수! (공포에 사라)"
    
    # 조건 3: 강한 급락
    if avg_change <= -4:
        return "BUY", MAX_DAILY_BUDGET, "🟢🟢 급락 - 최대 매수!"
    
    if avg_change <= -2.5:
        amount = int(MAX_DAILY_BUDGET * 0.7)
        return "BUY", amount, "🟢 하락 - 적극 매수"
    
    if avg_change <= -1.5:
        amount = int(BASE_DAILY_BUDGET * 1.5)
        return "BUY", amount, "🟢 하락 - 추가 매수"
    
    # 조건 4: VIX 급등 (공포)
    if vix >= 28 and vix_change >= 15:
        amount = int(MAX_DAILY_BUDGET * 0.6)
        return "BUY", amount, "🟢 VIX 급등 - 공포 매수"
    
    # 조건 5: 과열 (FGI 높음 + RSI 높음)
    if fgi >= 80 and avg_ticker_score <= 35:
        return "SELL", 50, "🟠 과열 - 절반 매도 권고"
    
    if fgi >= 75 and avg_ticker_score <= 40:
        return "SELL", 30, "🟡 과열 경계 - 일부 매도 권고"
    
    # 조건 6: 강한 급등 (추격 매수 금지)
    if avg_change >= 2.5:
        return "HOLD", 0, "⏸️ 급등 - 관망 (추격 금지)"
    
    if avg_change >= 1.5:
        return "HOLD", 0, "⏸️ 상승 - 관망"
    
    # 조건 7: 횡보장 (변동성 손실 주의)
    if abs(avg_change) <= 0.3 and vix <= 15:
        return "HOLD", 0, "⏸️ 횡보 + 저변동 - 관망 (변동성 손실 주의)"
    
    # =========================================
    # 일반 조건 (점수 기반)
    # =========================================
    
    # 종합 점수 계산
    market_score = int(
        fgi_contrarian * 0.30 +           # FGI 역추세
        avg_ticker_score * 0.40 +         # 종목 점수
        (65 if sp_above_200ma else 25) * 0.15 +  # 추세
        (70 if vix >= 22 else 50) * 0.15  # VIX (높으면 매수 기회)
    )
    
    # 점수 기반 결정 (레버리지 ETF는 보수적)
    if market_score <= 35:
        return "SELL", 30, "🟡 약세 - 일부 매도 권고"
    elif market_score <= 45:
        return "HOLD", 0, "⏸️ 관망"
    elif market_score <= 55:
        # 소량 매수 (레버리지는 신중하게)
        return "BUY", MIN_DAILY_BUDGET if MIN_DAILY_BUDGET > 0 else 5000, "소량 매수"
    elif market_score <= 65:
        return "BUY", BASE_DAILY_BUDGET, "일반 매수"
    elif market_score <= 75:
        amount = int(BASE_DAILY_BUDGET * 1.3)
        return "BUY", amount, "🟢 적극 매수"
    else:
        amount = int(BASE_DAILY_BUDGET * 1.5)
        return "BUY", min(amount, MAX_DAILY_BUDGET), "🟢 강력 매수"

# =============================================================================
# 8. 메시지 생성
# =============================================================================
def generate_message(
    market_data: Dict,
    sector_data: Dict,
    ticker_indicators: Dict[str, Dict],
    crypto_data: Dict,
    action: str,
    total_amount: int,
    reason: str,
    allocations: Dict[str, Dict],
) -> str:
    """텔레그램 메시지 생성 - 매수/매도/관망 분리"""
    
    dday = get_dday()
    
    # 시장 결론
    if action == "SELL":
        market_conclusion = f"🔴 시장 매도 권고 ({total_amount}%)"
    elif action == "HOLD":
        market_conclusion = "⏸️ 시장 관망"
    else:
        market_conclusion = f"🟢 시장 매수 ({total_amount:,}원)"
    
    # 섹터 강도 TOP 3
    sector_scores = sector_data.get("sector_scores", {})
    top_sectors = sector_data.get("top_sectors", [])
    top_sector_text = ""
    for etf in top_sectors[:3]:
        if etf in sector_scores:
            info = sector_scores[etf]
            top_sector_text += f"  {info['name']}: {info['relative_momentum']:+.1f}%\n"
    
    # =========================================
    # 종목별 매수/매도/관망 분리
    # =========================================
    buy_lines = []
    sell_lines = []
    hold_lines = []
    
    total_buy = 0
    
    for ticker, alloc in allocations.items():
        t_data = ticker_indicators.get(ticker, {})
        rsi = alloc.get("rsi", t_data.get("rsi", 0) if t_data else 0)
        momentum = alloc.get("momentum", t_data.get("momentum_weighted", 0) if t_data else 0)
        score = alloc.get("final_score", 50)
        daily_ret = alloc.get("daily_return", 0)
        action_label = alloc.get("action_label", "")
        reason_text = alloc.get("reason", "")
        
        if alloc["action"] == "BUY":
            final_amt = alloc.get("final_amount", 0)
            total_buy += final_amt
            multiplier = alloc.get("multiplier", 1.0)
            buy_lines.append(
                f"  ✅ <b>{ticker}</b>: {final_amt:,}원 (x{multiplier})\n"
                f"      점수:{score} | RSI:{rsi:.0f} | 일간:{daily_ret:+.1f}%"
            )
        elif alloc["action"] == "SELL":
            sell_pct = alloc.get("sell_pct", 0)
            sell_lines.append(
                f"  🚨 <b>{ticker}</b>: 보유분의 <b>{sell_pct}% 매도</b>\n"
                f"      📍 사유: {reason_text}\n"
                f"      점수:{score} | RSI:{rsi:.0f} | 일간:{daily_ret:+.1f}%"
            )
        else:  # HOLD
            hold_lines.append(
                f"  ⏸️ {ticker}: 관망 (점수:{score}, RSI:{rsi:.0f})"
            )
    
    # 텍스트 조합
    buy_text = "\n".join(buy_lines) if buy_lines else "  없음"
    sell_text = "\n".join(sell_lines) if sell_lines else "  없음"
    hold_text = "\n".join(hold_lines) if hold_lines else "  없음"
    
    # 암호화폐
    crypto_text = f"""• BTC: {crypto_data.get('btc_change', 0):+.1f}% (1M: {crypto_data.get('btc_mom_1m', 0):+.1f}%)
• ETH: {crypto_data.get('eth_change', 0):+.1f}% (1M: {crypto_data.get('eth_mom_1m', 0):+.1f}%)
• SOL: {crypto_data.get('sol_change', 0):+.1f}% (1M: {crypto_data.get('sol_mom_1m', 0):+.1f}%)"""
    
    # 요약 통계
    buy_count = len(buy_lines)
    sell_count = len(sell_lines)
    hold_count = len(hold_lines)
    
    msg = f"""📊 <b>레버리지 ETF 시그널 v3.0</b>

━━━━━━━━━━━━━━━━━━━━
📌 <b>시장: {market_conclusion}</b>
• 사유: {reason}
━━━━━━━━━━━━━━━━━━━━

📈 <b>시장 현황</b>
• S&P500: {market_data.get('sp_change', 0):+.2f}%
• NASDAQ: {market_data.get('ndx_change', 0):+.2f}%
• VIX: {market_data.get('vix', 20):.1f} ({market_data.get('vix_change', 0):+.1f}%)
• FGI: {market_data.get('fgi', 50)} (역추세: {market_data.get('fgi_contrarian', 50)})
• 200MA: {'상위 ✅' if market_data.get('sp_above_200ma', True) else '하위 ❌'}

📊 <b>섹터 강도 TOP 3</b>
{top_sector_text}
💹 <b>암호화폐</b>
{crypto_text}

━━━━━━━━━━━━━━━━━━━━
💼 <b>종목별 신호</b> (매수 {buy_count}개 / 매도 {sell_count}개 / 관망 {hold_count}개)
━━━━━━━━━━━━━━━━━━━━

🟢 <b>매수 종목</b> (총 {total_buy:,}원)
{buy_text}

🔴 <b>매도 종목</b>
{sell_text}

🟡 <b>관망 종목</b>
{hold_text}

━━━━━━━━━━━━━━━━━━━━
🌍 <b>매크로</b>
• 환율: {market_data.get('fx', 0):,.0f}원 | 금리: {market_data.get('tnx', 0):.2f}% | 유가: ${market_data.get('oil', 0):.1f}

📅 D-Day: {TARGET_DATE} (D-{dday})
"""
    
    return msg

# =============================================================================
# 9. 메인 실행
# =============================================================================
def main():
    print("=" * 50)
    print("레버리지 ETF 시그널 봇 v3.0 시작")
    print("=" * 50)
    
    # 1. 시장 지표
    print("📊 시장 지표 수집...")
    market_data = get_market_indicators()
    
    # 2. 섹터 강도
    print("📊 섹터 강도 분석...")
    sector_data = compute_sector_strength()
    
    # 3. 암호화폐
    print("📊 암호화폐 지표...")
    crypto_data = get_crypto_indicators()
    
    # 4. 종목별 지표
    print("📊 종목별 지표 계산...")
    ticker_indicators = {}
    for ticker in PORTFOLIO:
        print(f"  - {ticker}...")
        ticker_indicators[ticker] = compute_ticker_indicators(ticker)
    
    # 5. 전체 매매 결정
    print("📊 매매 결정...")
    action, amount, reason = compute_overall_decision(
        market_data, sector_data, ticker_indicators
    )
    
    # 6. 포트폴리오 배분
    if action == "BUY" and amount > 0:
        allocations = compute_portfolio_allocation(
            ticker_indicators, sector_data, market_data, crypto_data, amount
        )
    else:
        allocations = {t: {
            "final_amount": 0,
            "action": action,
            "final_score": 0,
            "daily_return": ticker_indicators.get(t, {}).get("daily_return", 0) if ticker_indicators.get(t) else 0
        } for t in PORTFOLIO}
    
    # 7. 메시지 생성
    message = generate_message(
        market_data, sector_data, ticker_indicators, crypto_data,
        action, amount, reason, allocations
    )
    
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50)
    
    send_telegram(message)
    print("\n✅ 완료")

if __name__ == "__main__":
    main()
