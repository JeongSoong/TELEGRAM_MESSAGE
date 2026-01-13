import requests
from bs4 import BeautifulSoup
import yfinance as yf

def send_telegram(message):
    bot_token = "8386665445:AAG5bEM30o9UzU-9NO9cGM7Lg0K7b1xcbFk"
    chat_id = "6983611450"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(url, data=payload)

def fetch_market_data():
    sp = yf.Ticker("^GSPC").history(period="1d")
    sp_change = float((sp["Close"][0] - sp["Open"][0]) / sp["Open"][0] * 100)

    ndx = yf.Ticker("^NDX").history(period="1d")
    ndx_change = float((ndx["Close"][0] - ndx["Open"][0]) / ndx["Open"][0] * 100)

    vix = yf.Ticker("^VIX").history(period="1d")
    vix_value = float(vix["Close"][0])

    hist = yf.Ticker("^GSPC").history(period="14d")
    delta = hist["Close"].diff()
    gain = delta.where(delta > 0, 0).mean()
    loss = -delta.where(delta < 0, 0).mean()
    rs = gain / loss if loss != 0 else 999
    rsi = 100 - (100 / (1 + rs))

    url = "https://news.google.com/search?q=stock+market"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    headlines = " ".join([h.text for h in soup.find_all("h3")])
    sentiment = 1 if ("surge" in headlines.lower() or "record" in headlines.lower()) else 0

    return sp_change, ndx_change, vix_value, rsi, sentiment

def main():
    sp_change, ndx_change, vix_value, rsi, sentiment = fetch_market_data()

    change = max(abs(sp_change), abs(ndx_change))

    score = 0
    if rsi >= 90: score += 3
    elif rsi >= 85: score += 2
    elif rsi >= 80: score += 1

    if change >= 10: score += 3
    elif change >= 8: score += 2
    elif change >= 5: score += 1

    if vix_value <= 12: score += 3
    elif vix_value <= 14: score += 2
    elif vix_value <= 16: score += 1

    if sentiment == 1: score += 2

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

    telegram_message = (
        f"[정수 버블 체크]\n"
        f"S&P 변동폭: {sp_change:.2f}%\n"
        f"나스닥 변동폭: {ndx_change:.2f}%\n"
        f"VIX: {vix_value:.2f}\n"
        f"RSI: {rsi:.2f}\n"
        f"심리: {'광기' if sentiment else '보통'}\n"
        f"총 점수: {score}\n"
        f"결론: {result}\n\n"
        f"지수 평균 변동률: {avg_change:.4f}%\n"
        f"기준금액: {base_amount:,}원\n\n"
        f"[포트폴리오 배분]\n{allocation}"
    )

    send_telegram(telegram_message)
    print("텔레그램 전송 완료")

if __name__ == "__main__":
    main()
