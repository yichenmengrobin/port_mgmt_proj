import csv
import math
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"
TSAT_LOW, TSAT_HIGH = -2.0, 2.0
FX_COLUMNS = {
    "MXN": "MXN/USD",
    "SGD": "SGD/USD",
    "CAD": "CAD/USD",
    "DKK": "DKK/USD",
    "EUR": "EUR/USD",
}


def read_returns(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        dates, values = [], {name: [] for name in reader.fieldnames if name != "date"}
        for row in reader:
            dates.append(row["date"])
            for name in values:
                values[name].append(float(row[name]) if row[name] else None)
    return dates, values


def regress(x, y):
    """OLS with intercept; return beta, t-statistic, residuals and N."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    xs, ys = zip(*pairs)
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((a - xbar) ** 2 for a in xs)
    beta = sum((a - xbar) * (b - ybar) for a, b in pairs) / sxx
    alpha = ybar - beta * xbar
    errors = [b - alpha - beta * a for a, b in pairs]
    se = math.sqrt(sum(e * e for e in errors) / (len(pairs) - 2) / sxx)
    tstat = beta / se if se else None
    residuals = [
        b - alpha - beta * a if a is not None and b is not None else None
        for a, b in zip(x, y)
    ]
    return {
        "beta": beta,
        "tstat": tstat,
        "alpha": alpha,
        "residuals": residuals,
        "nobs": len(pairs),
    }


def tsat_beta(fit):
    """Keep beta when its t-statistic is between -2 and 2, inclusive."""
    if fit is None or fit["tstat"] is None:
        return None
    return fit["beta"] if TSAT_LOW <= fit["tstat"] <= TSAT_HIGH else 0.0


def main():
    # Load the Group 3 returns and the stock currency labels.
    dates, returns = read_returns(DATA / "stagger_3_returns.csv")
    with (DATA / "companies.csv").open(newline="", encoding="utf-8-sig") as file:
        companies = list(csv.DictReader(file))

    results = []
    for company in companies:
        ticker, currency = company["ticker"], company["currency"]
        stock_returns = returns[ticker]
        currency_fit = None
        currency_beta = 0.0

        # 1. For foreign stocks, regress stock returns on the local currency and estimate its betas
        if currency != "USD":
            currency_fit = regress(
                returns[FX_COLUMNS[currency]], stock_returns
            )
            currency_beta = tsat_beta(currency_fit)

            # 2. Apply the currency-beta t-stat rule. Keep the fitted residuals for the next stage only when -2 <= t-stat <= 2.
            if TSAT_LOW <= currency_fit["tstat"] <= TSAT_HIGH:
                stock_returns = currency_fit["residuals"]

        # 3. Regress the remaining stock returns on AUD and estimate its beta.
        aud_fit = regress(returns["AUD/USD"], stock_returns)

        # 4. Apply the same -2 to 2 t-stat rule to the AUD beta.
        results.append({
            "ticker": ticker,
            "stagger": "stagger_3",
            "currency": currency,
            "currency_beta_raw": None if currency_fit is None else currency_fit["beta"],
            "currency_tstat": None if currency_fit is None else currency_fit["tstat"],
            "currency_beta": currency_beta,
            "aud_beta_raw": aud_fit["beta"],
            "aud_tstat": aud_fit["tstat"],
            "aud_beta": tsat_beta(aud_fit),
            "nobs": aud_fit["nobs"],
            "start_date": dates[0],
            "end_date": dates[-1],
        })

    # 5. Save one set of currency/AUD beta results per stock for stagger 3.
    with (DATA / "betas.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} stagger 3 beta rows to {DATA / 'betas.csv'}")


if __name__ == "__main__":
    main()
