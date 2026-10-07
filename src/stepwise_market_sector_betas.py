import csv
import math
from pathlib import Path

from risk_model_betas import TSAT_HIGH, TSAT_LOW, regress, tsat_beta

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REMAINING_SECTOR_T_THRESHOLD = 2.0
SECTOR_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def read_table(path):
    """Load a date plus numeric factor/return columns from CSV."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        table = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in table:
                table[name].append(
                    row[name] if name == "date" else float(row[name]) if row[name] else None
                )
    return table


def read_companies(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {row["ticker"]: row for row in csv.DictReader(file)}


def aligned_series(table, dates, column):
    values = dict(zip(table["date"], table[column]))
    return [values[day] for day in dates]


def passes_tsat(fit):
    return fit["tstat"] is not None and TSAT_LOW <= fit["tstat"] <= TSAT_HIGH


def standard_deviation(values):
    values = [value for value in values if value is not None]
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def remaining_sector_regressions(current, factors, own_sector):
    """Add the strongest unused sector with |t| > 2 until none qualify."""
    remaining = sorted(name for name in factors if name != own_sector)
    # Keep one common sample so candidate residual sums of squares are comparable.
    valid = [
        i for i, value in enumerate(current)
        if value is not None and all(factors[name][i] is not None for name in remaining)
    ]
    current = [current[i] for i in valid]
    factors = {name: [values[i] for i in valid] for name, values in factors.items()}
    selected, unselected = {}, {}

    while remaining:
        unselected = {name: regress(factors[name], current) for name in remaining}
        candidates = [
            (sum(value ** 2 for value in fit["residuals"]), name, fit)
            for name, fit in unselected.items()
            if fit["tstat"] is not None
            and abs(fit["tstat"]) > REMAINING_SECTOR_T_THRESHOLD
        ]
        if not candidates:
            break
        _, name, fit = min(candidates, key=lambda item: (item[0], item[1]))
        selected[name] = fit
        current = fit["residuals"]
        remaining.remove(name)
        del unselected[name]

    return selected, current, unselected


def main():
    # Load Group 3's local stagger 3 returns, sector residuals, and company map.
    companies = read_companies(DATA_DIR / "companies.csv")
    returns_table = read_table(DATA_DIR / "stagger_3_returns.csv")
    factor_table = read_table(DATA_DIR / "stagger_3_sector_factors.csv")
    dates = sorted(returns_table["date"])
    returns = {
        name: aligned_series(returns_table, dates, name)
        for name in returns_table if name != "date"
    }
    factors = {
        name: aligned_series(factor_table, dates, name)
        for name in factor_table if name != "date"
    }

    rows = []
    for ticker, company in companies.items():
        current = returns[ticker]
        currency = company["currency"]
        currency_fit = None

        # 1. For foreign stocks, regress on local currency and estimate beta.
        if currency != "USD":
            currency_fit = regress(returns[f"{currency}/USD"], current)

            # 2. Keep the currency beta when -2 <= t-stat <= 2, carry its fitted residuals forward only when retained
            if passes_tsat(currency_fit):
                current = currency_fit["residuals"]

        # 3. Regress the remaining returns on AUD; apply the same t-stat rule.
        aud_fit = regress(returns["AUD/USD"], current)
        if passes_tsat(aud_fit):
            current = aud_fit["residuals"]

        # 4. Regress on SPY, keep its beta, and carry its residuals forward.
        spy_fit = regress(returns["SPY"], current)
        current = spy_fit["residuals"]

        # 5. Regress the residuals on the stock's own sector residual factor.
        sector_factor = SECTOR_TO_ETF[company["sector"]] + "_residual"
        sector_fit = regress(factors[sector_factor], current)
        current = sector_fit["residuals"]

        # 6. Step through remaining sector factors. Add the best candidate when |t-stat| > 2
        risk_before_remaining = standard_deviation(current)
        selected, current, unselected = remaining_sector_regressions(
            current, factors, sector_factor
        )

        # 7. Save every stage's betas, selection order, and final residual risk.
        row = {
            "ticker": ticker,
            "stagger": "stagger_3",
            "sector": company["sector"],
            "currency": currency,
            "currency_beta": None if currency_fit is None else tsat_beta(currency_fit),
            "currency_tstat": None if currency_fit is None else currency_fit["tstat"],
            "aud_beta": tsat_beta(aud_fit),
            "aud_tstat": aud_fit["tstat"],
            "spy_beta": spy_fit["beta"],
            "sector_factor": sector_factor,
            "sector_beta": sector_fit["beta"],
            "nobs": spy_fit["nobs"],
            "remaining_sector_order": ";".join(selected),
            "remaining_sector_count": len(selected),
            "remaining_nobs": len(current),
            "residual_risk_before_remaining": risk_before_remaining,
            "residual_risk": standard_deviation(current),
        }
        for name in sorted(factors):
            fit = selected.get(name, unselected.get(name))
            prefix = "remaining_" + name.removesuffix("_residual")
            row[prefix + "_beta"] = (
                None if fit is None else fit["beta"] if name in selected else 0.0
            )
            row[prefix + "_tstat"] = None if fit is None else fit["tstat"]
        rows.append(row)

    # Write one result row per stock.
    with (DATA_DIR / "market_sector_betas.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(
            {name: "" if value is None else f"{value:.12g}" if isinstance(value, float) else value
             for name, value in row.items()}
            for row in rows
        )
    print(f"Wrote {len(rows)} stagger 3 beta rows to {DATA_DIR / 'market_sector_betas.csv'}")


if __name__ == "__main__":
    main()
