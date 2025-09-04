from typing import Dict, List, Tuple

import csv
from statistics import NormalDist

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def read_positions(csv_path: str, limit: int | None = None) -> Tuple[List[str], np.ndarray]:
    """Parse positions CSV and return tickers with dollar exposures.

    The input file contains a few metadata lines before the actual header.
    The data rows provide a `Ticker` column and a `Mkt Val` column representing
    the dollar exposure of each position.  We ignore the other columns and
    convert the exposures to floats.  Cash lines reported with ticker ``USD``
    and long name ``US DOLLAR`` are skipped so they don't collide with the
    ProShares Ultra Semiconductors ETF, which also uses ticker ``USD``.
    """

    tickers: List[str] = []
    exposures: List[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)

        # Skip metadata lines until the real header starting with "Long Name"
        for row in reader:
            if row and row[0].strip() == "Long Name":
                break

        for row in reader:
            if limit is not None and len(tickers) >= limit:
                break
            if not row or len(row) < 8:
                continue

            long_name = row[0].strip()
            ticker = row[7].strip()
            # Some custodians report cash as a position with ticker "USD".
            # That clashes with the ticker for the ProShares Ultra Semiconductors ETF
            # (also "USD"). If the row represents the actual currency position,
            # identified by the long name "US DOLLAR", skip it.
            if ticker.upper() == "USD" and long_name.upper() == "US DOLLAR":
                continue

            exposure_str = row[6].replace(",", "").strip()
            try:
                exposure = float(exposure_str)
            except ValueError:
                continue

            tickers.append(ticker)
            exposures.append(exposure)

    return tickers, np.array(exposures)

def fetch_covariance(tickers: List[str], period: str = "5y") -> Tuple[pd.DataFrame, List[str]]:
    """Download adjusted close prices and compute return covariance."""
    data = yf.download(tickers, period=period, auto_adjust=False)["Adj Close"]
    data = data.bfill()
    returns = data.pct_change().dropna(axis=1, how="all")
    cov = returns.cov()
    return cov, returns.corr(), returns.columns.tolist()


def portfolio_var(
    csv_path: str,
    period: str = "5y",
    confidence: float = 0.99,
    limit: int | None = None,
) -> Tuple[float, pd.Series, pd.Series, pd.DataFrame]:
    """Compute portfolio Value at Risk and related sensitivities.

    Returns
    -------
    total_var: float
        Portfolio Value at Risk in dollar terms.
    contributions: pd.Series
        VaR contribution for each ticker.
    grad_var: pd.Series
        Gradient of variance with respect to dollar exposures (2ΣE).
    deriv_matrix: pd.DataFrame
        Matrix of Σ multiplied by exposures before summing rows.

    """

    tickers, exposures = read_positions(csv_path, limit=limit)
    cov, corr, valid_tickers = fetch_covariance(tickers, period)
    exposure_map: Dict[str, float] = dict(zip(tickers, exposures))
    aligned = np.array([exposure_map[t] for t in valid_tickers])

    cov_matrix = cov.values

    z = NormalDist().inv_cdf(confidence)

    variance_matrix = aligned.reshape([1,-1]) * cov_matrix * aligned.reshape([-1,1])
    variance = np.sum(variance_matrix)
    VaR = np.sqrt(252) * z * np.sqrt(variance)

    variance_prime_matrix = 2 * cov_matrix * aligned.reshape([-1,1])
    variance_prime = np.sum(variance_prime_matrix,axis=0)
    VaR_prime = np.sqrt(252) * z * 1/(2 * np.sqrt(variance)) * variance_prime
    
    contrib = np.sqrt(252) * z * np.sum(variance_matrix,axis=1) / np.sqrt(variance)
    contributions = pd.Series(contrib, index=valid_tickers)

    return (
        VaR,
        pd.Series(contributions, index=valid_tickers, name="Contributions"),
        pd.Series(VaR_prime, index=valid_tickers, name="dVaR/dExposure"),
        pd.DataFrame(corr, index=valid_tickers, columns=valid_tickers),
        pd.DataFrame(np.sqrt(252) * z * variance_matrix / np.sqrt(variance), index=valid_tickers, columns=valid_tickers),
        pd.DataFrame(variance_prime_matrix, index=valid_tickers, columns=valid_tickers)
    )

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute portfolio VaR")
    parser.add_argument("--csv", default="positions.csv", help="Path to positions CSV")
    parser.add_argument("--period", default="5y", help="History period for returns")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Confidence level for VaR",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of tickers for demo",
    )
    args = parser.parse_args()

    var,contributions,var_prime,corr,contrib_matrix,var_matrix_prime = portfolio_var(
        args.csv, period=args.period, confidence=args.confidence, limit=args.limit
    )
    print(f"Portfolio {int(args.confidence*100)}% VaR: ${var:,.2f}")
    print("\nVaR contribution by position:")
    print(contributions.sort_values(ascending=False))
    print("\nDerivative of VaR w.r.t. exposure:")
    print(var_prime.sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(contrib_matrix, cmap="viridis")
    ax.set_xticks(range(len(contrib_matrix.columns)))
    ax.set_xticklabels(contrib_matrix.columns, rotation=90)
    ax.set_yticks(range(len(contrib_matrix.index)))
    ax.set_yticklabels(contrib_matrix.index)
    fig.colorbar(im, ax=ax, label="Contribution to VaR")
    ax.set_title("Contribution matrix")
    plt.tight_layout()
    plt.savefig("contribution_matrix.png",dpi=1000)
    print("Contribution matrix heatmap saved to contribution_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(var_matrix_prime, cmap="viridis")
    ax.set_xticks(range(len(var_matrix_prime.columns)))
    ax.set_xticklabels(var_matrix_prime.columns, rotation=90)
    ax.set_yticks(range(len(var_matrix_prime.index)))
    ax.set_yticklabels(var_matrix_prime.index)
    fig.colorbar(im, ax=ax, label="Σ_ij * E_j")
    ax.set_title("VaR deriv matrix")
    plt.tight_layout()
    plt.savefig("derivative_matrix.png.png",dpi=1000)
    print("\nDerivative matrix heatmap saved to derivative_matrix.png")

    # Row-normalized matrix to show percentage attribution per asset
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr,vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Position Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png",dpi=1000)
    print("correlation matrix heatmap saved to correlation_matrix.png")


if __name__ == "__main__":
    # Show all rows
    pd.set_option("display.max_rows", None)
    main()

