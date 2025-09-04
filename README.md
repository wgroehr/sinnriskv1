# Portfolio VaR Tool — README (Markdown)

Compute portfolio Value at Risk (VaR) from a positions CSV, decompose it by asset, and visualize how names and pairs drive risk.

---

## What this tool does

Given a CSV of positions (tickers + **dollar exposures**), the tool:

1. Downloads historical prices with `yfinance`, builds **daily returns**, then computes **covariance** (Σ) and **correlation** (R).
2. Computes:
   - **Total VaR** at a chosen confidence.
   - **Per-asset VaR contributions** (Euler allocation).
   - **Sensitivity of VaR to each exposure** dVaR/dE
   - **Pairwise contribution matrix** that sums (by rows) to per-asset contributions, and over all cells to total VaR.
   - **Derivative (variance) matrix** that sums (by columns) to the variance gradient.
3. Prints the key numbers and saves three heatmaps:
   - `contribution_matrix.png`
   - `derivative_matrix.png`
   - `correlation_matrix.png`

---

## Install

```bash
pip install numpy pandas matplotlib yfinance
