import pandas as pd

signals = pd.read_csv("volatility_signals.csv", parse_dates=["Date"])
signals = signals.sort_values("Date")

MODEL = "ridge"          # or "ewma", "garch", "har_rv", etc.
target = 0.10            # 10% annual vol target
cap = 1.0                # 1.0 = no leverage. set >1 only if you can/choose to lever.
floor = 0.05             # prevents extreme lever-up
equity = 10_000          # <-- your account equity in dollars

last = signals.dropna(subset=[MODEL]).iloc[-1]
sigma = float(last[MODEL])          # annualized vol forecast
price = float(last["Close"])        # last close

exposure_mult = min(cap, target / max(sigma, floor))
desired_exposure = equity * exposure_mult
desired_shares = desired_exposure / price

print("Signal date (computed after close):", last["Date"].date())
print("Model:", MODEL)
print("Forecast vol (ann):", round(sigma, 4))
print("Exposure multiplier for NEXT session:", round(exposure_mult, 3))
print("Desired $ exposure:", round(desired_exposure, 2))
print("Desired shares:", round(desired_shares, 4))