# High-Performance Limit Order Book Engine

A from-scratch central limit order book (CLOB) with self-exciting order arrival modeling and market microstructure analysis — the core infrastructure of every electronic exchange and quant trading firm.

## What This Is

Electronic markets process millions of orders per second through a data structure called a **limit order book** (LOB). Every time you place a stock order, it enters an LOB. This project implements one from scratch, then layers on the statistical machinery used by quant researchers to understand *how* markets work — not just simulate them.

**The project has five interconnected components:**

| Component | What it does |
|---|---|
| `engine/` | Price-time priority matching engine (the core) |
| `hawkes/` | Self-exciting point process for order arrival modeling |
| `microstructure/` | Five academic metrics: Kyle's λ, VPIN, OFI, Amihud, Roll |
| `backtest/` | Event-driven strategy backtester with two pluggable strategies |
| `visualization/` | Interactive Plotly Dash dashboard |

---

## Core Concepts

### 1. The Matching Engine (`engine/`)

A limit order book maintains two sorted queues of resting orders:

- **Bids** — buyers waiting at prices below the current ask (sorted descending)
- **Asks** — sellers waiting at prices above the current bid (sorted ascending)

When a new order arrives, it either *rests* (if it doesn't cross the spread) or *matches* against the opposite side. Matching follows **price-time priority**: best price first, then submission order among orders at the same price.

**Data structure choices:**
- Bids: `SortedDict` keyed by `−price` → O(log P) insertion, O(1) best-bid lookup
- Asks: `SortedDict` keyed by `+price` → same
- Per price level: `deque` of orders in FIFO order → O(1) amortized pop
- Order cancellation: flat `dict[order_id → Order]` → O(1) lookup

**Complexity:**
```
add_limit_order  O(log P)   P = distinct price levels
cancel_order     O(1)       hash lookup + lazy deque cleanup
best_bid/ask     O(1)
match            O(k)       k = orders consumed
```

### 2. Hawkes Process (`hawkes/`)

Real order arrivals are not Poisson — they **cluster**. A burst of buys begets more buys. This is captured by a *self-exciting point process* known as the Hawkes process (Hawkes 1971).

The conditional intensity function — the instantaneous rate of order arrivals given history — is:

```
λ*(t) = μ + Σ_{tᵢ < t} α · exp(−β · (t − tᵢ))
```

Where:
- **μ** = baseline arrival rate (events/second)
- **α** = excitation: how much each event amplifies the rate
- **β** = decay: how quickly the excitation fades
- **n = α/β** = branching ratio (must be < 1 for stationarity)

The branching ratio has an intuitive meaning: each event spawns on average `n` descendant events. At `n = 0` the process degenerates to Poisson; as `n → 1` the process becomes critical and highly clustered — matching observations in real LOB data (Bacry et al. 2015).

**Implementation:**
- **Simulation**: Ogata's (1981) modified thinning algorithm — O(n) amortized, no rejection overhead beyond `α/β`
- **Fitting**: Maximum likelihood via L-BFGS-B with multiple random restarts. The log-likelihood is evaluated in O(n) using the recursion `A(i) = exp(−β Δt) · (1 + A(i−1))`
- **Extension**: Full bivariate Hawkes for bid/ask sides — bids can excite asks (market-maker response) and vice versa

### 3. Market Microstructure Metrics (`microstructure/`)

#### Kyle's Lambda (Kyle 1985)
Measures price impact and adverse selection via OLS regression:
```
ΔP_t = λ · Q_t + ε_t
```
Where `ΔP_t` is the midprice change and `Q_t` is net signed order flow. Higher λ means more informed trading — each unit of order flow moves the price more, because market makers widen their spread to protect against informed traders.

#### VPIN — Volume-Synchronized Probability of Informed Trading (Easley et al. 2012)
Classifies volume into equal-size buckets and measures the imbalance within each:
```
VPIN = E[|V_buy − V_sell|] / V_bucket
```
VPIN ≈ 1 signals one-sided (potentially informed) flow. VPIN was famously used to predict the 2010 Flash Crash.

#### Order Flow Imbalance (Cont, Kukanov, Stoikov 2014)
Captures the net change in liquidity supply at the top of the book:
```
OFI_t = ΔBid_volume − ΔAsk_volume
```
Empirically, OFI explains 60–80% of short-horizon price variance — more than any other single variable.

#### Amihud Illiquidity Ratio (Amihud 2002)
```
ILLIQ = E[|R_t| / Vol_t]
```
Price impact per unit of dollar volume. Higher ILLIQ = less liquid market.

#### Roll's Spread Estimator (Roll 1984)
Recovers the implied bid-ask spread purely from the serial covariance of price changes — no trade direction data needed:
```
Roll Spread = 2√(−Cov(ΔP_t, ΔP_{t−1}))
```

---

## Project Structure

```
.
├── engine/
│   ├── order.py          # Order, Trade dataclasses
│   └── order_book.py     # CLOB matching engine (PriceLevel, OrderBook)
├── hawkes/
│   └── process.py        # simulate(), fit(), intensity(), simulate_bivariate()
├── microstructure/
│   └── metrics.py        # kyle_lambda(), vpin(), order_flow_imbalance(),
│                         # amihud_illiquidity(), roll_spread(), bin_trades()
├── feed/
│   └── simulator.py      # MarketSimulator — Hawkes → Orders → LOB
├── backtest/
│   └── engine.py         # BacktestEngine, MarketMaker, MomentumTrader
├── visualization/
│   └── dashboard.py      # Plotly Dash multi-panel dashboard
├── benchmarks/
│   └── benchmark_lob.py  # Latency / throughput benchmarks
├── tests/
│   ├── test_engine.py
│   ├── test_microstructure.py
│   └── test_hawkes.py
└── run_demo.py           # End-to-end demo with all components
```

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: install Numba for JIT-compiled Hawkes likelihood (~5× faster fitting)
pip install numba

# Run the full demo (30-min simulation, Hawkes fitting, metrics, dashboard)
python run_demo.py

# Shorter run, no browser dashboard
python run_demo.py --duration 300 --no-dashboard

# Run benchmarks
python benchmarks/benchmark_lob.py

# Run tests
pytest tests/ -v
```

---

## Performance

Benchmarks on a single core (Python reference implementation):

| Operation | Throughput | p50 latency | p99 latency |
|---|---|---|---|
| `add_limit_order` (resting) | ~400,000 /s | ~2 µs | ~8 µs |
| `cancel_order` | ~600,000 /s | ~1.5 µs | ~5 µs |
| `add_limit_order` (matched) | ~200,000 /s | ~4 µs | ~15 µs |
| Mixed workload | ~300,000 /s | — | — |

> **Note:** Production LOB engines (Jane Street, Citadel, NYSE) are implemented in C++/Rust and operate at nanosecond latency. This Python implementation is a validated reference — the microstructure models and statistical machinery run in the same language regardless of the underlying matching engine.

---

## Dashboard

Running `python run_demo.py` launches an interactive dashboard at `http://localhost:8050` with:

- **Order book depth** — bid/ask ladder at simulation end
- **Mid-price & spread time series** — from LOB snapshots
- **Hawkes conditional intensity** — λ*(t) with event rug plots
- **Kyle's λ regression** — scatter of (signed volume, ΔP) with OLS fit
- **Rolling VPIN** — toxicity over volume buckets
- **Strategy P&L** — mark-to-market for MarketMaker and MomentumTrader

---

## References

- Hawkes, A.G. (1971). "Spectra of some self-exciting and mutually exciting point processes." *Biometrika*.
- Ogata, Y. (1981). "On Lewis' simulation method for point processes." *IEEE Transactions on Information Theory*.
- Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*.
- Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread." *Journal of Finance*.
- Amihud, Y. (2002). "Illiquidity and Stock Returns." *Journal of Financial Markets*.
- Easley, D., Lopez de Prado, M., O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High Frequency World." *Review of Financial Studies*.
- Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order Book Events." *Journal of Financial Econometrics*.
- Bacry, E., Mastromatteo, I., Muzy, J.F. (2015). "Hawkes Processes in Finance." *Market Microstructure and Liquidity*.
