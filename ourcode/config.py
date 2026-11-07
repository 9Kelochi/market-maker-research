"""Shared constants so every runner/plot uses identical settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "independent_ppo"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]
N_PPO = 3

# Market opens 09:30. PPO agents start at +5 min, so with the shock at +20 min
#  ~15 min pre-shock and ~30 min post-shock.
MKT_OPEN_FALLBACK = "2021-02-05 09:30:00"   # rmsc04 default date verified at runtime
END_TIME = "10:20:00"
SHOCK_OFFSET = "00:20:00"                    # after market open
SHOCK_SECONDS = 20 * 60
SHOCK_SIZE = 5000
SHOCK_SIDE = "ASK"                           # sell shock

PPO_FIRST_INTERVAL = "00:05:00"
PPO_STEP = "60s"
OBSERVER_FIRST_INTERVAL = "00:02:00"
OBSERVER_STEP = "5s"                       

BASELINE_WINDOW_S = 600                      # pre-shock window used as baseline
POST_HORIZON_S = 20 * 60                     # integrate excess spread this long