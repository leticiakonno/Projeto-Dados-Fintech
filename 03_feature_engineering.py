"""Etapa 3 - Features comportamentais calculadas SOMENTE com dados anteriores à data de corte."""
import json

import numpy as np
import pandas as pd

from config import *

print("Carregando bases limpas...")
users = pd.read_csv(PROC_DIR / "users_profile_clean.csv", parse_dates=["signup_date"])
logs = pd.read_csv(PROC_DIR / "app_logs_clean.csv", parse_dates=["timestamp"])

logs = logs[(logs["timestamp"] >= WINDOW_START) & (logs["timestamp"] < SNAPSHOT)].copy()
logs["days_ago"] = (SNAPSHOT - logs["timestamp"]).dt.total_seconds() / 86400
logs["recent"] = logs["days_ago"] <= RECENT_DAYS

is_err = logs["action"] == ACTION_ERROR
is_sup = logs["action"] == ACTION_SUPPORT
is_use = ~(is_err | is_sup)
idx = users["user_id"]


def count(mask):
    return logs.loc[mask].groupby("user_id").size().reindex(idx).fillna(0).astype(int).to_numpy()


f = users.copy()
f["tenure_days"] = (SNAPSHOT - f["signup_date"]).dt.days
f["events_30d"] = count(is_use & logs["recent"])
f["events_prev_60d"] = count(is_use & ~logs["recent"])
# razão entre o ritmo diário recente e o ritmo dos 60 dias anteriores (1 = estável)
f["activity_trend"] = (
    (((f["events_30d"] + 1) / 30) / ((f["events_prev_60d"] + 1) / 60)).clip(upper=5).round(3)
)
f["days_since_last_activity"] = (
    logs.loc[is_use].groupby("user_id")["days_ago"].min().reindex(idx).fillna(WINDOW_DAYS).round(1).to_numpy()
)
f["api_errors_30d"] = count(is_err & logs["recent"])
f["api_errors_prev_60d"] = count(is_err & ~logs["recent"])
f["support_calls_90d"] = count(is_sup)

cols = ["user_id", "age", "income", "plan", "acquisition_channel", "tenure_days", "events_30d",
        "events_prev_60d", "activity_trend", "days_since_last_activity", "api_errors_30d",
        "api_errors_prev_60d", "support_calls_90d", "churn"]
f = f[cols]
assert f.isna().sum().sum() == 0, "Há valores nulos na base final"
f.to_csv(PROC_DIR / "churn_modeling_ready.csv", index=False)

(PROC_DIR / "stats.json").write_text(json.dumps({"n_events": int(len(logs))}))
print(f"\nBase final: {len(f):,} clientes | churn em {CHURN_HORIZON_DAYS} dias: {f['churn'].mean():.1%}")
print(f.describe().T[["mean", "50%", "max"]].round(2).to_string())
