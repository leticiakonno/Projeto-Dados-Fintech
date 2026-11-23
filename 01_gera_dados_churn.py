"""Etapa 1 - Geração dos dados BRUTOS (sintéticos, mas com estrutura realista).

Diferenças em relação à versão anterior:
  * O churn NÃO é sorteado por "perfil". Primeiro geramos o comportamento (logs
    de uso, erros de API, chamados) e só depois o cancelamento é sorteado a
    partir de uma função de risco que depende desse comportamento + ruído.
  * Há ruído e sobreposição: nem todo cliente com erros cancela, e nem todo
    cancelamento tem causa visível (efeito não observado).
  * Idade e renda NÃO influenciam o churn (como costuma ocorrer em B2B/SaaS).
  * Os arquivos saem "sujos" de propósito (nulos, duplicatas, órfãos, fusos),
    para que a etapa de limpeza tenha trabalho real a fazer.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config import *

rng = np.random.default_rng(SEED)
n = N_USERS
user_id = np.arange(1, n + 1)

# ----------------------------------------------------------------- perfil
plan = rng.choice(PLANS, n, p=[0.55, 0.33, 0.12])
plan_idx = pd.Series(plan).map({p: i for i, p in enumerate(PLANS)}).to_numpy()
channel = rng.choice(CHANNELS, n, p=[0.35, 0.20, 0.30, 0.15])

tenure_start = np.clip(rng.gamma(1.6, 220, n), 30, 1100).astype(int)  # dias no início da janela
signup_date = WINDOW_START - pd.to_timedelta(tenure_start, unit="D")

age = np.clip(rng.normal(38, 11, n), 18, 75).round().astype(int)
income = np.clip(
    rng.lognormal(np.log(5200), 0.45, n) * np.array([1.0, 1.25, 1.6])[plan_idx], 1800, 40000
).round(2)

# ------------------------------------------------------------ comportamento
base_rate = np.array([0.18, 0.36, 0.72])[plan_idx]  # eventos de uso por dia
engagement = rng.lognormal(0, 0.6, n)  # heterogeneidade entre clientes

# Taxa de erro de API por cliente (cauda longa: poucos clientes sofrem muito)
err_rate = rng.gamma(1.2, 0.05, n) * np.array([0.9, 1.0, 1.1])[plan_idx]

# Tendência de uso: d > 0 concentra o uso no início da janela (uso em queda).
# Quem sofre mais erros tende a reduzir o uso (frustração), mas há muita variação própria.
trend = np.clip(rng.normal(0.15, 0.5, n) + 5 * (err_rate - err_rate.mean()), -1.5, 3)

n_usage = rng.poisson(base_rate * engagement * WINDOW_DAYS)
n_err = rng.poisson(n_usage * err_rate)
n_sup = rng.poisson(0.15 + 0.30 * n_err)

HOUR_P = np.array([1, 1, 1, 1, 1, 2, 4, 8, 12, 14, 14, 13, 11, 12, 13, 13, 12, 10, 7, 5, 4, 3, 2, 1], float)
HOUR_P /= HOUR_P.sum()


def to_timestamps(t_days):
    day = np.floor(t_days).clip(0, WINDOW_DAYS - 1).astype(int)
    hour = rng.choice(24, size=len(t_days), p=HOUR_P)
    sec = rng.integers(0, 3600, len(t_days))
    return pd.Series(WINDOW_START + pd.to_timedelta(day * 86400 + hour * 3600 + sec, unit="s"))


# eventos de uso (distribuição ao longo do tempo depende da tendência)
uid_u = np.repeat(user_id, n_usage)
d = np.repeat(trend, n_usage)
u = rng.random(len(uid_u))
small = np.abs(d) < 1e-3
d_safe = np.where(small, 1.0, d)
t_usage = np.where(small, u * WINDOW_DAYS, -(WINDOW_DAYS / d_safe) * np.log1p(-u * (1 - np.exp(-d_safe))))
usage = pd.DataFrame(
    {"user_id": uid_u, "action": rng.choice(ACTIONS_USAGE, len(uid_u), p=[0.28, 0.34, 0.20, 0.12, 0.06]),
     "ts": to_timestamps(t_usage)}
)

# erros e chamados ao suporte: um pouco mais concentrados nas últimas semanas
uid_e = np.repeat(user_id, n_err)
errors = pd.DataFrame({"user_id": uid_e, "action": ACTION_ERROR,
                       "ts": to_timestamps(WINDOW_DAYS * rng.random(len(uid_e)) ** 0.6)})
uid_s = np.repeat(user_id, n_sup)
support = pd.DataFrame({"user_id": uid_s, "action": ACTION_SUPPORT,
                        "ts": to_timestamps(WINDOW_DAYS * rng.random(len(uid_s)) ** 0.6)})

logs = pd.concat([usage, errors, support], ignore_index=True)

# --------------------------------------------- churn (a partir do comportamento)
days_ago = (SNAPSHOT - logs["ts"]).dt.total_seconds() / 86400
recent = days_ago <= RECENT_DAYS
is_usage = ~logs["action"].isin([ACTION_ERROR, ACTION_SUPPORT])


def per_user(mask, how="size"):
    g = logs.loc[mask].groupby("user_id")
    s = g.size() if how == "size" else g["days_ago"].min()
    return s.reindex(user_id).to_numpy()


logs["days_ago"] = days_ago
ev30 = np.nan_to_num(per_user(is_usage & recent))
evprev = np.nan_to_num(per_user(is_usage & ~recent))
err30 = np.nan_to_num(per_user((logs["action"] == ACTION_ERROR) & recent))
sup90 = np.nan_to_num(per_user(logs["action"] == ACTION_SUPPORT))
last_days = np.nan_to_num(per_user(is_usage, how="min"), nan=WINDOW_DAYS)
log_trend = np.clip(np.log(((ev30 + 1) / 30) / ((evprev + 1) / 60)), -2, 2)

plan_effect = np.array([0.35, 0.0, -0.6])[plan_idx]
tenure_effect = np.where(tenure_start + WINDOW_DAYS < 150, 0.5, 0.0)
channel_effect = np.where(channel == "Anúncios pagos", 0.25, 0.0)
unobserved = rng.normal(0, 0.8, n)  # motivos que não aparecem nos dados

z0 = (0.50 * np.minimum(err30, 8) + 0.05 * np.minimum(last_days, 21) - 0.45 * log_trend
      + 0.12 * np.minimum(sup90, 6) + plan_effect + tenure_effect + channel_effect + unobserved)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
b0 = brentq(lambda b: sigmoid(z0 + b).mean() - TARGET_CHURN_RATE, -15, 5)
churn = (rng.random(n) < sigmoid(z0 + b0)).astype(int)

# ------------------------------------------------ "sujando" os dados de propósito
# Usuários: nulos, outliers de digitação, datas em dois formatos e duplicatas
age_raw = age.astype(float)
age_raw[rng.random(n) < 0.04] = np.nan
typo = rng.random(n) < 0.0015
age_raw[typo] = rng.choice([0, 150, 999], typo.sum())
income_raw = income.copy()
income_raw[rng.random(n) < 0.06] = np.nan
alt_fmt = rng.random(n) < 0.05
signup_str = np.where(alt_fmt, signup_date.strftime("%d/%m/%Y"), signup_date.strftime("%Y-%m-%d"))

users = pd.DataFrame({"user_id": user_id, "signup_date": signup_str, "age": age_raw, "income": income_raw,
                      "plan": plan, "acquisition_channel": channel, "churn": churn})
users = pd.concat([users, users.sample(frac=0.005, random_state=SEED)]).sort_values("user_id", kind="stable")

# Logs: fuso horário, caixa/espaços, órfãos e duplicatas
logs = logs.drop(columns="days_ago")
orph = logs.sample(frac=0.008, random_state=SEED).copy()
orph["user_id"] = rng.integers(n + 1, n + 8000, len(orph))
logs = pd.concat([logs, orph], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
logs.insert(0, "log_id", np.arange(1, len(logs) + 1))

tz = rng.choice(["America/Sao_Paulo", "UTC", "Europe/Lisbon"], len(logs), p=[0.80, 0.15, 0.05])
local = pd.Series(index=logs.index, dtype=object)
for name in np.unique(tz):
    m = tz == name
    conv = logs.loc[m, "ts"].dt.tz_localize("UTC").dt.tz_convert(name).dt.tz_localize(None)
    local.loc[m] = conv.dt.strftime("%Y-%m-%d %H:%M:%S")
action = logs["action"].to_numpy(dtype=object)
m1 = rng.random(len(logs)) < 0.01
action[m1] = np.char.upper(action[m1].astype(str))
m2 = rng.random(len(logs)) < 0.01
action[m2] = np.char.add(" ", action[m2].astype(str))
out_logs = pd.DataFrame({"log_id": logs["log_id"], "user_id": logs["user_id"], "action": action,
                         "timestamp": local, "timezone_tag": tz})
out_logs = pd.concat([out_logs, out_logs.sample(frac=0.004, random_state=SEED)]).sort_values("log_id", kind="stable")

users.to_csv(RAW_DIR / "users_profile.csv", index=False)
out_logs.to_csv(RAW_DIR / "app_logs.csv", index=False)

print(f"Usuários: {len(users):,} linhas | churn em {CHURN_HORIZON_DAYS} dias: {churn.mean():.1%}")
print(f"Logs: {len(out_logs):,} linhas | janela {WINDOW_START.date()} a {SNAPSHOT.date()} (UTC)")
print(f"Arquivos brutos salvos em {RAW_DIR}")
