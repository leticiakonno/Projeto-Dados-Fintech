"""Etapa 2 - Limpeza e validação dos dados brutos."""
import numpy as np
import pandas as pd

from config import *

print("Carregando dados brutos...")
users = pd.read_csv(RAW_DIR / "users_profile.csv")
logs = pd.read_csv(RAW_DIR / "app_logs.csv")
rep = {"usuarios_brutos": len(users), "logs_brutos": len(logs)}

# ---------------------------------------------------------------- usuários
users = users.drop_duplicates(subset="user_id", keep="first")
rep["usuarios_duplicados_removidos"] = rep["usuarios_brutos"] - len(users)

# datas em dois formatos (YYYY-MM-DD e DD/MM/YYYY)
iso = pd.to_datetime(users["signup_date"], format="%Y-%m-%d", errors="coerce")
br = pd.to_datetime(users["signup_date"], format="%d/%m/%Y", errors="coerce")
users["signup_date"] = iso.fillna(br)
rep["datas_invalidas_removidas"] = int(users["signup_date"].isna().sum())
users = users.dropna(subset=["signup_date"])

# idade: valores impossíveis viram nulos; nulos recebem a mediana
invalid_age = ~users["age"].between(18, 90) & users["age"].notna()
rep["idades_invalidas"] = int(invalid_age.sum())
users.loc[invalid_age, "age"] = np.nan
rep["idades_nulas"] = int(users["age"].isna().sum())
users["age"] = users["age"].fillna(users["age"].median()).round().astype(int)

# renda: nulos recebem a mediana do próprio plano (renda varia por plano)
rep["rendas_nulas"] = int(users["income"].isna().sum())
users["income"] = users["income"].fillna(users.groupby("plan")["income"].transform("median")).round(2)

# ------------------------------------------------------------------- logs
logs["action"] = logs["action"].str.strip().str.lower()
valid_actions = set(ACTIONS_USAGE + [ACTION_ERROR, ACTION_SUPPORT])
bad = ~logs["action"].isin(valid_actions)
rep["acoes_desconhecidas_removidas"] = int(bad.sum())
logs = logs[~bad]

n0 = len(logs)
logs = logs.drop_duplicates(subset="log_id")
rep["logs_duplicados_removidos"] = n0 - len(logs)

n0 = len(logs)
logs = logs[logs["user_id"].isin(users["user_id"])].copy()  # integridade referencial
rep["logs_orfaos_removidos"] = n0 - len(logs)

# timestamps locais -> UTC (cada linha tem o fuso em que foi gravada)
parsed = pd.to_datetime(logs["timestamp"], format="%Y-%m-%d %H:%M:%S")
parts = []
for tz, idx in parsed.groupby(logs["timezone_tag"]).groups.items():
    parts.append(
        parsed.loc[idx]
        .dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
logs["timestamp"] = pd.concat(parts)
rep["logs_horario_ambiguo_removidos"] = int(logs["timestamp"].isna().sum())
logs = logs.dropna(subset=["timestamp"])
logs = logs[(logs["timestamp"] >= WINDOW_START) & (logs["timestamp"] < SNAPSHOT)]
logs = logs.drop(columns="timezone_tag").sort_values("timestamp")

users.to_csv(PROC_DIR / "users_profile_clean.csv", index=False)
logs.to_csv(PROC_DIR / "app_logs_clean.csv", index=False)

print("\n--- RELATÓRIO DE QUALIDADE ---")
for k, v in rep.items():
    print(f"{k:<34}{v:>10,}")
print(f"\nBase final: {len(users):,} usuários | {len(logs):,} eventos (UTC)")
