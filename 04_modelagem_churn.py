"""Etapa 4 - Modelagem, avaliação e explicação do churn.

Principais cuidados:
  * Métricas medidas em dados que o modelo NÃO viu (holdout 20% + validação cruzada de 5 folds).
  * Probabilidades salvas para cada cliente são "out-of-fold": o modelo que pontua um cliente
    nunca foi treinado com ele (a versão anterior pontuava a própria base de treino).
  * Importância medida por permutação no holdout (impacto real na capacidade de prever).
  * Compara o modelo com uma regra simples e com regressão logística (baseline).
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import *

df = pd.read_csv(PROC_DIR / "churn_modeling_ready.csv")
NUM = ["age", "income", "tenure_days", "events_30d", "activity_trend", "days_since_last_activity",
       "api_errors_30d", "api_errors_prev_60d", "support_calls_90d"]
CAT = ["plan", "acquisition_channel"]
X, y = df[NUM + CAT], df["churn"]


def make_lr():
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])


def make_gb():
    prep = ColumnTransformer([("num", "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def top_decile(y_true, p, frac=0.10):
    y_true = np.asarray(y_true)
    k = int(len(p) * frac)
    top = np.argsort(-np.asarray(p))[:k]
    precision = y_true[top].mean()
    return {"precision_top10": float(precision), "recall_top10": float(y_true[top].sum() / y_true.sum()),
            "lift_top10": float(precision / y_true.mean())}


def evaluate(y_true, p):
    return {"roc_auc": float(roc_auc_score(y_true, p)), "pr_auc": float(average_precision_score(y_true, p)),
            "brier": float(brier_score_loss(y_true, p)), **top_decile(y_true, p)}


# -------------------------------------------------------------- holdout
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Treino: {len(X_tr):,} | Teste: {len(X_te):,} | churn base: {y.mean():.1%}")

lr, gb = make_lr().fit(X_tr, y_tr), make_gb().fit(X_tr, y_tr)
res = {"regra_simples_3_erros": evaluate(y_te, (X_te["api_errors_30d"] >= 3).astype(float)),
       "regressao_logistica": evaluate(y_te, lr.predict_proba(X_te)[:, 1]),
       "gradient_boosting": evaluate(y_te, gb.predict_proba(X_te)[:, 1])}
table = pd.DataFrame(res).T
print("\n--- DESEMPENHO NO HOLDOUT ---")
print(table.round(3).to_string())

# ---------------- classificação por limiar de decisão (precisão, recall e F1)
p_te = gb.predict_proba(X_te)[:, 1]
thr_rows = []
for thr in (0.15, RISK_HIGH, 0.50):
    pred = (p_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    thr_rows.append({"limiar": float(thr), "pct_sinalizados": float(pred.mean()),
                     "precisao": float(precision_score(y_te, pred)), "recall": float(recall_score(y_te, pred)),
                     "f1": float(f1_score(y_te, pred)), "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn)})
print("\n--- CLASSIFICAÇÃO POR LIMIAR (HOLDOUT, GRADIENT BOOSTING) ---")
print(pd.DataFrame(thr_rows).round(3).to_string(index=False))

# ------------------------------------------- AUC de cada variável isolada
uni = {c: float(max(roc_auc_score(y, df[c]), 1 - roc_auc_score(y, df[c]))) for c in NUM}
print("\n--- AUC DE CADA VARIÁVEL ISOLADA (0,5 = nenhuma informação) ---")
for k, v in sorted(uni.items(), key=lambda kv: -kv[1]):
    print(f"{k:<28}{v:.3f}")

# ------------------------------------------- importância por permutação
perm = permutation_importance(gb, X_te, y_te, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1)
imp = (pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean,
                     "importance_std": perm.importances_std})
       .sort_values("importance_mean", ascending=False).reset_index(drop=True))
print("\n--- QUEDA DE AUC AO EMBARALHAR A VARIÁVEL (maior = mais importante) ---")
print(imp.round(4).to_string(index=False))

# ------------------------------------------- churn por faixa de erros
bins = pd.cut(df["api_errors_30d"], [-1, 0, 2, 4, np.inf], labels=["0", "1-2", "3-4", "5+"])
by_err = df.groupby(bins, observed=True)["churn"].agg(["size", "mean"])
print("\n--- CHURN POR ERROS DE API NOS ÚLTIMOS 30 DIAS ---")
print((by_err.assign(mean=(by_err["mean"] * 100).round(1))).rename(columns={"size": "clientes", "mean": "churn_%"}).to_string())

# ---------------------------- pontuação out-of-fold para toda a carteira
# Também calcula o cenário "what-if": e se ninguém tivesse mais que CF_ERROR_CAP erros em 30 dias?
X_cf = X.copy()
X_cf["api_errors_30d"] = X_cf["api_errors_30d"].clip(upper=CF_ERROR_CAP)
oof, oof_cf = np.zeros(len(df)), np.zeros(len(df))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
    m = make_gb().fit(X.iloc[tr], y.iloc[tr])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    oof_cf[va] = m.predict_proba(X_cf.iloc[va])[:, 1]
res["gradient_boosting_cv5"] = evaluate(y, oof)
print(f"\nROC-AUC (validação cruzada 5 folds): {res['gradient_boosting_cv5']['roc_auc']:.3f}")

scores = df.copy()
scores["mrr"] = scores["plan"].map(PLAN_MRR)
scores["churn_probability"] = oof
scores["churn_probability_cf"] = oof_cf
scores["risk_tier"] = pd.cut(oof, [-0.01, RISK_MED, RISK_HIGH, 1.01], labels=["Baixo", "Médio", "Alto"], right=False)
scores.to_csv(OUT_DIR / "churn_scores.csv", index=False)

stats_file = PROC_DIR / "stats.json"
n_events = json.loads(stats_file.read_text())["n_events"] if stats_file.exists() else None
metrics = {"n_events": n_events, "snapshot": str(SNAPSHOT.date()), "window_days": WINDOW_DAYS, "horizon_days": CHURN_HORIZON_DAYS,
           "n_customers": int(len(df)), "churn_rate": float(y.mean()), "models": res, "limiares": thr_rows,
           "univariate_auc": uni, "importance": imp.to_dict(orient="records"),
           "thresholds": {"medio": RISK_MED, "alto": RISK_HIGH}}
(OUT_DIR / "model_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n--- DISTRIBUIÇÃO POR FAIXA DE RISCO (out-of-fold) ---")
print(scores.groupby("risk_tier", observed=True).agg(clientes=("user_id", "size"), churn_observado=("churn", "mean"),
                                                     prob_media=("churn_probability", "mean")).round(3).to_string())
hi = df["api_errors_30d"] >= 3
base_rate = df.loc[~hi, "churn"].mean()
excess = df.loc[hi, "churn"].sum() - hi.sum() * base_rate
print("\n--- RESUMO EXECUTIVO ---")
print(f"Clientes com 3+ erros de API: {hi.mean():.1%} da base e {df.loc[hi, 'churn'].sum() / df['churn'].sum():.1%} dos cancelamentos")
print(f"Churn: {df.loc[hi, 'churn'].mean():.1%} (3+ erros) vs {base_rate:.1%} (até 2 erros)")
print(f"Cancelamentos em excesso associados ao grupo: {excess:,.0f} ({excess / df['churn'].sum():.1%} do total)")
print(f"Cenário de estabilidade: cancelamentos esperados {oof.sum():,.0f} -> {oof_cf.sum():,.0f} ({1 - oof_cf.sum() / oof.sum():.1%} a menos)")
print(f"\nArquivos salvos em {OUT_DIR}")
