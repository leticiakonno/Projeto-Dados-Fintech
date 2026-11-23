"""Configuração central do projeto. Todos os scripts importam daqui,
para que janela de observação, semente e caminhos fiquem consistentes."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
for _d in (RAW_DIR, PROC_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_USERS = 100_000

# Janela de observação (UTC): 90 dias de logs ANTES da data de corte.
WINDOW_START = pd.Timestamp("2025-07-01")
WINDOW_DAYS = 90
RECENT_DAYS = 30
SNAPSHOT = WINDOW_START + pd.Timedelta(days=WINDOW_DAYS)  # data de corte

# Alvo: cliente cancelou nos 30 dias SEGUINTES à data de corte.
CHURN_HORIZON_DAYS = 30
TARGET_CHURN_RATE = 0.085

PLANS = ["Essencial", "Profissional", "Empresarial"]
PLAN_MRR = {"Essencial": 99, "Profissional": 299, "Empresarial": 899}  # R$/mês
CHANNELS = ["Orgânico", "Indicação", "Anúncios pagos", "Parceiros"]

ACTION_ERROR = "error_api_500"
ACTION_SUPPORT = "open_support"
ACTIONS_USAGE = ["login", "view_dashboard", "create_transaction", "export_report", "update_settings"]

# Faixas de risco (probabilidade prevista de churn em 30 dias)
RISK_MED = 0.12
RISK_HIGH = 0.30

# Cenário "what-if": limite de erros de API em 30 dias usado na simulação de estabilidade
CF_ERROR_CAP = 2
