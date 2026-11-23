"""Painel de risco de churn - visual corporativo.
Execute com:  streamlit run app/app_dashboard.py
Pré-requisito: rodar o pipeline em src/ (01 a 04).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import OUT_DIR, RISK_HIGH, RISK_MED  # noqa: E402

# ------------------------------------------------------------------ identidade visual
INK, SLATE, MUTED, LINE = "#0E1A2B", "#475467", "#98A2B3", "#E4E7EC"
BLUE, BAR, RED, AMBER, GREEN = "#1F4E8C", "#B8C6D9", "#B42318", "#B54708", "#067647"
FONT = "IBM Plex Sans, Segoe UI, Arial, sans-serif"

st.set_page_config(page_title="Risco de churn | Painel executivo", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [class*="st-"] {{ font-family: {FONT}; }}
.stApp {{ background: #F5F6F8; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1360px; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
.topbar {{ display:flex; justify-content:space-between; align-items:baseline; gap:16px; flex-wrap:wrap;
          border-bottom:1px solid {LINE}; padding-bottom:14px; margin-bottom:20px; }}
.topbar h1 {{ font-size:22px; font-weight:600; margin:0; letter-spacing:-0.01em; }}
.topbar .meta {{ font-size:13px; color:{SLATE}; }}
.kpis {{ display:grid; grid-template-columns:repeat(5, 1fr); background:#fff; border:1px solid {LINE};
        border-radius:6px; margin-bottom:16px; }}
.kpi {{ padding:16px 20px; border-right:1px solid {LINE}; }}
.kpi:last-child {{ border-right:none; }}
.kpi .label {{ font-size:13px; color:{SLATE}; }}
.kpi .value {{ font-size:28px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }}
.kpi .note {{ font-size:12px; color:#667085; margin-top:2px; }}
.kpis.four {{ grid-template-columns:repeat(4, 1fr); }}
@media (max-width: 900px) {{ .kpis, .kpis.four {{ grid-template-columns:1fr 1fr; }} .kpi {{ border-bottom:1px solid {LINE}; }} }}
.finding {{ background:#fff; border:1px solid {LINE}; border-left:3px solid {BLUE}; border-radius:6px;
           padding:14px 18px; font-size:15px; line-height:1.5; margin-bottom:16px; }}
.panel-title {{ font-size:15px; font-weight:600; margin:0 0 2px 0; }}
.panel-sub {{ font-size:12.5px; color:#667085; margin:0 0 6px 0; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-color:{LINE} !important; border-radius:6px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:500; color:{SLATE}; padding:8px 14px; }}
.stTabs [aria-selected="true"] {{ color:{BLUE}; }}
.small {{ font-size:12.5px; color:#667085; }}
</style>
""", unsafe_allow_html=True)

LABELS = {"api_errors_30d": "Erros de API (30 dias)", "api_errors_prev_60d": "Erros de API (60 dias anteriores)",
          "days_since_last_activity": "Dias sem uso", "activity_trend": "Tendência de uso",
          "support_calls_90d": "Chamados ao suporte", "events_30d": "Volume de uso (30 dias)",
          "tenure_days": "Tempo de casa", "plan": "Plano", "acquisition_channel": "Canal de aquisição",
          "age": "Idade", "income": "Renda"}


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:  # versões antigas do Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def show_df(data, **kw):
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kw)


def num(v):
    return f"{int(round(v)):,}".replace(",", ".")


def brl(v):
    if v >= 1e6:
        return f"R$ {v / 1e6:.2f} mi".replace(".", ",")
    if v >= 1e3:
        return f"R$ {v / 1e3:.0f} mil"
    return f"R$ {v:.0f}"


def pct(v, d=1):
    return f"{v:.{d}f}%".replace(".", ",")


def kpis(items, four=False):
    cells = "".join(f'<div class="kpi"><div class="label">{l}</div><div class="value">{v}</div>'
                    f'<div class="note">{n}</div></div>' for l, v, n in items)
    st.markdown(f'<div class="kpis{" four" if four else ""}">{cells}</div>', unsafe_allow_html=True)


def panel_header(title, sub=""):
    st.markdown(f'<p class="panel-title">{title}</p><p class="panel-sub">{sub}</p>', unsafe_allow_html=True)


def style(fig, height=300):
    fig.update_layout(height=height, margin=dict(t=18, b=8, l=8, r=8), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family=FONT, size=12, color=SLATE), showlegend=False, bargap=0.35,
                      xaxis=dict(showgrid=False, linecolor=LINE, ticks="", title=None),
                      yaxis=dict(gridcolor="#EEF0F3", zeroline=False, ticks="", title=None))
    return fig


def rate_bar(data, bucket, ref, ymax=None, highlight=True, height=290, min_n=30):
    g = data.groupby(bucket, observed=True).agg(n=("churn", "size"), rate=("churn", "mean")).reset_index()
    g = g[g["n"] >= min_n].reset_index(drop=True)
    z = 1.96  # intervalo de confiança de 95% (Wilson)
    den = 1 + z * z / g["n"]
    centre = (g["rate"] + z * z / (2 * g["n"])) / den
    half = z * np.sqrt(g["rate"] * (1 - g["rate"]) / g["n"] + z * z / (4 * g["n"] ** 2)) / den
    g["lo"], g["hi"] = (centre - half) * 100, (centre + half) * 100
    g["rate"] *= 100
    colors = [BAR] * len(g)
    if highlight and len(g):
        colors[int(np.argmax(g["rate"].values))] = BLUE
    fig = go.Figure(go.Bar(x=g[bucket].astype(str), y=g["rate"], marker_color=colors,
                           text=[pct(r) for r in g["rate"]], textposition="outside", cliponaxis=False,
                           customdata=np.column_stack([g["n"], g["lo"], g["hi"]]),
                           hovertemplate="%{x}<br>Churn: %{y:.1f}% (IC 95%: %{customdata[1]:.1f}% a %{customdata[2]:.1f}%)<br>Clientes: %{customdata[0]:,}<extra></extra>"))
    fig.add_hline(y=ref, line_dash="dot", line_color=MUTED,
                  annotation_text=f"Média {pct(ref)}", annotation_position="top left",
                  annotation_font=dict(size=11, color=SLATE))
    top = ymax if ymax else max(g["rate"].max() * 1.25, ref * 1.5) if len(g) else 10
    style(fig, height).update_yaxes(range=[0, top], ticksuffix="%")
    return fig


# ------------------------------------------------------------------------ dados
@st.cache_data
def load():
    scores = pd.read_csv(OUT_DIR / "churn_scores.csv")
    metrics = json.loads((OUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    return scores, metrics


try:
    base, M = load()
except FileNotFoundError:
    st.error("Arquivos do modelo não encontrados em /outputs. Execute o pipeline na ordem: gera_dados_churn.py, "
             "limpeza_dados.py, feature_engineering.py e machine_learning_churn.py.")
    st.stop()

base["risk_tier"] = pd.Categorical(base["risk_tier"], ["Alto", "Médio", "Baixo"], ordered=True)

# ---------------------------------------------------------------------- filtros
st.sidebar.markdown("**Filtros**")
plans = st.sidebar.multiselect("Plano", ["Essencial", "Profissional", "Empresarial"],
                               default=["Essencial", "Profissional", "Empresarial"])
chans = st.sidebar.multiselect("Canal de aquisição", sorted(base["acquisition_channel"].unique()),
                               default=sorted(base["acquisition_channel"].unique()))
t_min, t_max = int(base["tenure_days"].min()), int(base["tenure_days"].max())
tenure = st.sidebar.slider("Tempo de casa (dias)", t_min, t_max, (t_min, t_max))
st.sidebar.markdown(f'<p class="small">Data de corte: {M["snapshot"]}. Janela de comportamento: '
                    f'{M["window_days"]} dias. Previsão: próximos {M["horizon_days"]} dias.</p>', unsafe_allow_html=True)

f = base[base["plan"].isin(plans) & base["acquisition_channel"].isin(chans)
         & base["tenure_days"].between(*tenure)].copy()
if f.empty:
    st.warning("Nenhum cliente atende aos filtros selecionados. Amplie a seleção na barra lateral.")
    st.stop()

ref = f["churn"].mean() * 100
high = f[f["risk_tier"] == "Alto"]
auc = M["models"]["gradient_boosting_cv5"]["roc_auc"]

st.markdown(f"""<div class="topbar"><h1>Risco de churn da carteira</h1>
<span class="meta">Base: {num(len(base))} clientes ativos em {M["snapshot"]} &nbsp;|&nbsp; Dados sintéticos para fins de portfólio</span></div>""",
            unsafe_allow_html=True)

kpis([("Clientes na seleção", num(len(f)), f"de {num(len(base))} na carteira"),
      ("Churn em 30 dias", pct(ref), "cancelamentos observados no período"),
      ("Clientes em risco alto", num(len(high)), f"{pct(len(high) / len(f) * 100)} da seleção"),
      ("Receita mensal em risco alto", brl(high["mrr"].sum()), f"{pct(high['mrr'].sum() / f['mrr'].sum() * 100)} da receita da seleção"),
      ("AUC do modelo", f"{auc:.2f}".replace(".", ","), "validação cruzada, 5 partes")])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visão executiva", "Fatores de churn", "Carteira em risco",
                                        "Simulador de retenção", "Metodologia"])

# ---------------------------------------------------------------- 1. executiva
with tab1:
    f["faixa_erros"] = pd.cut(f["api_errors_30d"], [-1, 0, 2, 4, np.inf], labels=["0", "1 a 2", "3 a 4", "5 ou mais"])
    top_feat = LABELS[M["importance"][0]["feature"]]
    hi_e = f[f["api_errors_30d"] >= 3]
    lo_e = f[f["api_errors_30d"] < 3]
    if len(hi_e) >= 100 and len(lo_e) >= 100 and f["churn"].sum() > 0:
        r_hi, r_lo = hi_e["churn"].mean() * 100, lo_e["churn"].mean() * 100
        share_base = len(hi_e) / len(f) * 100
        share_churn = hi_e["churn"].sum() / f["churn"].sum() * 100
        excess = hi_e["churn"].sum() - len(hi_e) * lo_e["churn"].mean()
        excess_mrr = excess * hi_e["mrr"].mean()
        d_n = (f["churn_probability"] - f["churn_probability_cf"]).sum()
        d_mrr = ((f["churn_probability"] - f["churn_probability_cf"]) * f["mrr"]).sum()
        d_pct = d_n / f["churn_probability"].sum() * 100
        ratio_txt = f"{r_hi / r_lo:.1f}".replace(".", ",")
        weak = f[f["activity_trend"] < 0.5]
        steady = f[f["activity_trend"].between(0.8, 1.2)]
        rows = [
            f"<b>Fator principal: {top_feat}.</b> Os clientes com 3 ou mais erros de API em 30 dias são apenas "
            f"<b>{pct(share_base)} da base</b> ({num(len(hi_e))} clientes), mas <b>{pct(share_churn)} de todos os cancelamentos</b> "
            f"da carteira vieram desse grupo.",
            f"<b>Risco dentro de cada grupo.</b> Entre os clientes com 3 ou mais erros, {pct(r_hi)} cancelaram. Entre os demais "
            f"{pct(100 - share_base)} da base, {pct(r_lo)} cancelaram. O risco do primeiro grupo é {ratio_txt} vezes maior.",
            f"<b>Peso no resultado.</b> Cerca de <b>{num(excess)} cancelamentos</b> ({pct(excess / f['churn'].sum() * 100)} do total de "
            f"cancelamentos da carteira) são excesso em relação ao que ocorreria se esse grupo tivesse o churn dos demais clientes. "
            f"Isso equivale a ~{brl(excess_mrr)} de receita mensal.",
            f"<b>Cenário de estabilidade.</b> Se nenhum cliente tivesse mais de 2 erros de API em 30 dias, o modelo estima "
            f"<b>{pct(d_pct)} menos cancelamentos</b> ({num(d_n)} clientes) e {brl(d_mrr)} de receita mensal preservada.",
        ]
        if len(weak) >= 100 and len(steady) >= 100:
            rows.append(f"<b>Segundo sinal: queda de uso.</b> Clientes com uso recente abaixo de metade do ritmo anterior cancelam "
                        f"em {pct(weak['churn'].mean() * 100)}, contra {pct(steady['churn'].mean() * 100)} dos estáveis.")
        st.markdown('<div class="finding">' + "<br><br>".join(rows) + "</div>", unsafe_allow_html=True)

    with st.container(border=True):
        panel_header("Importância de cada variável no modelo",
                     "Queda do AUC quando a variável é embaralhada no conjunto de teste. Quanto maior, mais o modelo depende dela")
        imp = pd.DataFrame(M["importance"]).sort_values("importance_mean")
        imp["nome"] = imp["feature"].map(LABELS)
        vals = imp["importance_mean"].clip(lower=0)
        fig = go.Figure(go.Bar(y=imp["nome"], x=vals, orientation="h",
                               marker_color=[BLUE if v == vals.max() else BAR for v in vals],
                               text=[f"{v:.3f}".replace(".", ",") for v in vals], textposition="outside", cliponaxis=False))
        style(fig, 380).update_xaxes(showgrid=True, gridcolor="#EEF0F3", range=[0, vals.max() * 1.15])
        fig.update_yaxes(showgrid=False)
        show(fig)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        panel_header("Churn por erros de API nos últimos 30 dias", "Taxa de cancelamento no mês seguinte, por faixa")
        show(rate_bar(f, "faixa_erros", ref))
    with c2, st.container(border=True):
        panel_header("Previsto × observado", "Probabilidade média prevista e churn real, por décimo da carteira")
        f["decil"] = pd.qcut(f["churn_probability"].rank(method="first"), 10, labels=False) + 1
        d = f.groupby("decil").agg(p=("churn_probability", "mean"), o=("churn", "mean")) * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d.index, y=d["o"], name="Observado", marker_color=BAR))
        fig.add_trace(go.Scatter(x=d.index, y=d["p"], name="Previsto", mode="lines+markers",
                                 line=dict(color=BLUE, width=2), marker=dict(size=6)))
        style(fig, 290).update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0))
        fig.update_xaxes(title="Décimo da carteira (1 = menor risco)", dtick=1).update_yaxes(ticksuffix="%")
        show(fig)

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        panel_header("Clientes por nível de risco", f"Alto: probabilidade ≥ {RISK_HIGH:.0%}. Médio: ≥ {RISK_MED:.0%}")
        t = (f.groupby("risk_tier", observed=True).agg(c=("user_id", "size"), ch=("churn", "mean"),
                                                        p=("churn_probability", "mean"), m=("mrr", "sum")).reset_index())
        t = pd.DataFrame({"Nível": t["risk_tier"].astype(str), "Clientes": t["c"].map(num),
                          "Previsto": (t["p"] * 100).map(pct), "Observado": (t["ch"] * 100).map(pct),
                          "Receita mensal": t["m"].map(brl)})
        show_df(t)
        st.markdown('<p class="small">Previsto e observado próximos em cada faixa indicam que as probabilidades são confiáveis.</p>',
                    unsafe_allow_html=True)
    with c4, st.container(border=True):
        panel_header("Receita mensal em risco alto, por plano", "Soma da receita recorrente dos clientes em risco alto")
        m = high.groupby("plan")["mrr"].sum().reindex(["Essencial", "Profissional", "Empresarial"]).fillna(0)
        fig = go.Figure(go.Bar(x=m.index, y=m.values / 1e3, marker_color=BLUE, text=[brl(v) for v in m.values],
                               textposition="outside", cliponaxis=False))
        style(fig, 280).update_yaxes(range=[0, max(m.max() / 1e3 * 1.25, 1)], ticksuffix=" mil")
        show(fig)

    with st.container(border=True):
        panel_header("Receita em risco por plano",
                     "Todos os planos lado a lado, sem precisar filtrar. Os valores respeitam os filtros da barra lateral")

        def plan_row(label, g):
            g_high = g[g["risk_tier"] == "Alto"]
            return {"Plano": label, "Clientes": num(len(g)), "Em risco alto": num(len(g_high)),
                    "Previsto": pct(g["churn_probability"].mean() * 100), "Observado": pct(g["churn"].mean() * 100),
                    "Receita mensal": brl(g["mrr"].sum()), "Receita em risco alto": brl(g_high["mrr"].sum()),
                    "% da receita em risco alto": pct(g_high["mrr"].sum() / g["mrr"].sum() * 100)}

        plan_rows = [plan_row(p, f[f["plan"] == p]) for p in ["Essencial", "Profissional", "Empresarial"] if (f["plan"] == p).any()]
        plan_rows.append(plan_row("Total", f))
        show_df(pd.DataFrame(plan_rows))

# ------------------------------------------------------------------ 2. fatores
with tab2:
    st.markdown("##### Sinais de comportamento")
    f["faixa_dias"] = pd.cut(f["days_since_last_activity"], [-1, 1, 3, 7, 14, 999],
                             labels=["até 1 dia", "2 a 3 dias", "4 a 7 dias", "8 a 14 dias", "mais de 14 dias"])
    f["faixa_tend"] = pd.cut(f["activity_trend"], [-1, 0.5, 0.8, 1.2, 99],
                             labels=["queda forte", "queda leve", "estável", "crescimento"])
    f["faixa_sup"] = pd.cut(f["support_calls_90d"], [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3 ou mais"])
    behavior = [("Dias desde o último uso", "faixa_dias"), ("Tendência de uso (30 dias vs. 60 anteriores)", "faixa_tend"),
                ("Chamados ao suporte em 90 dias", "faixa_sup")]
    f["faixa_errors_prev"] = pd.cut(f["api_errors_prev_60d"], [-1, 0, 2, 4, np.inf], labels=["0", "1 a 2", "3 a 4", "5 ou mais"])
    behavior.append(("Erros de API nos 60 dias anteriores", "faixa_errors_prev"))
    ymax = max(f.groupby(c, observed=True)["churn"].mean().max() for _, c in behavior) * 100 * 1.3
    cols = st.columns(2)
    for i, (title, col) in enumerate(behavior):
        with cols[i % 2], st.container(border=True):
            panel_header(title, "Taxa de churn por faixa")
            show(rate_bar(f, col, ref, ymax=ymax, height=260))

    st.markdown("##### Perfil do cliente")
    st.caption("Mesma escala de eixo dos gráficos acima, para comparação direta.")
    f["faixa_idade"] = pd.cut(f["age"], [17, 29, 39, 49, 99], labels=["18 a 29", "30 a 39", "40 a 49", "50 ou mais"])
    f["faixa_renda"] = pd.qcut(f["income"], 4, labels=["1º quartil", "2º quartil", "3º quartil", "4º quartil"])
    f["faixa_casa"] = pd.cut(f["tenure_days"], [0, 180, 365, 730, 9999], labels=["até 6 meses", "6 a 12 meses", "1 a 2 anos", "mais de 2 anos"])
    profile = [("Idade", "faixa_idade"), ("Renda (quartis)", "faixa_renda"), ("Tempo de casa", "faixa_casa"), ("Plano", "plan")]
    cols = st.columns(2)
    for i, (title, col) in enumerate(profile):
        with cols[i % 2], st.container(border=True):
            panel_header(title, "Taxa de churn por faixa")
            show(rate_bar(f, col, ref, ymax=ymax, highlight=False, height=240))

# ------------------------------------------------------------------- 3. carteira
with tab3:
    c1, c2 = st.columns([1, 3])
    tier = c1.radio("Nível de risco", ["Alto", "Médio", "Baixo", "Todos"], horizontal=False)
    t = f if tier == "Todos" else f[f["risk_tier"] == tier]
    t = t.assign(sinal=np.select(
        [t["api_errors_30d"] >= 3, t["days_since_last_activity"] >= 7, t["activity_trend"] < 0.5, t["support_calls_90d"] >= 3],
        ["Erros de API recorrentes", "Inatividade", "Queda de uso", "Muitos chamados ao suporte"],
        default="Combinação de fatores")).sort_values("churn_probability", ascending=False)
    c2.markdown(f"**{num(len(t))} clientes** | receita mensal {brl(t['mrr'].sum())}")
    view = t[["user_id", "plan", "acquisition_channel", "tenure_days", "api_errors_30d", "days_since_last_activity",
              "activity_trend", "support_calls_90d", "sinal", "mrr", "churn_probability"]].rename(columns={
        "user_id": "Cliente", "plan": "Plano", "acquisition_channel": "Canal", "tenure_days": "Dias de casa",
        "api_errors_30d": "Erros API (30d)", "days_since_last_activity": "Dias sem uso", "activity_trend": "Tendência de uso",
        "support_calls_90d": "Chamados (90d)", "sinal": "Sinal predominante", "mrr": "Receita mensal (R$)",
        "churn_probability": "Prob. de churn"})
    show_df(view.head(500), height=460, column_config={
        "Prob. de churn": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Receita mensal (R$)": st.column_config.NumberColumn(format="R$ %d")})
    st.caption("Exibe os 500 clientes de maior probabilidade. O download inclui todos os clientes do filtro. "
               "O sinal predominante é uma regra de apoio à abordagem, não uma explicação causal individual.")
    st.download_button("Baixar lista (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                       file_name="carteira_risco_churn.csv", mime="text/csv")

# ------------------------------------------------------------------ 4. simulador
with tab4:
    st.markdown("Estimativa do retorno de uma ação de retenção. Os parâmetros abaixo são **hipóteses editáveis**, "
                "não resultados medidos.")
    c1, c2, c3, c4 = st.columns(4)
    alvo = c1.selectbox("Público da ação", ["Risco alto", "Risco alto e médio"])
    sucesso = c2.slider("Taxa de sucesso da ação (%)", 5, 60, 20)
    custo = c3.number_input("Custo por conta abordada (R$)", 0, 2000, 80, step=10)
    meses = c4.slider("Meses de receita preservada", 1, 24, 12)
    pub = f[f["risk_tier"] == "Alto"] if alvo == "Risco alto" else f[f["risk_tier"].isin(["Alto", "Médio"])]
    esperados = pub["churn_probability"].sum()
    salvos = esperados * sucesso / 100
    receita = (pub["churn_probability"] * pub["mrr"]).sum() * sucesso / 100 * meses
    gasto = len(pub) * custo
    liquido = receita - gasto
    roi = liquido / gasto * 100 if gasto else 0
    kpis([("Contas abordadas", num(len(pub)), "público selecionado"),
          ("Cancelamentos evitados", num(salvos), f"de {num(esperados)} esperados no público"),
          ("Receita preservada", brl(receita), f"{meses} meses de receita recorrente"),
          ("Resultado líquido", brl(liquido), f"ROI de {pct(roi, 0)} sobre {brl(gasto)} investidos")], four=True)
    st.caption("Cancelamentos esperados = soma das probabilidades previstas. Receita preservada = probabilidade × receita mensal "
               "× taxa de sucesso × meses. A taxa de sucesso só pode ser confirmada com um teste controlado (A/B) real.")

# ---------------------------------------------------------------- 5. metodologia
with tab5:
    mm = M["models"]
    rows = [("Regra simples (3+ erros de API)", "regra_simples_3_erros"), ("Regressão logística", "regressao_logistica"),
            ("Gradient boosting", "gradient_boosting"), ("Gradient boosting, validação cruzada", "gradient_boosting_cv5")]
    tbl = pd.DataFrame([{"Modelo": n, "AUC": mm[k]["roc_auc"], "AUC-PR": mm[k]["pr_auc"],
                         "Precisão nos 10% de maior risco": mm[k]["precision_top10"],
                         "Cancelamentos capturados nos 10%": mm[k]["recall_top10"], "Lift nos 10%": mm[k]["lift_top10"]}
                        for n, k in rows])
    st.markdown(f"""
**Objetivo.** Identificar, na data de corte ({M["snapshot"]}), os clientes com maior probabilidade de cancelar nos
{M["horizon_days"]} dias seguintes, e quantificar o que mais influencia o cancelamento.

**Dados.** Base sintética de {num(M["n_customers"])} clientes e {num(M["n_events"] or 0)} eventos de uso, gerada para este portfólio. O cancelamento é
sorteado a partir do comportamento (erros de API, uso, chamados) somado a ruído, de modo que o modelo não alcança acerto perfeito.
Os dados de origem têm falhas deliberadas (nulos, duplicatas, registros órfãos, fusos horários) tratadas na etapa de limpeza.

**Variáveis.** Calculadas apenas com os {M["window_days"]} dias anteriores à data de corte: erros de API, chamados ao suporte, volume e tendência de uso,
dias desde o último uso, tempo de casa, plano, canal, idade e renda.

**Validação.** Métricas medidas em 20% dos clientes reservados para teste e em validação cruzada de 5 partes. A probabilidade
exibida para cada cliente vem de um modelo que não foi treinado com ele.
""")
    show_df(tbl, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["AUC", "AUC-PR", "Precisão nos 10% de maior risco",
                                                                                        "Cancelamentos capturados nos 10%"]} |
            {"Lift nos 10%": st.column_config.NumberColumn(format="%.1fx")})
    lim = pd.DataFrame(M["limiares"])
    st.markdown("**Classificação por limiar de decisão (conjunto de teste).** Limiar menor captura mais cancelamentos (recall maior), "
                "ao custo de mais alarmes falsos (precisão menor).")
    show_df(pd.DataFrame({"Limiar de probabilidade": [f"{t:.0%}" for t in lim["limiar"]],
                          "Clientes sinalizados": [pct(v * 100) for v in lim["pct_sinalizados"]],
                          "Precisão": [pct(v * 100) for v in lim["precisao"]], "Recall": [pct(v * 100) for v in lim["recall"]],
                          "F1-score": [f"{v:.2f}".replace(".", ",") for v in lim["f1"]]}))
    st.markdown("""
**Limitações.** Os resultados refletem as relações definidas na geração dos dados. Em dados reais, a associação entre erros de API
e cancelamento precisa ser confirmada com análise causal ou experimento. O cenário de estabilidade é uma estimativa do modelo (o que ele prevê se os erros fossem limitados a 2 em 30 dias). A simulação de retenção usa hipóteses, não efeitos medidos.
""")
