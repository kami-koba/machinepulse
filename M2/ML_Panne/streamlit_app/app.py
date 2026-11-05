from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MachinePulse | Maintenance prédictive", page_icon="⚙️", layout="wide", initial_sidebar_state="collapsed")

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stApp"] {background:radial-gradient(circle at 80% 5%, #152e47 0%, #0b1424 38%, #080f1b 85%);color:#e8f1fc;}
.block-container {padding-top:2rem;max-width:1220px;padding-bottom:4rem;}
#MainMenu,footer,header {visibility:hidden;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.04em;}
.hero {padding:32px 36px;background:linear-gradient(110deg,rgba(21,44,70,.95),rgba(12,27,47,.85));border:1px solid #2b455f;border-radius:22px;margin-bottom:25px;position:relative;overflow:hidden;}
.eyebrow {font-size:12px;font-weight:700;letter-spacing:.2em;color:#4ce0cb;text-transform:uppercase;margin-bottom:12px;}
.hero h1 {font-size:clamp(32px,4vw,52px);line-height:1.08;margin:0 0 13px;color:#f4f8ff;}
.hero p {color:#a9bfd6;font-size:16px;max-width:660px;margin:0;line-height:1.6;}
.pill {display:inline-block;background:#143d45;color:#69e9d7;border:1px solid #2e777a;border-radius:30px;padding:7px 12px;font-size:12px;font-weight:700;margin-bottom:18px;}
.stat {background:#111f31;border:1px solid #263c54;border-radius:16px;padding:19px 22px;min-height:112px;}
.stat-label {color:#94a9c1;font-size:12px;text-transform:uppercase;letter-spacing:.11em;font-weight:700;}
.stat-value {font-family:'Space Grotesk',sans-serif;color:#f2f7ff;font-size:29px;font-weight:700;margin-top:9px;}
.section-title {font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#f2f7ff;margin:23px 0 8px;}
.section-sub {color:#93abc4;font-size:14px;margin-bottom:16px;}
[data-testid="stForm"] {background:#111e30;border:1px solid #2a4059;border-radius:18px;padding:22px;}
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] {background-color:#0a1728!important;color:#e9f3ff!important;}
label, [data-testid="stWidgetLabel"] {color:#c4d4e8!important;}
.stButton>button, [data-testid="stFormSubmitButton"] button {background:linear-gradient(100deg,#35d3c0,#37a8f5)!important;color:#041322!important;border:0!important;border-radius:11px!important;font-weight:800!important;min-height:47px;width:100%;}
.result {background:#111e30;border:1px solid #2a4059;border-radius:18px;padding:28px;min-height:360px;}
.result-big {font-family:'Space Grotesk',sans-serif;font-size:47px;font-weight:700;margin:12px 0;color:#f4f8ff;}
.muted {color:#9cb2c9;font-size:14px;line-height:1.6;}
.risk-track {height:13px;border-radius:50px;background:#253b50;overflow:hidden;margin:18px 0 9px;}
.risk-fill {height:100%;border-radius:50px;}
.status {display:inline-block;border-radius:30px;padding:8px 13px;font-size:13px;font-weight:800;}
.status-ok {color:#72e5bb;background:#153b37;border:1px solid #266c59;}
.status-bad {color:#ffad91;background:#4a292d;border:1px solid #8b484c;}
.foot {font-size:12px;color:#718aa4;margin-top:25px;border-top:1px solid #263c54;padding-top:17px;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def charger_modele():
    path = Path(__file__).parent / "modele_maintenance.joblib"
    return joblib.load(path)

st.markdown('''<div class="hero"><span class="pill">● MACHINE LEARNING · RANDOM FOREST</span><div class="eyebrow">INDUSTRIAL INTELLIGENCE / 01</div><h1>Machine<span style="color:#48dbc9">Pulse</span><br>Maintenance prédictive.</h1><p>Anticipez les risques de défaillance de vos équipements à partir de leurs paramètres de fonctionnement. Une analyse simple, instantanée et explicable.</p></div>''', unsafe_allow_html=True)

for col, label, value in zip(st.columns(3), ["MODÈLE", "DONNÉES D'ENTRAÎNEMENT", "F1-SCORE SUR TEST"], ["Random Forest", "10 000 observations", "0,63"]):
    with col:
        st.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Centre de diagnostic</div><div class="section-sub">Renseignez les mesures de votre équipement, puis lancez l’analyse.</div>', unsafe_allow_html=True)
left, right = st.columns([1.15, 1], gap="large")
with left:
    with st.form("diagnostic"):
        st.markdown("#### Paramètres de fonctionnement")
        type_machine = st.selectbox("Type de produit", ["L", "M", "H"], help="Catégorie du produit : L, M ou H")
        a, b = st.columns(2)
        with a:
            air = st.number_input("Température de l'air (K)", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
            vitesse = st.number_input("Vitesse de rotation (rpm)", min_value=0, max_value=10000, value=1500, step=10)
            usure = st.number_input("Usure de l'outil (min)", min_value=0, max_value=1000, value=100, step=1)
        with b:
            process = st.number_input("Température du processus (K)", min_value=250.0, max_value=400.0, value=310.0, step=0.1)
            couple = st.number_input("Couple mécanique (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=0.5)
        submitted = st.form_submit_button("⚡ Analyser la machine", use_container_width=True)

with right:
    if submitted:
        try:
            modele = charger_modele()
            observation = pd.DataFrame({"Type":[type_machine],"Air temperature":[air],"Process temperature":[process],"Rotational speed":[vitesse],"Torque":[couple],"Tool wear":[usure]})
            pred = int(modele.predict(observation)[0])
            prob = float(modele.predict_proba(observation)[0][1])
            pct = prob * 100
            color = "#f18c78" if pred else "#46d8ae"
            status = "⚠ Risque de panne détecté" if pred else "✓ Aucune panne prédite"
            status_class = "status-bad" if pred else "status-ok"
            st.markdown(f'''<div class="result"><div class="eyebrow">RÉSULTAT DE L'ANALYSE</div><span class="status {status_class}">{status}</span><div class="result-big">{pct:.1f}<span style="font-size:24px;color:#9fb4c9"> %</span></div><div class="muted">Probabilité estimée de panne</div><div class="risk-track"><div class="risk-fill" style="width:{pct:.2f}%;background:{color}"></div></div><div class="muted">Seuil de classification du modèle : 50 %</div><p class="muted" style="margin-top:25px">{'Une vérification technique peut être envisagée.' if pred else 'Le modèle ne signale pas de panne pour ces paramètres.'} Cette prédiction est indicative et ne constitue pas un diagnostic de sécurité.</p></div>''', unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Fichier modele_maintenance.joblib introuvable. Place-le dans le même dossier que app.py.")
        except Exception as e:
            st.error(f"Impossible d'effectuer la prédiction : {e}")
    else:
        st.markdown('''<div class="result"><div class="eyebrow">TABLEAU DE RÉSULTAT</div><div style="font-size:52px;margin:30px 0 16px">◎</div><h3 style="color:#f0f6ff">En attente d'analyse</h3><p class="muted">Complétez les paramètres à gauche et cliquez sur « Analyser la machine » pour afficher la probabilité de panne et le diagnostic.</p><div class="risk-track"><div class="risk-fill" style="width:0%"></div></div></div>''', unsafe_allow_html=True)

st.markdown('<div class="foot">MachinePulse · Démonstrateur pédagogique · Données synthétiques AI4I 2020 · Les prédictions ne remplacent pas une expertise de maintenance industrielle.</div>', unsafe_allow_html=True)
