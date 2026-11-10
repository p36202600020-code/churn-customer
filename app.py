"""Dashboard prediksi churn industri gas alam (jalankan: streamlit run app.py)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Gas Alam | Customer Churn", page_icon="", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_hybrid.pkl"
META_PATH = ROOT / "model_metadata.json"
DATA_PATH = ROOT / "dataset.xlsx"


@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


@st.cache_data(show_spinner=False)
def load_customers():
    raw = pd.read_excel(DATA_PATH, sheet_name="Master Customer Data", engine="openpyxl")
    d = raw.copy()
    d.columns = d.columns.astype(str).str.strip()
    d["Cust. ID"] = d["Cust. ID"].astype(str).str.strip()
    d["segment"] = d["Industry Class Type"]
    for c in ["tenure_month", "avg_usage_m3_6m", "bill_avg_6m", "bill_shock_pct",
              "late_payment_6m", "complaint_3m", "resolution_days_avg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["segment", "usage_trend", "has_competitor", "contract_status", "City", "State/Prov"]:
        d[c] = d[c].map(lambda v: v.strip() if isinstance(v, str) else np.nan)
        d[c] = d[c].replace("", np.nan)
    # Engineered features MUST use the same deterministic formula as the Colab notebook.
    tenure_safe = d["tenure_month"].clip(lower=1)
    d["complaint_per_year"] = d["complaint_3m"] * 12 / tenure_safe
    d["late_payment_per_year"] = d["late_payment_6m"] * 12 / tenure_safe
    d["bill_shock_x_declining"] = d["bill_shock_pct"] * d["usage_trend"].eq("Decreasing").astype(float)
    d = d.drop_duplicates(subset=["Cust. ID"], keep="first").reset_index(drop=True)
    return d


def classify_churn(probability: float, decision_threshold: float) -> tuple[str, str, str]:
    """Prediksi Yes/No dan kategori risiko memakai threshold model yang sama.

    Threshold ditentukan saat evaluasi model di notebook, lalu disimpan di
    model_metadata.json; Streamlit hanya membaca dan menerapkannya.
    """
    if probability >= decision_threshold:
        return "Yes", "HIGH RISK", "🔴"
    return "No", "LOW RISK", "🟢"


def format_churn_probability(probability: float) -> str:
    """Tampilkan skor kecil tanpa membulatkan nilai nonnol menjadi 0,0%."""
    percent = probability * 100
    if probability == 0.0:
        return "0%"  # Nol benar-benar berasal dari output numerik model.
    if percent < 0.0001:
        return "< 0,0001%"
    if percent < 1:
        return f"{percent:.4f}%".replace(".", ",")
    return f"{percent:.2f}%".replace(".", ",")


def get_churn_yes_probability(model, customer_frame: pd.DataFrame) -> tuple[float, int]:
    """Ambil kelas Churn=1 secara eksplisit; bukan mengasumsikan kolom ke-2."""
    classes = np.asarray(model.classes_)
    indices = np.flatnonzero(classes == 1)
    if len(indices) != 1:
        raise ValueError(f"Model tidak memiliki kelas Churn=1. Kelas ditemukan: {classes.tolist()}")
    yes_index = int(indices[0])
    result = np.asarray(model.predict_proba(customer_frame), dtype=float)
    if result.ndim != 2 or result.shape[0] != 1 or yes_index >= result.shape[1]:
        raise ValueError("Bentuk hasil predict_proba tidak sesuai.")
    probability = float(result[0, yes_index])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"Skor probabilitas tidak valid: {probability}")
    return probability, yes_index


def recommend_actions(risk_label: str) -> str:
    """Rekomendasi sederhana berdasarkan kategori risiko hasil prediksi."""
    if risk_label == "HIGH RISK":
        return "Rekomendasi: Service Recovery - Telepon dalam 24 jam"
    return "Aman"


@st.cache_resource(show_spinner=False)
def make_shap_explainer(_model, feature_names: tuple, _background, yes_index: int):
    """SHAP pada fitur numerik yang telah di-encode oleh pipeline terlatih."""
    def probability_positive(values):
        return _model.named_steps["classifier"].predict_proba(np.asarray(values))[:, yes_index]

    return shap.Explainer(
        probability_positive,
        shap.maskers.Independent(_background),
        algorithm="permutation",
    )


st.title("Dashboard Customer Churn — Industri Gas Alam")
st.caption("Prediksi berbasis model Machine Learning + penjelasan SHAP + rekomendasi retensi")

missing = [p.name for p in [MODEL_PATH, META_PATH, DATA_PATH] if not p.exists()]
if missing:
    st.error("Berkas belum tersedia: " + ", ".join(missing))
    st.info("Jalankan notebook Colab sampai tahap Save Model, lalu tempatkan berkas di folder yang sama dengan app.py.")
    st.stop()

try:
    model, metadata = load_model_and_metadata()
    df = load_customers()
except Exception as exc:
    st.error(f"Gagal memuat model/data: {exc}")
    st.stop()

features = metadata["feature_columns"]
numeric = metadata["numeric_features"]
missing_columns = sorted(set(features) - set(df.columns))
if missing_columns:
    st.error(f"Kolom input tidak ditemukan: {missing_columns}")
    st.stop()

st.info(
    f"Model terpilih: **{metadata['selected_model']}** | "
    f"AUC test: **{metadata['test_auc_selected_model']:.3f}** | "
    f"Accuracy test: **{metadata.get('test_accuracy_selected_model', float('nan')):.1%}** | "
    f"Training: **{metadata['trained_rows']:,} pelanggan**"
)
left, center, right = st.columns([1.05, 1.65, 1.35], gap="large")

with left:
    st.subheader("1 · Pilih Pelanggan")
    selected_id = st.selectbox("ID Pelanggan", options=df["Cust. ID"].tolist(), index=0)
    customer = df.loc[df["Cust. ID"] == selected_id].iloc[0]
    st.text(f"Kota: {customer.get('City', '-')}")
    st.text(f"Segment: {customer.get('segment', '-')}")
    st.text(f"Status kontrak: {customer.get('contract_status', '-')}")
    st.metric("Tenure", f"{customer['tenure_month']:.0f} bulan")
    st.metric("Keluhan 3 bulan", f"{customer['complaint_3m']:.0f}")
    st.metric("Perubahan tagihan", f"{customer['bill_shock_pct']:+.1f}%")

X_selected = pd.DataFrame([customer[features].to_dict()], columns=features)
try:
    prob, yes_index = get_churn_yes_probability(model, X_selected)
    decision_threshold = float(metadata["decision_threshold"])
    if not (np.isfinite(decision_threshold) and 0 < decision_threshold < 1):
        raise ValueError("decision_threshold harus berada di antara 0 dan 1.")
except Exception as exc:
    st.error(f"Gagal menghitung probabilitas atau membaca threshold model: {exc}")
    st.stop()
churn_prediction, risk_label, risk_icon = classify_churn(prob, decision_threshold)

with center:
    st.subheader("2 · Prediksi & Penjelasan")
    st.metric("Prediksi Churn (Yes/No)", churn_prediction)
    st.metric("Probabilitas Churn Yes (skor model)", format_churn_probability(prob))
    st.progress(max(0.0, min(prob, 1.0)))
    st.markdown(f"### {risk_icon} {risk_label}")
    if prob == 0.0:
        st.info("Model mengembalikan probabilitas tepat 0 pada presisi numerik yang tersedia. "
                "Tampilan ini bukan kesalahan pembulatan; tinjau fitur dan validitas model bila hasil terasa tidak wajar.")
    elif prob * 100 < 0.05:
        st.caption("Skor churn sangat kecil tetapi **bukan nol**. Angka utama ditampilkan dengan presisi tambahan agar tidak tampak sebagai 0%.")
    st.caption(
        f"Batas prediksi dan kategori risiko: {decision_threshold:.1%}.  \n"
        f"Skor ≥ {decision_threshold:.1%} = Churn Yes / HIGH RISK;  \n"
        f"skor < {decision_threshold:.1%} = Churn No / LOW RISK."
    )
    st.markdown("**SHAP Waterfall — Faktor yang berkontribusi terhadap prediksi**")
    if st.checkbox("Tampilkan SHAP Waterfall", value=True):
        with st.spinner("Menghitung kontribusi SHAP untuk pelanggan terpilih..."):
            preprocessor = model.named_steps["preprocess"]
            encoded_names = list(preprocessor.get_feature_names_out())
            background = df[features].sample(n=min(12, len(df)), random_state=42)
            background_encoded = pd.DataFrame(
                preprocessor.transform(background), columns=encoded_names
            )
            selected_encoded = pd.DataFrame(
                preprocessor.transform(X_selected), columns=encoded_names
            )
            explainer = make_shap_explainer(model, tuple(encoded_names), background_encoded, yes_index)
            explanation = explainer(selected_encoded, max_evals=2 * len(encoded_names) + 1)
            pretty_names = [
                name.replace("numeric__", "").replace("categorical__", "")
                    .replace("segment_", "Sektor: ").replace("usage_trend_", "Tren: ")
                    .replace("has_competitor_", "Kompetitor: ").replace("contract_status_", "Kontrak: ")
                    .replace("City_", "Kota: ").replace("State/Prov_", "Provinsi: ")
                for name in encoded_names
            ]
            display_values = selected_encoded.to_numpy().copy()
            for col in numeric:
                display_values[0, encoded_names.index("numeric__" + col)] = float(customer[col])
            explanation.feature_names = pretty_names
            explanation.data = display_values
            shap.plots.waterfall(explanation[0], max_display=12, show=False)
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)
            contributions = pd.Series(explanation.values[0], index=pretty_names).sort_values(ascending=False)
            important = contributions[contributions > 0].head(3)
            if not important.empty:
                st.markdown("**Fitur yang paling menaikkan skor churn:** " + ", ".join(important.index.tolist()))

with right:
    st.subheader("3 · Strategi Retensi")
    st.caption(f"Pelanggan: {selected_id}")
    recommendation = recommend_actions(risk_label)
    if risk_label == "HIGH RISK":
        st.warning(recommendation)
    else:
        st.success(recommendation)
