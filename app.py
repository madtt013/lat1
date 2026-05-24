import streamlit as st
import pandas as pd
import plotly.express as px

from forecast import run_forecast

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Inventory Forecast",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("Inventory Forecast Dashboard")

st.write(
    "Cluster-Based Forecasting menggunakan "
    "Prophet, Exponential Smoothing, "
    "dan Moving Average"
)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload File Excel",
    type=['xlsx']
)

# ==========================================
# RUN FORECAST
# ==========================================

if uploaded_file:

    with st.spinner("Processing Forecast..."):

        result = run_forecast(uploaded_file)

    st.success("Forecast selesai!")

    # ======================================
    # TAMPILKAN DATA
    # ======================================

    st.subheader("Hasil Forecast")

    st.dataframe(result)

    # ======================================
    # FILTER PRODUK
    # ======================================

    produk_list = result['id_produk'].unique()

    selected_produk = st.selectbox(
        "Pilih Produk",
        produk_list
    )

    filtered = result[
        result['id_produk'] == selected_produk
    ]

    # ======================================
    # INFO PRODUK
    # ======================================

    kategori = filtered['kategori'].iloc[0]

    st.metric(
        "Kategori Cluster",
        kategori
    )

    # ======================================
    # VISUALISASI
    # ======================================

    st.subheader("Grafik Forecast")

fig = px.line(
    filtered,
    x='tanggal',
    y='forecast',
    title=f'Forecast Produk {selected_produk}'
)

st.plotly_chart(fig, use_container_width=True)
