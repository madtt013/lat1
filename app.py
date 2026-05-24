import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# =========================================
# CONFIG STREAMLIT
# =========================================
st.set_page_config(
    page_title="Forecasting Barang Keluar SARIMA",
    layout="wide"
)

st.title("Forecasting Barang Keluar Menggunakan SARIMA")

st.write("""
Aplikasi ini digunakan untuk melakukan forecasting data barang keluar
menggunakan metode SARIMA.
""")

# =========================================
# UPLOAD FILE
# =========================================
uploaded_file = st.file_uploader(
    "Upload File Excel",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:

    # =========================================
    # LOAD DATA
    # =========================================
    df = pd.read_excel(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(df.head())

    # =========================================
    # PREPROCESSING
    # =========================================
    st.subheader("Preprocessing Data")

    # Mengubah tanggal
    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    # Menghapus data kosong
    df = df.dropna(subset=['keluar'])

    # Mengubah keluar menjadi numerik
    df['keluar'] = pd.to_numeric(df['keluar'])

    # Agregasi per bulan
    df_bulanan = df.groupby(
        pd.Grouper(key='tgl_input', freq='M')
    )['keluar'].sum().reset_index()

    st.write("Data Bulanan")
    st.dataframe(df_bulanan)

    # =========================================
    # SET INDEX
    # =========================================
    df_bulanan.set_index('tgl_input', inplace=True)

    # =========================================
    # VISUALISASI
    # =========================================
    st.subheader("Grafik Data Bulanan")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df_bulanan.index, df_bulanan['keluar'])
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Barang Keluar")
    ax.set_title("Data Barang Keluar Bulanan")

    st.pyplot(fig)

    # =========================================
    # TRAIN TEST SPLIT
    # =========================================
    train_size = int(len(df_bulanan) * 0.8)

    train = df_bulanan.iloc[:train_size]
    test = df_bulanan.iloc[train_size:]

    # =========================================
    # PARAMETER SARIMA
    # =========================================
    st.subheader("Parameter SARIMA")

    p = st.number_input("p", 0, 5, 1)
    d = st.number_input("d", 0, 2, 1)
    q = st.number_input("q", 0, 5, 1)

    P = st.number_input("P", 0, 5, 1)
    D = st.number_input("D", 0, 2, 1)
    Q = st.number_input("Q", 0, 5, 1)

    seasonal_period = st.number_input(
        "Seasonal Period",
        1,
        24,
        12
    )

    # =========================================
    # TRAIN MODEL
    # =========================================
    if st.button("Jalankan Forecasting"):

        st.subheader("Training Model SARIMA")

        model = SARIMAX(
            train['keluar'],
            order=(p,d,q),
            seasonal_order=(P,D,Q,seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit(disp=False)

        st.success("Model berhasil dilatih!")

        # =========================================
        # PREDIKSI TEST
        # =========================================
        pred = model_fit.predict(
            start=test.index[0],
            end=test.index[-1]
        )

        # =========================================
        # EVALUASI
        # =========================================
        mae = mean_absolute_error(test['keluar'], pred)
        rmse = np.sqrt(mean_squared_error(test['keluar'], pred))

        st.subheader("Evaluasi Model")

        st.write(f"MAE : {mae:.2f}")
        st.write(f"RMSE : {rmse:.2f}")

        # =========================================
        # GRAFIK HASIL
        # =========================================
        st.subheader("Hasil Prediksi")

        fig2, ax2 = plt.subplots(figsize=(12,5))

        ax2.plot(train.index, train['keluar'], label='Train')
        ax2.plot(test.index, test['keluar'], label='Actual')
        ax2.plot(pred.index, pred, label='Prediction')

        ax2.set_title("SARIMA Forecasting")
        ax2.legend()

        st.pyplot(fig2)

        # =========================================
        # FORECAST KE DEPAN
        # =========================================
        st.subheader("Forecast Masa Depan")

        n_forecast = st.number_input(
            "Jumlah Bulan Forecast",
            1,
            24,
            6
        )

        future_forecast = model_fit.forecast(steps=n_forecast)

        future_dates = pd.date_range(
            start=df_bulanan.index[-1] + pd.offsets.MonthEnd(),
            periods=n_forecast,
            freq='M'
        )

        forecast_df = pd.DataFrame({
            'Tanggal': future_dates,
            'Forecast': future_forecast
        })

        st.dataframe(forecast_df)

        # =========================================
        # GRAFIK FORECAST
        # =========================================
        fig3, ax3 = plt.subplots(figsize=(12,5))

        ax3.plot(
            df_bulanan.index,
            df_bulanan['keluar'],
            label='Data Asli'
        )

        ax3.plot(
            future_dates,
            future_forecast,
            label='Forecast'
        )

        ax3.set_title("Forecast Masa Depan")
        ax3.legend()

        st.pyplot(fig3)

        # =========================================
        # DOWNLOAD HASIL
        # =========================================
        csv = forecast_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Hasil Forecast",
            data=csv,
            file_name='hasil_forecast.csv',
            mime='text/csv'
        )
