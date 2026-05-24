import pandas as pd
import numpy as np

from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =====================================================
# MOVING AVERAGE FUNCTION
# =====================================================

def moving_average_forecast(data, window=7, steps=30):

    avg = data['y'].rolling(window=window).mean().iloc[-1]

    future_dates = pd.date_range(
        start=data['ds'].max(),
        periods=steps + 1,
        freq='D'
    )[1:]

    forecast = pd.DataFrame({
        'tanggal': future_dates,
        'forecast': [avg] * steps
    })

    return forecast


# =====================================================
# MAIN FORECAST FUNCTION
# =====================================================

def run_forecast(file):

    # =========================================
    # LOAD DATA
    # =========================================

    df = pd.read_excel(file)

    df.columns = [
        'id',
        'id_relasi',
        'id_produk',
        'tgl_input',
        'keluar'
    ]

    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    df['keluar'] = pd.to_numeric(df['keluar'])

    df = df.dropna()

    # =========================================
    # FEATURE ENGINEERING
    # =========================================

    cluster_df = df.groupby('id_produk').agg({
        'keluar': ['sum', 'mean', 'std', 'count']
    }).reset_index()

    cluster_df.columns = [
        'id_produk',
        'total_keluar',
        'rata_keluar',
        'std_keluar',
        'frekuensi'
    ]

    cluster_df['std_keluar'] = cluster_df[
        'std_keluar'
    ].fillna(0)

    # =========================================
    # NORMALISASI
    # =========================================

    features = [
        'total_keluar',
        'rata_keluar',
        'std_keluar',
        'frekuensi'
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        cluster_df[features]
    )

    # =========================================
    # KMEANS
    # =========================================

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_df['cluster'] = kmeans.fit_predict(
        X_scaled
    )

    # =========================================
    # LABEL CLUSTER
    # =========================================

    cluster_summary = cluster_df.groupby(
        'cluster'
    )['total_keluar'].mean()

    cluster_sorted = cluster_summary.sort_values(
        ascending=False
    )

    cluster_A = cluster_sorted.index[0]
    cluster_B = cluster_sorted.index[1]
    cluster_C = cluster_sorted.index[2]

    def label_cluster(x):

        if x == cluster_A:
            return 'A'

        elif x == cluster_B:
            return 'B'

        else:
            return 'C'

    cluster_df['kategori'] = cluster_df[
        'cluster'
    ].apply(label_cluster)

    # =========================================
    # MERGE KE DATA UTAMA
    # =========================================

    df = df.merge(
        cluster_df[['id_produk', 'kategori']],
        on='id_produk',
        how='left'
    )

    # =========================================
    # FORECASTING
    # =========================================

    all_forecast = []

    produk_list = df['id_produk'].unique()

    for produk in produk_list:

        try:

            produk_df = df[
                df['id_produk'] == produk
            ]

            kategori = produk_df[
                'kategori'
            ].iloc[0]

            ts = produk_df.groupby(
                'tgl_input'
            )['keluar'].sum().reset_index()

            ts.columns = ['ds', 'y']

            # minimal data
            if len(ts) < 15:
                continue

            # =================================
            # CLUSTER A -> PROPHET
            # =================================

            if kategori == 'A':

                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )

                model.fit(ts)

                future = model.make_future_dataframe(
                    periods=30
                )

                forecast = model.predict(future)

                hasil = forecast[
                    ['ds', 'yhat']
                ].tail(30)

                hasil.columns = [
                    'tanggal',
                    'forecast'
                ]

            # =================================
            # CLUSTER B -> EXP SMOOTHING
            # =================================

            elif kategori == 'B':

                model = ExponentialSmoothing(
                    ts['y'],
                    trend='add',
                    seasonal=None
                ).fit()

                pred = model.forecast(30)

                future_dates = pd.date_range(
                    start=ts['ds'].max(),
                    periods=31,
                    freq='D'
                )[1:]

                hasil = pd.DataFrame({
                    'tanggal': future_dates,
                    'forecast': pred.values
                })

            # =================================
            # CLUSTER C -> MOVING AVERAGE
            # =================================

            else:

                hasil = moving_average_forecast(ts)

            hasil['id_produk'] = produk
            hasil['kategori'] = kategori

            all_forecast.append(hasil)

        except:
            continue

    # =========================================
    # FINAL RESULT
    # =========================================

    final_forecast = pd.concat(all_forecast)

    return final_forecast
