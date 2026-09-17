import os
import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="Dashboard Iklim Pesisir Pulau Sumatera",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "hasil_prediksi"
RESULT_DIR.mkdir(exist_ok=True)

STATIONS = {
    "Stasiun Minangkabau": "minang kabau data FIX.xlsx",
    "Stasiun Pesawaran": "pesawaran data FIX.xlsx",
    "Stasiun Maritim Panjang": "Maritim panjang data FIX.xlsx",
}

TARGETS = ["TN", "TX", "TAVG", "RH_AVG", "RR", "SS", "FF_X", "FF_AVG"]
WINDOW = 6
FORECAST_MONTHS = 360
HIST_START = pd.Timestamp("1985-01-01")
HIST_END = pd.Timestamp("2025-12-31")
FORECAST_START = pd.Timestamp("2026-01-01")

PARAMETER_INFO = {
    "TN": ("Temperatur Minimum", "°C"),
    "TX": ("Temperatur Maksimum", "°C"),
    "TAVG": ("Temperatur Rata-rata", "°C"),
    "RH_AVG": ("Kelembapan Relatif Rata-rata", "%"),
    "RR": ("Curah Hujan", "mm"),
    "SS": ("Lama Penyinaran Matahari", "jam"),
    "FF_X": ("Kecepatan Angin Maksimum", "m/s"),
    "FF_AVG": ("Kecepatan Angin Rata-rata", "m/s"),
}

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    .main-title {font-size: 30px; font-weight: 700; margin-bottom: 0;}
    .sub-title {font-size: 17px; color: #555; margin-top: 2px;}
    .small-note {font-size: 13px; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PEMBACAAN EXCEL FLEXIBLE
# ============================================================
def normalize_col_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Z0-9_]+", "", s)
    return s


def find_header_row(raw):
    expected = set(TARGETS + ["YEAR", "DOY", "DATE", "TANGGAL"])
    best_row = None
    best_score = -1
    for i in range(min(len(raw), 80)):
        vals = {normalize_col_name(v) for v in raw.iloc[i].tolist() if not pd.isna(v)}
        score = len(vals.intersection(expected))
        if score > best_score:
            best_score = score
            best_row = i
    if best_score < 4:
        raise ValueError(
            "Header Excel tidak ditemukan. Pastikan menggunakan file dataset FIX."
        )
    return best_row


def canonicalize_columns(df):
    aliases = {
        "TAHUN": "YEAR",
        "THN": "YEAR",
        "HARI_KE": "DOY",
        "HARIKE": "DOY",
        "DAY_OF_YEAR": "DOY",
        "DATE": "DATE",
        "TANGGAL": "DATE",
        "TN": "TN",
        "TX": "TX",
        "TAVG": "TAVG",
        "RH_AVG": "RH_AVG",
        "RHAVG": "RH_AVG",
        "RR": "RR",
        "SS": "SS",
        "FF_X": "FF_X",
        "FFX": "FF_X",
        "FF_AVG": "FF_AVG",
        "FFAVG": "FF_AVG",
    }
    renamed = {}
    for c in df.columns:
        n = normalize_col_name(c)
        renamed[c] = aliases.get(n, n)
    df = df.rename(columns=renamed)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def read_excel_flex(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    raw = pd.read_excel(path, header=None, engine="openpyxl")
    if raw.empty:
        raise ValueError("File Excel kosong.")

    header_row = find_header_row(raw)
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = [normalize_col_name(x) for x in raw.iloc[header_row].tolist()]
    df = canonicalize_columns(df)
    df = df.dropna(how="all")

    # Konversi kolom numerik yang dibutuhkan
    for col in [c for c in TARGETS + ["YEAR", "DOY"] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bentuk DATE dari YEAR + DOY jika belum tersedia
    if "DATE" in df.columns:
        parsed = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=True)
    elif "YEAR" in df.columns and "DOY" in df.columns:
        parsed = (
            pd.to_datetime(df["YEAR"].astype("Int64").astype(str) + "-01-01", errors="coerce")
            + pd.to_timedelta(df["DOY"] - 1, unit="D")
        )
    else:
        missing = [c for c in ["YEAR", "DOY"] if c not in df.columns]
        raise ValueError("Kolom tanggal tidak lengkap. Tidak ditemukan: " + ", ".join(missing))

    df["DATE"] = parsed

    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise ValueError(
            "Kolom parameter dataset tidak lengkap: " + ", ".join(missing)
        )

    keep = ["DATE"] + TARGETS
    df = df[keep].copy()
    df = df.dropna(subset=["DATE"])
    df = df[(df["DATE"] >= HIST_START) & (df["DATE"] <= HIST_END)]
    df = df.sort_values("DATE").drop_duplicates("DATE")
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_station_data(filename):
    return read_excel_flex(str(BASE_DIR / filename))


# ============================================================
# PREPROCESSING
# ============================================================
def monthly_aggregate(df):
    x = df.set_index("DATE")[TARGETS]
    monthly = pd.DataFrame(index=x.resample("MS").asfreq().index)
    monthly["RR"] = x["RR"].resample("MS").sum(min_count=1)
    for col in TARGETS:
        if col != "RR":
            monthly[col] = x[col].resample("MS").mean()
    return monthly.reset_index()


def make_supervised(monthly, window=WINDOW):
    df = monthly.copy().sort_values("DATE").reset_index(drop=True)

    for lag in range(1, window + 1):
        for col in TARGETS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    month_num = df["DATE"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    feature_cols = [
        f"{col}_lag{lag}"
        for lag in range(1, window + 1)
        for col in TARGETS
    ] + ["month_sin", "month_cos"]

    data = df.dropna(subset=feature_cols + TARGETS).reset_index(drop=True)
    return data, feature_cols


# ============================================================
# MODEL CEPAT
# ============================================================
@st.cache_resource(show_spinner=False)
def fit_station_model(station_name, start_date, end_date):
    """Satu Random Forest multi-output untuk 8 parameter.
    Ini jauh lebih cepat daripada melatih 8 Random Forest terpisah.
    GridSearch tetap digunakan, tetapi dibuat ringan.
    """
    daily = load_station_data(STATIONS[station_name])
    selected = daily[
        (daily["DATE"] >= pd.Timestamp(start_date))
        & (daily["DATE"] <= pd.Timestamp(end_date))
    ].copy()

    monthly = monthly_aggregate(selected)
    supervised, feature_cols = make_supervised(monthly, WINDOW)

    if len(supervised) < 80:
        raise ValueError(
            f"Data bulanan setelah preprocessing hanya {len(supervised)} baris."
        )

    split = int(len(supervised) * 0.80)
    train_df = supervised.iloc[:split].copy()
    test_df = supervised.iloc[split:].copy()

    X_train = train_df[feature_cols]
    Y_train = train_df[TARGETS]
    X_test = test_df[feature_cols]
    Y_test = test_df[TARGETS]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    Y_train_s = scaler_y.fit_transform(Y_train)

    # GridSearch ringan: hanya 2 kombinasi, 3-fold time-series CV.
    base = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [80, 120],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }
    n_splits = 3 if len(X_train_s) >= 120 else 2
    cv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train_s, Y_train_s)
    model = grid.best_estimator_

    # Evaluasi test set
    pred_test_s = model.predict(scaler_x.transform(X_test))
    pred_test = scaler_y.inverse_transform(pred_test_s)
    actual_test = Y_test.to_numpy(dtype=float)

    metrics = []
    for i, target in enumerate(TARGETS):
        actual = actual_test[:, i]
        pred = pred_test[:, i]
        metrics.append(
            {
                "Parameter": target,
                "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                "MAE": float(mean_absolute_error(actual, pred)),
                "R²": float(r2_score(actual, pred)) if len(np.unique(actual)) > 1 else np.nan,
            }
        )

    # Forecast 2026-2055 secara rekursif
    history = monthly.copy().sort_values("DATE").reset_index(drop=True)
    future_rows = []

    progress = st.progress(0, text="Membangun prediksi 2026–2055...")
    for step in range(FORECAST_MONTHS):
        future_date = FORECAST_START + pd.DateOffset(months=step)
        row_features = {}

        for lag in range(1, WINDOW + 1):
            source = history.iloc[-lag]
            for col in TARGETS:
                row_features[f"{col}_lag{lag}"] = float(source[col])

        month_num = future_date.month
        row_features["month_sin"] = np.sin(2 * np.pi * month_num / 12)
        row_features["month_cos"] = np.cos(2 * np.pi * month_num / 12)

        X_future = pd.DataFrame([row_features])[feature_cols]
        pred_s = model.predict(scaler_x.transform(X_future))
        pred = scaler_y.inverse_transform(pred_s)[0]

        predicted = {TARGETS[i]: float(pred[i]) for i in range(len(TARGETS))}
        predicted["DATE"] = future_date
        future_rows.append(predicted)
        history = pd.concat([history, pd.DataFrame([predicted])], ignore_index=True)

        if step % 12 == 0 or step == FORECAST_MONTHS - 1:
            progress.progress(
                (step + 1) / FORECAST_MONTHS,
                text=f"Membangun prediksi: {step + 1}/{FORECAST_MONTHS} bulan",
            )
    progress.empty()

    forecast = pd.DataFrame(future_rows)[["DATE"] + TARGETS]

    # Simpan hasil di runtime untuk download/backup.
    key = hashlib.md5(
        f"{station_name}|{start_date}|{end_date}|{WINDOW}|{FORECAST_MONTHS}".encode()
    ).hexdigest()[:12]
    forecast_path = RESULT_DIR / f"forecast_{key}.csv"
    forecast.to_csv(forecast_path, index=False)

    return {
        "monthly": monthly,
        "train": train_df,
        "test": test_df,
        "metrics": pd.DataFrame(metrics),
        "forecast": forecast,
        "best_params": grid.best_params_,
        "cache_name": forecast_path.name,
        "n_daily": len(daily),
    }


# ============================================================
# TAMPILAN
# ============================================================
def show_header():
    st.markdown(
        '<div class="main-title">MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM WILAYAH PESISIR PANTAI PULAU SUMATERA</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Analisis Temporal Jangka Panjang Berbasis Random Forest - 1985 - 2025</div>',
        unsafe_allow_html=True,
    )
    st.divider()


def annual_forecast_table(forecast):
    x = forecast.copy()
    x["Tahun"] = x["DATE"].dt.year
    rows = []
    for year, g in x.groupby("Tahun"):
        row = {"Tahun": int(year)}
        for col in TARGETS:
            row[col] = g[col].sum() if col == "RR" else g[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def show_dashboard(station_name, start_date, end_date):
    show_header()

    try:
        with st.spinner(f"Memuat dan melatih model {station_name}... (pertama kali saja)"):
            result = fit_station_model(station_name, start_date, end_date)
    except Exception as e:
        st.error(f"Gagal membaca atau memproses {station_name}.")
        st.exception(e)
        return

    st.success(f"{station_name} siap. Jika stasiun ini dipilih lagi, hasil model akan menggunakan cache.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Harian", f"{result['n_daily']:,}")
    c2.metric("Data Bulanan", f"{len(result['monthly']):,}")
    c3.metric("Data Training", f"{len(result['train']):,}")
    c4.metric("Forecast", "2026–2055")

    st.subheader(f"Visualisasi Historis — {station_name}")
    parameter = st.selectbox(
        "Pilih parameter",
        TARGETS,
        format_func=lambda x: f"{x} — {PARAMETER_INFO[x][0]}",
        key=f"parameter_{station_name}",
    )
    chart_df = result["monthly"].set_index("DATE")[[parameter]].rename(
        columns={parameter: f"{parameter} ({PARAMETER_INFO[parameter][1]})"}
    )
    st.line_chart(chart_df)

    st.subheader("Hasil Prediksi Tahunan 2026–2055")
    annual = annual_forecast_table(result["forecast"])
    shown = annual.copy()
    for col in TARGETS:
        shown[col] = shown[col].round(2)
    st.dataframe(shown, use_container_width=True, hide_index=True)

    csv = annual.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Prediksi CSV",
        data=csv,
        file_name=f"prediksi_{station_name.lower().replace(' ', '_')}_2026_2055.csv",
        mime="text/csv",
        key=f"download_{station_name}",
    )

    st.caption(f"Parameter terbaik GridSearchCV: {result['best_params']}")
    st.caption(f"File hasil runtime: {result['cache_name']}")


def show_validation(station_name, start_date, end_date):
    show_header()
    st.subheader("Validasi & Evaluasi Model")
    try:
        with st.spinner("Memuat hasil evaluasi..."):
            result = fit_station_model(station_name, start_date, end_date)
    except Exception as e:
        st.error("Evaluasi belum dapat ditampilkan.")
        st.exception(e)
        return

    shown = result["metrics"].copy()
    for c in ["RMSE", "MAE", "R²"]:
        shown[c] = shown[c].round(4)
    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.info(
        "Pembagian data menggunakan 80% data training dan 20% data pengujian secara kronologis. "
        "RMSE dan MAE menunjukkan besarnya kesalahan prediksi, sedangkan R² menunjukkan proporsi variasi data uji yang dijelaskan model."
    )


def show_profile():
    show_header()
    st.subheader("Profil Peneliti")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Nama:** Huriyatul Firdausi  
            **NIM:** 06111382328074  
            **Program Studi:** Pendidikan Fisika  
            **Fakultas:** Keguruan dan Ilmu Pendidikan  
            **Universitas:** Universitas Sriwijaya  
            **Tahun:** 2026
            """
        )
    with col2:
        st.markdown(
            """
            **Dosen Pembimbing:** Dr. Melly Ariska, S.Pd., M.Sc.  
            **Metode:** Random Forest Regressor  
            **Periode historis:** 1985–2025  
            **Periode prediksi:** 2026–2055  
            **Window:** 6 bulan  
            **Evaluasi:** RMSE, MAE, R²
            """
        )

    st.divider()
    st.subheader("Parameter Dataset")
    rows = [
        {"Kode": code, "Parameter": PARAMETER_INFO[code][0], "Satuan": PARAMETER_INFO[code][1]}
        for code in TARGETS
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info("Dataset penelitian menggunakan parameter meteorologi TN, TX, TAVG, RH_AVG, RR, SS, FF_X, dan FF_AVG.")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih halaman",
    ["Dashboard", "Validasi & Evaluasi", "Profil Peneliti"],
)

st.sidebar.divider()
st.sidebar.subheader("Konfigurasi Data")
station = st.sidebar.selectbox("Pilih stasiun", list(STATIONS.keys()))

start_date = st.sidebar.date_input(
    "Mulai",
    value=HIST_START.date(),
    min_value=HIST_START.date(),
    max_value=HIST_END.date(),
)
end_date = st.sidebar.date_input(
    "Selesai",
    value=HIST_END.date(),
    min_value=HIST_START.date(),
    max_value=HIST_END.date(),
)

if start_date > end_date:
    st.sidebar.error("Tanggal mulai harus lebih kecil atau sama dengan tanggal selesai.")
    st.stop()

st.sidebar.caption("Periode historis: 1985–2025")
st.sidebar.caption("Forecast: 2026–2055")
st.sidebar.caption("Model: Random Forest Regressor")
st.sidebar.caption("Training: Multi-output + GridSearchCV ringan")

if menu == "Dashboard":
    show_dashboard(station, start_date, end_date)
elif menu == "Validasi & Evaluasi":
    show_validation(station, start_date, end_date)
else:
    show_profile()
