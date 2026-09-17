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
FORECAST_END = pd.Timestamp("2055-12-31")

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
    .card {
        padding: 18px; border-radius: 12px; border: 1px solid #ddd;
        background: #fafafa; margin-bottom: 12px;
    }
    .small-note {font-size: 13px; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FUNGSI BANTUAN DATA
# ============================================================
def normalize_col_name(x):
    """Normalisasi nama kolom agar tahan terhadap variasi spasi/tanda baca."""
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Z0-9_]+", "", s)
    return s


def find_header_row(raw, expected=TARGETS + ["YEAR", "DOY"]):
    """Cari baris header sebenarnya. File FIX mempunyai beberapa baris kosong/judul."""
    expected_set = {normalize_col_name(c) for c in expected}
    best_row = None
    best_score = -1

    max_rows = min(len(raw), 80)
    for i in range(max_rows):
        vals = {normalize_col_name(v) for v in raw.iloc[i].tolist() if not pd.isna(v)}
        score = len(vals.intersection(expected_set))
        if score > best_score:
            best_score = score
            best_row = i

    if best_score < 5:
        raise ValueError(
            "Baris header dataset tidak dapat ditemukan. "
            "Pastikan file Excel merupakan dataset FIX yang benar."
        )
    return best_row


def read_excel_flex(path):
    """Membaca Excel FIX tanpa mengasumsikan header berada di baris pertama."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    raw = pd.read_excel(path, header=None, engine="openpyxl")
    if raw.empty:
        raise ValueError("File Excel terbaca kosong.")

    header_row = find_header_row(raw)
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = [normalize_col_name(x) for x in raw.iloc[header_row].tolist()]

    # Buang kolom kosong dan baris kosong
    df = df.loc[:, [c != "" for c in df.columns]]
    df = df.dropna(how="all")

    # Jika ada nama kolom duplikat, pertahankan kemunculan pertama
    df = df.loc[:, ~df.columns.duplicated()]

    # Konversi angka
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["YEAR", "DOY"] + TARGETS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Kolom dataset tidak lengkap. Kolom yang belum ditemukan: "
            + ", ".join(missing)
        )

    df = df[required].copy()
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["DOY"] = pd.to_numeric(df["DOY"], errors="coerce")
    df = df.dropna(subset=["YEAR", "DOY"])
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)

    # Membentuk tanggal dari YEAR + DOY
    df["DATE"] = (
        pd.to_datetime(df["YEAR"].astype(str) + "-01-01", errors="coerce")
        + pd.to_timedelta(df["DOY"] - 1, unit="D")
    )
    df = df.dropna(subset=["DATE"])
    df = df[(df["DATE"] >= HIST_START) & (df["DATE"] <= HIST_END)]
    df = df.sort_values("DATE").drop_duplicates("DATE")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_station_data(filename):
    path = BASE_DIR / filename
    df = read_excel_flex(str(path))
    return df


def monthly_aggregate(df):
    """Agregasi harian menjadi bulanan. RR dijumlahkan, variabel lain dirata-ratakan."""
    x = df.set_index("DATE")[TARGETS].copy()
    monthly = pd.DataFrame(index=x.resample("MS").asfreq().index)

    monthly["RR"] = x["RR"].resample("MS").sum(min_count=1)
    for col in TARGETS:
        if col != "RR":
            monthly[col] = x[col].resample("MS").mean()

    return monthly.reset_index().rename(columns={"DATE": "DATE"})


# ============================================================
# FEATURE ENGINEERING
# ============================================================
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
# RANDOM FOREST
# ============================================================
def build_model(X_train, y_train):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    Xs = scaler_x.fit_transform(X_train)
    ys = scaler_y.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()

    base = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1],
    }

    n_splits = min(5, max(2, len(X_train) // 60))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid = GridSearchCV(
        base,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(Xs, ys)
    return grid.best_estimator_, scaler_x, scaler_y, grid.best_params_


@st.cache_data(show_spinner=False)
def train_station(station_name, start_date, end_date):
    """Latih model per stasiun dan simpan hasil forecast ke CSV cache."""
    filename = STATIONS[station_name]
    daily = load_station_data(filename)

    selected = daily[
        (daily["DATE"] >= pd.Timestamp(start_date))
        & (daily["DATE"] <= pd.Timestamp(end_date))
    ].copy()

    monthly = monthly_aggregate(selected)
    supervised, feature_cols = make_supervised(monthly, WINDOW)

    if len(supervised) < 80:
        raise ValueError(
            f"Data bulanan setelah preprocessing hanya {len(supervised)} baris. "
            "Data terlalu sedikit untuk pelatihan model."
        )

    # Split kronologis 80:20 agar data masa depan tidak masuk ke training.
    split = int(len(supervised) * 0.80)
    train_df = supervised.iloc[:split].copy()
    test_df = supervised.iloc[split:].copy()

    models = {}
    metrics = []

    for target in TARGETS:
        model, scaler_x, scaler_y, best_params = build_model(
            train_df[feature_cols], train_df[target]
        )
        models[target] = (model, scaler_x, scaler_y, best_params)

        X_test = scaler_x.transform(test_df[feature_cols])
        pred_scaled = model.predict(X_test).reshape(-1, 1)
        pred = scaler_y.inverse_transform(pred_scaled).ravel()
        actual = test_df[target].to_numpy(dtype=float)

        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae = float(mean_absolute_error(actual, pred))
        r2 = float(r2_score(actual, pred)) if len(np.unique(actual)) > 1 else np.nan
        metrics.append({
            "Parameter": target,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
        })

    metrics_df = pd.DataFrame(metrics)

    # Forecast 2026-2055 secara rekursif.
    history = monthly.copy().sort_values("DATE").reset_index(drop=True)
    future_rows = []

    for step in range(FORECAST_MONTHS):
        future_date = FORECAST_START + pd.DateOffset(months=step)
        month_num = future_date.month
        row_features = {}

        for lag in range(1, WINDOW + 1):
            source = history.iloc[-lag]
            for col in TARGETS:
                row_features[f"{col}_lag{lag}"] = float(source[col])

        row_features["month_sin"] = np.sin(2 * np.pi * month_num / 12)
        row_features["month_cos"] = np.cos(2 * np.pi * month_num / 12)
        X_future = pd.DataFrame([row_features])[feature_cols]

        predicted = {}
        for target in TARGETS:
            model, scaler_x, scaler_y, _ = models[target]
            Xs = scaler_x.transform(X_future)
            ps = model.predict(Xs).reshape(-1, 1)
            value = float(scaler_y.inverse_transform(ps)[0, 0])
            predicted[target] = value

        predicted["DATE"] = future_date
        future_rows.append(predicted)
        history = pd.concat([history, pd.DataFrame([predicted])], ignore_index=True)

    forecast = pd.DataFrame(future_rows)
    forecast = forecast[["DATE"] + TARGETS]

    # Simpan cache hasil agar tidak perlu melatih ulang ketika stasiun dipilih kembali.
    key = hashlib.md5(
        f"{station_name}|{start_date}|{end_date}|{WINDOW}|{FORECAST_MONTHS}".encode()
    ).hexdigest()[:12]
    forecast_path = RESULT_DIR / f"forecast_{key}.csv"
    forecast.to_csv(forecast_path, index=False)

    return monthly, train_df, test_df, metrics_df, forecast, forecast_path.name


# ============================================================
# FUNGSI TAMPILAN
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


def show_dashboard(station_name, start_date, end_date):
    show_header()

    try:
        with st.spinner(f"Memproses {station_name} ..."):
            monthly, train_df, test_df, metrics_df, forecast, cache_name = train_station(
                station_name, start_date, end_date
            )
    except Exception as e:
        st.error(f"Gagal membaca atau memproses {station_name}.")
        st.exception(e)
        st.info(
            "Pastikan file Excel FIX berada satu folder dengan app_dashboard_iklim_sumatera.py "
            "dan nama file sama persis dengan konfigurasi STATIONS."
        )
        return

    st.success(f"Data {station_name} berhasil diproses.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Harian", f"{len(load_station_data(STATIONS[station_name])):,}")
    c2.metric("Data Bulanan", f"{len(monthly):,}")
    c3.metric("Data Training", f"{len(train_df):,}")
    c4.metric("Forecast", "2026–2055")

    st.subheader(f"Visualisasi Historis — {station_name}")
    parameter = st.selectbox(
        "Pilih parameter",
        TARGETS,
        format_func=lambda x: f"{x} — {PARAMETER_INFO[x][0]}",
    )
    chart_df = monthly.set_index("DATE")[[parameter]].rename(
        columns={parameter: f"{parameter} ({PARAMETER_INFO[parameter][1]})"}
    )
    st.line_chart(chart_df)

    st.subheader("Hasil Prediksi Tahunan 2026–2055")
    annual = forecast.copy()
    annual["Tahun"] = annual["DATE"].dt.year
    annual["Bulan"] = annual["DATE"].dt.month

    # RR dijumlahkan tahunan; parameter lain dirata-ratakan tahunan.
    annual_rows = []
    for year, g in annual.groupby("Tahun"):
        row = {"Tahun": int(year)}
        for col in TARGETS:
            if col == "RR":
                row[col] = g[col].sum()
            else:
                row[col] = g[col].mean()
        annual_rows.append(row)
    annual_table = pd.DataFrame(annual_rows)

    display_table = annual_table.copy()
    for col in TARGETS:
        display_table[col] = display_table[col].round(2)
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    csv = annual_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Prediksi CSV",
        data=csv,
        file_name=f"prediksi_{station_name.lower().replace(' ', '_')}_2026_2055.csv",
        mime="text/csv",
    )

    st.caption(f"Cache hasil model: {cache_name}")


def show_validation(station_name, start_date, end_date):
    show_header()
    st.subheader("Validasi & Evaluasi Model")

    try:
        with st.spinner("Mengambil hasil evaluasi model ..."):
            _, _, _, metrics_df, _, _ = train_station(station_name, start_date, end_date)
    except Exception as e:
        st.error("Evaluasi belum dapat ditampilkan karena model gagal diproses.")
        st.exception(e)
        return

    shown = metrics_df.copy()
    shown["RMSE"] = shown["RMSE"].round(4)
    shown["MAE"] = shown["MAE"].round(4)
    shown["R²"] = shown["R²"].round(4)
    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.markdown(
        "**Keterangan:** RMSE dan MAE menunjukkan besarnya kesalahan prediksi, sedangkan R² menunjukkan proporsi variasi data uji yang dapat dijelaskan model."
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
    rows = []
    for code in TARGETS:
        name, unit = PARAMETER_INFO[code]
        rows.append({"Kode": code, "Parameter": name, "Satuan": unit})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "Dataset penelitian menggunakan parameter meteorologi: TN, TX, TAVG, RH_AVG, RR, SS, FF_X, dan FF_AVG."
    )


# ============================================================
# SIDEBAR + MAIN
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

if menu == "Dashboard":
    show_dashboard(station, start_date, end_date)
elif menu == "Validasi & Evaluasi":
    show_validation(station, start_date, end_date)
else:
    show_profile()
