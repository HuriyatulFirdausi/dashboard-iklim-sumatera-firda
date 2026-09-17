import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Prediksi Iklim Sumatera",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #123b67;
    font-size: 29px;
    font-weight: 700;
    line-height: 1.35;
    margin-bottom: 4px;
}
.sub-title {
    text-align: center;
    color: #666666;
    font-size: 14px;
    margin-bottom: 24px;
}
.section-title {
    color: #173f6b;
    font-size: 21px;
    font-weight: 650;
    border-bottom: 1px solid #d9dfe7;
    padding-bottom: 7px;
    margin-top: 16px;
    margin-bottom: 12px;
}
.profile-box {
    padding: 17px 18px;
    border-radius: 8px;
    min-height: 145px;
    border: 1px solid #d9dfe7;
}
.box-blue {
    background: #e7f1ff;
}
.box-yellow {
    background: #fffde7;
}
.box-green {
    background: #e8f8ed;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# KONFIGURASI PENELITIAN
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

STATIONS = {
    "Stasiun Minangkabau": {
        "key": "minangkabau",
        "file": "minang kabau data nasa.csv",
    },
    "Stasiun Pesawaran": {
        "key": "pesawaran",
        "file": "pesawaran data nasa DAN bmkg(1).xlsx",
    },
    "Stasiun Maritim Panjang": {
        "key": "maritim_panjang",
        "file": "maritim panjang data nasa.csv",
    },
}

TARGETS = ["TN", "TX", "TAVG", "RH_AVG", "RR", "SS", "FF_X", "FF_AVG"]

DISPLAY_COLUMNS = {
    "TN": "Temperatur Minimum (TN)",
    "TX": "Temperatur Maksimum (TX)",
    "TAVG": "Temperatur Rata-rata (TAVG)",
    "RH_AVG": "Kelembapan Relatif Rata-rata (RH_AVG)",
    "RR": "Curah Hujan (RR)",
    "SS": "Radiasi Surya (SS)",
    "FF_X": "Kecepatan Angin Maksimum (FF_X)",
    "FF_AVG": "Kecepatan Angin Rata-rata (FF_AVG)",
}

UNITS = {
    "TN": "°C",
    "TX": "°C",
    "TAVG": "°C",
    "RH_AVG": "%",
    "RR": "mm/bulan",
    "SS": "MJ/m²/hari",
    "FF_X": "m/s",
    "FF_AVG": "m/s",
}

# =========================================================
# PEMBACAAN DATA
# =========================================================
def find_csv_header(path):
    """Mencari baris header NASA POWER secara otomatis."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            first = line.strip().split(",")[0].strip().upper()
            if first == "YEAR":
                return i
    raise ValueError("Header YEAR pada file CSV NASA POWER tidak ditemukan.")


def read_nasa_csv(path):
    header_row = find_csv_header(path)
    df = pd.read_csv(path, skiprows=header_row)

    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "YEAR", "DOY", "T2M_MIN", "T2M_MAX", "T2M", "RH2M",
        "PRECTOTCORR", "WS10M_MAX", "WS2M", "ALLSKY_SFC_SW_DWN"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom NASA POWER tidak lengkap: {missing}")

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace(-999, np.nan)

    df["DATE"] = (
        pd.to_datetime(
            df["YEAR"].astype("Int64").astype(str) + "-01-01",
            errors="coerce"
        )
        + pd.to_timedelta(df["DOY"] - 1, unit="D")
    )

    df = df.dropna(subset=["DATE"])

    return standardize_columns(df)


def find_excel_header(path):
    raw = pd.read_excel(path, header=None)
    for i in range(len(raw)):
        values = raw.iloc[i].astype(str).str.strip().str.upper().tolist()
        if "YEAR" in values and "DOY" in values:
            return i
    raise ValueError("Header YEAR/DOY pada file Excel tidak ditemukan.")


def read_nasa_excel(path):
    header_row = find_excel_header(path)
    df = pd.read_excel(path, header=header_row)

    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "YEAR", "DOY", "T2M_MIN", "T2M_MAX", "T2M", "RH2M",
        "PRECTOTCORR", "WS10M_MAX", "WS2M", "ALLSKY_SFC_SW_DWN"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom NASA POWER tidak lengkap: {missing}")

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace(-999, np.nan)

    df["DATE"] = (
        pd.to_datetime(
            df["YEAR"].astype("Int64").astype(str) + "-01-01",
            errors="coerce"
        )
        + pd.to_timedelta(df["DOY"] - 1, unit="D")
    )

    df = df.dropna(subset=["DATE"])

    return standardize_columns(df)


def standardize_columns(df):
    df = df.rename(columns={
        "T2M_MIN": "TN",
        "T2M_MAX": "TX",
        "T2M": "TAVG",
        "RH2M": "RH_AVG",
        "PRECTOTCORR": "RR",
        "ALLSKY_SFC_SW_DWN": "SS",
        "WS10M_MAX": "FF_X",
        "WS2M": "FF_AVG",
    })

    # Periode penelitian
    df = df[
        (df["DATE"] >= pd.Timestamp("1985-01-01")) &
        (df["DATE"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    return df[["DATE"] + TARGETS].sort_values("DATE").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_station(station_name):
    info = STATIONS[station_name]
    path = BASE_DIR / info["file"]

    if not path.exists():
        raise FileNotFoundError(
            f"File '{info['file']}' tidak ditemukan di repository."
        )

    if path.suffix.lower() == ".csv":
        return read_nasa_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return read_nasa_excel(path)

    raise ValueError(f"Format file tidak didukung: {path.suffix}")


# =========================================================
# TRANSFORMASI HARIAN -> BULANAN
# =========================================================
@st.cache_data(show_spinner=False)
def to_monthly(df):
    x = df.set_index("DATE")

    monthly = pd.DataFrame()
    monthly["TN"] = x["TN"].resample("MS").min()
    monthly["TX"] = x["TX"].resample("MS").max()
    monthly["TAVG"] = x["TAVG"].resample("MS").mean()
    monthly["RH_AVG"] = x["RH_AVG"].resample("MS").mean()

    # Curah hujan harian dijumlahkan menjadi akumulasi bulanan
    monthly["RR"] = x["RR"].resample("MS").sum()

    monthly["SS"] = x["SS"].resample("MS").mean()
    monthly["FF_X"] = x["FF_X"].resample("MS").max()
    monthly["FF_AVG"] = x["FF_AVG"].resample("MS").mean()

    monthly = monthly.reset_index()
    monthly["MONTH"] = monthly["DATE"].dt.month
    monthly["YEAR"] = monthly["DATE"].dt.year

    return monthly


# =========================================================
# FEATURE ENGINEERING
# =========================================================
@st.cache_data(show_spinner=False)
def make_features(monthly):
    d = monthly.copy()

    # RR lag 1, 2, 3
    for lag in [1, 2, 3]:
        d[f"RR_lag{lag}"] = d["RR"].shift(lag)

    # Parameter lain menggunakan lag 1
    for col in ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "FF_AVG"]:
        d[f"{col}_lag1"] = d[col].shift(1)

    # Representasi siklik bulan
    d["month_sin"] = np.sin(2 * np.pi * d["MONTH"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["MONTH"] / 12)

    return d.dropna().reset_index(drop=True)


FEATURES = [
    "RR_lag1", "RR_lag2", "RR_lag3",
    "TN_lag1",
    "TX_lag1",
    "TAVG_lag1",
    "RH_AVG_lag1",
    "SS_lag1",
    "FF_X_lag1",
    "FF_AVG_lag1",
    "month_sin",
    "month_cos",
]


# =========================================================
# MODEL RANDOM FOREST
# =========================================================
@st.cache_data(show_spinner=False)
def train_model(monthly):
    data = make_features(monthly)

    if len(data) < 60:
        raise ValueError("Data bulanan tidak cukup untuk melatih model.")

    # Time series split, tanpa mengacak urutan waktu
    split = int(len(data) * 0.80)

    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    X_train = train[FEATURES]
    y_train = train[TARGETS]
    X_test = test[FEATURES]
    y_test = test[TARGETS]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train_scaled)

    test_prediction_scaled = model.predict(
        scaler_x.transform(X_test)
    )
    test_prediction = scaler_y.inverse_transform(test_prediction_scaled)

    metric_rows = []

    for i, target in enumerate(TARGETS):
        actual = y_test.iloc[:, i].values
        predicted = test_prediction[:, i]

        metric_rows.append({
            "Parameter": DISPLAY_COLUMNS[target],
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAE": mean_absolute_error(actual, predicted),
            "R²": r2_score(actual, predicted),
        })

    metrics = pd.DataFrame(metric_rows)

    return model, scaler_x, scaler_y, test, test_prediction, metrics


# =========================================================
# FORECAST 30 TAHUN / 360 BULAN
# =========================================================
def forecast_30_years(monthly, model, scaler_x, scaler_y):
    history = monthly[TARGETS].copy().reset_index(drop=True)

    last_date = monthly["DATE"].iloc[-1]

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=360,
        freq="MS",
    )

    predictions = []

    for date in future_dates:
        row = {}

        # RR lag 1, 2, 3
        row["RR_lag1"] = history["RR"].iloc[-1]
        row["RR_lag2"] = history["RR"].iloc[-2]
        row["RR_lag3"] = history["RR"].iloc[-3]

        # Parameter lainnya lag 1
        for col in ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "FF_AVG"]:
            row[f"{col}_lag1"] = history[col].iloc[-1]

        row["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * date.month / 12)

        X_future = pd.DataFrame(
            [[row[col] for col in FEATURES]],
            columns=FEATURES,
        )

        prediction_scaled = model.predict(
            scaler_x.transform(X_future)
        )

        prediction = scaler_y.inverse_transform(prediction_scaled)[0]

        # Curah hujan tidak boleh negatif
        rr_index = TARGETS.index("RR")
        prediction[rr_index] = max(0, prediction[rr_index])

        predictions.append([date] + list(prediction))

        # Hasil prediksi menjadi input untuk bulan berikutnya
        history.loc[len(history)] = prediction

    return pd.DataFrame(
        predictions,
        columns=["DATE"] + TARGETS,
    )


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 📌 Menu Navigasi")

page = st.sidebar.radio(
    "Pilih Tampilan:",
    [
        "🏠 Dashboard",
        "📊 Validasi & Evaluasi",
        "👤 Profil Peneliti",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🌍 Wilayah Pesisir")

selected_station = st.sidebar.selectbox(
    "Pilih wilayah:",
    list(STATIONS.keys()),
)

st.sidebar.markdown("### 📅 Rentang Waktu")

start_date = st.sidebar.date_input(
    "Mulai",
    value=pd.Timestamp("1985-01-01").date(),
    min_value=pd.Timestamp("1985-01-01").date(),
    max_value=pd.Timestamp("2025-12-31").date(),
    format="YYYY/MM/DD",
)

end_date = st.sidebar.date_input(
    "Selesai",
    value=pd.Timestamp("2025-12-31").date(),
    min_value=pd.Timestamp("1985-01-01").date(),
    max_value=pd.Timestamp("2025-12-31").date(),
    format="YYYY/MM/DD",
)

if start_date > end_date:
    st.sidebar.error("Tanggal mulai harus lebih awal dari tanggal selesai.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Data historis: 1985–2025")
st.sidebar.caption("Forecast: 2026–2055 (360 bulan)")


# =========================================================
# LOAD DATA OTOMATIS DARI REPOSITORY
# =========================================================
try:
    daily = load_station(selected_station)
except Exception as e:
    st.error(f"Data {selected_station} belum dapat dibaca.")
    st.exception(e)
    st.stop()

daily_filtered = daily[
    (daily["DATE"].dt.date >= start_date) &
    (daily["DATE"].dt.date <= end_date)
].copy()

if daily_filtered.empty:
    st.warning("Tidak ada data pada rentang waktu yang dipilih.")
    st.stop()

monthly = to_monthly(daily_filtered)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM WILAYAH PESISIR PANTAI PULAU SUMATERA</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025</div>',
    unsafe_allow_html=True,
)


# =========================================================
# PAGE: DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="profile-box box-blue">
        <b>Identitas Peneliti</b>
        <br><br>
        • <b>Nama Peneliti:</b> Huriyatul Firdausi
        <br>
        • <b>NIM:</b> 06111382328074
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="profile-box box-yellow">
        <b>Dosen Pembimbing</b>
        <br><br>
        • Dr. Melly Ariska, S.Pd., M.Sc.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="profile-box box-green">
        <b>Informasi Akademik</b>
        <br><br>
        • <b>Program Studi:</b> Pendidikan Fisika
        <br>
        • <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan
        <br>
        • <b>Universitas:</b> Universitas Sriwijaya
        <br>
        • <b>Tahun:</b> 2026
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🛠️ Metadata Konfigurasi Model</div>',
        unsafe_allow_html=True,
    )

    meta = pd.DataFrame({
        "Konfigurasi": [
            "Wilayah",
            "Model",
            "Window",
            "Forecast Horizon",
            "Periode Historis",
            "Parameter",
            "Scaling",
            "Evaluation Metrics",
        ],
        "Nilai": [
            selected_station,
            "Random Forest",
            "6 bulan",
            "30 Tahun (360 bulan)",
            "1985–2025",
            "TN, TX, TAVG, RH_AVG, RR, SS, FF_X, FF_AVG + lag features + sin/cos bulan",
            "MinMaxScaler",
            "RMSE, MAE, R²",
        ],
    })

    st.dataframe(
        meta,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">📈 Data Historis Bulanan</div>',
        unsafe_allow_html=True,
    )

    parameter = st.selectbox(
        "Pilih parameter:",
        TARGETS,
        format_func=lambda x: DISPLAY_COLUMNS[x],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["DATE"],
            y=monthly[parameter],
            mode="lines",
            name=DISPLAY_COLUMNS[parameter],
        )
    )

    fig.update_layout(
        height=430,
        xaxis_title="Waktu",
        yaxis_title=f"{DISPLAY_COLUMNS[parameter]} ({UNITS[parameter]})",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">📊 Statistik Data</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Data Harian", f"{len(daily_filtered):,}")
    c2.metric("Data Bulanan", f"{len(monthly):,}")
    c3.metric("Rata-rata", f"{monthly[parameter].mean():.2f}")
    c4.metric("Maksimum", f"{monthly[parameter].max():.2f}")


# =========================================================
# PAGE: VALIDASI & EVALUASI
# =========================================================
elif page == "📊 Validasi & Evaluasi":

    st.markdown(
        '<div class="section-title">📊 Validasi & Evaluasi Random Forest</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Model dilatih menggunakan urutan waktu: 80% data untuk pelatihan "
        "dan 20% data untuk pengujian. Data tidak diacak agar karakteristik "
        "time series tetap dipertahankan."
    )

    if st.button(
        "🚀 Latih Model & Buat Prediksi 30 Tahun",
        type="primary",
    ):

        with st.spinner(
            "Melatih Random Forest dan membuat forecast 360 bulan..."
        ):
            try:
                (
                    model,
                    scaler_x,
                    scaler_y,
                    test,
                    test_prediction,
                    metrics,
                ) = train_model(monthly)

                future = forecast_30_years(
                    monthly,
                    model,
                    scaler_x,
                    scaler_y,
                )

                st.session_state["result"] = {
                    "metrics": metrics,
                    "future": future,
                }

            except Exception as e:
                st.error("Terjadi kesalahan saat melatih model.")
                st.exception(e)

    result = st.session_state.get("result")

    if result is None:
        st.info(
            "Klik **Latih Model & Buat Prediksi 30 Tahun** "
            "untuk menghasilkan evaluasi dan forecast."
        )
    else:
        metrics = result["metrics"]
        future = result["future"]

        st.subheader("Hasil Evaluasi Model")

        st.dataframe(
            metrics.style.format({
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
                "R²": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Prediksi 30 Tahun (2026–2055)")

        forecast_parameter = st.selectbox(
            "Pilih parameter prediksi:",
            TARGETS,
            format_func=lambda x: DISPLAY_COLUMNS[x],
            key="forecast_parameter",
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly["DATE"],
                y=monthly[forecast_parameter],
                mode="lines",
                name="Historis",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future["DATE"],
                y=future[forecast_parameter],
                mode="lines",
                name="Prediksi 2026–2055",
            )
        )

        fig.update_layout(
            height=470,
            xaxis_title="Waktu",
            yaxis_title=(
                f"{DISPLAY_COLUMNS[forecast_parameter]} "
                f"({UNITS[forecast_parameter]})"
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tabel Forecast Tahunan")

        annual = future.copy()
        annual["Tahun"] = annual["DATE"].dt.year

        annual = (
            annual.groupby("Tahun")[TARGETS]
            .mean()
            .reset_index()
        )

        st.dataframe(
            annual.style.format({
                col: "{:.3f}" for col in TARGETS
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv = future.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv,
            file_name="prediksi_iklim_2026_2055.csv",
            mime="text/csv",
        )


# =========================================================
# PAGE: PROFIL
# =========================================================
else:

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="profile-box box-blue">
        <b>Identitas Peneliti</b>
        <br><br>
        • <b>Nama Peneliti:</b> Huriyatul Firdausi
        <br>
        • <b>NIM:</b> 06111382328074
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="profile-box box-yellow">
        <b>Dosen Pembimbing</b>
        <br><br>
        • Dr. Melly Ariska, S.Pd., M.Sc.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="profile-box box-green">
        <b>Informasi Akademik</b>
        <br><br>
        • <b>Program Studi:</b> Pendidikan Fisika
        <br>
        • <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan
        <br>
        • <b>Universitas:</b> Universitas Sriwijaya
        <br>
        • <b>Tahun:</b> 2026
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📚 Judul Penelitian</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "**MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM "
        "WILAYAH PESISIR PANTAI PULAU SUMATERA**"
    )

    st.caption(
        "Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025"
    )

    st.markdown(
        '<div class="section-title">🌍 Wilayah Penelitian</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Penelitian mencakup tiga wilayah pesisir, yaitu Stasiun "
        "Minangkabau, Stasiun Pesawaran, dan Stasiun Maritim Panjang."
    )

    st.markdown(
        '<div class="section-title">⚠️ Keterangan Variabel SS</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Pada data NASA POWER, ALLSKY_SFC_SW_DWN merupakan radiasi "
        "gelombang pendek permukaan dengan satuan MJ/m²/hari. "
        "Jika variabel ini akan disebut SS (lama penyinaran matahari), "
        "definisi dan satuannya perlu diseragamkan dengan metodologi penelitian."
    )
