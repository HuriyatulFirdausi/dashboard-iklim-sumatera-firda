import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Dashboard Peramalan Iklim Pesisir Sumatera",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #123b67;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 2px;
}
.sub-title {
    text-align: center;
    color: #666;
    font-size: 14px;
    margin-bottom: 25px;
}
.section-title {
    color: #173f6b;
    font-size: 21px;
    font-weight: 650;
    border-bottom: 1px solid #d9dfe7;
    padding-bottom: 7px;
    margin-top: 15px;
}
.info-box {
    padding: 13px 17px;
    border-radius: 8px;
    border: 1px solid #d8e3ef;
    background: #f5f9ff;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# KONFIGURASI PENELITIAN
# -----------------------------
STATIONS = {
    "Stasiun Minangkabau": "minangkabau",
    "Stasiun Pesawaran": "pesawaran",
    "Stasiun Maritim Panjang": "maritim_panjang",
}

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

TARGETS = ["TN", "TX", "TAVG", "RH_AVG", "RR", "SS", "FF_X", "FF_AVG"]

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def read_uploaded_file(uploaded_file, station_key):
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(uploaded_file, header=None)
        # NASA POWER data pada file Pesawaran mulai pada baris dengan YEAR, DOY...
        header_row = None
        for i in range(min(len(raw), 100)):
            vals = raw.iloc[i].astype(str).str.strip().tolist()
            if "YEAR" in vals and "DOY" in vals:
                header_row = i
                break
        if header_row is None:
            raise ValueError("Baris header YEAR/DOY tidak ditemukan pada file Excel.")
        df = pd.read_excel(uploaded_file, header=header_row)
    else:
        # NASA POWER CSV: 16 baris header metadata
        df = pd.read_csv(uploaded_file, skiprows=16)

    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "YEAR", "DOY", "T2M_MIN", "T2M_MAX", "T2M", "RH2M",
        "PRECTOTCORR", "WS10M_MAX", "WS2M", "ALLSKY_SFC_SW_DWN"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom NASA POWER tidak lengkap: {missing}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -999 adalah kode missing dari NASA POWER
    df = df.replace(-999, np.nan)

    # YEAR + DOY -> tanggal
    df["DATE"] = pd.to_datetime(
        df["YEAR"].astype("Int64").astype(str) + "-01-01",
        errors="coerce"
    ) + pd.to_timedelta(df["DOY"] - 1, unit="D")

    df = df.dropna(subset=["DATE"]).sort_values("DATE")
    df = df[(df["DATE"].dt.year >= 1985) & (df["DATE"].dt.year <= 2025)]

    # Standardisasi nama variabel penelitian
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
    df["STATION"] = station_key
    return df[["DATE", "STATION"] + TARGETS]


@st.cache_data
def to_monthly(df):
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.set_index("DATE")
    monthly = pd.DataFrame(index=x.resample("MS").size().index)
    monthly["TN"] = x["TN"].resample("MS").min()
    monthly["TX"] = x["TX"].resample("MS").max()
    monthly["TAVG"] = x["TAVG"].resample("MS").mean()
    monthly["RH_AVG"] = x["RH_AVG"].resample("MS").mean()
    monthly["RR"] = x["RR"].resample("MS").sum()
    monthly["SS"] = x["SS"].resample("MS").mean()
    monthly["FF_X"] = x["FF_X"].resample("MS").max()
    monthly["FF_AVG"] = x["FF_AVG"].resample("MS").mean()
    monthly = monthly.reset_index().rename(columns={"index": "DATE"})
    monthly["MONTH"] = monthly["DATE"].dt.month
    monthly["YEAR"] = monthly["DATE"].dt.year
    return monthly


@st.cache_data
def make_features(monthly):
    d = monthly.copy()
    # Lag sesuai rancangan penelitian
    for col in ["RR", "TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "FF_AVG"]:
        d[f"{col}_lag1"] = d[col].shift(1)

    for col in ["RR", "TN", "TX"]:
        d[f"{col}_lag2"] = d[col].shift(2)
        d[f"{col}_lag3"] = d[col].shift(3)

    # Representasi siklik bulan
    d["month_sin"] = np.sin(2 * np.pi * d["MONTH"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["MONTH"] / 12)
    return d.dropna().reset_index(drop=True)


def feature_columns():
    cols = [
        "RR_lag1", "RR_lag2", "RR_lag3",
        "TN_lag1", "TN_lag2", "TN_lag3",
        "TX_lag1", "TX_lag2", "TX_lag3",
        "TAVG_lag1", "RH_AVG_lag1", "SS_lag1",
        "FF_X_lag1", "FF_AVG_lag1",
        "month_sin", "month_cos",
    ]
    return cols


def recursive_forecast(monthly, years=30, window=6, n_estimators=300):
    """RF multi-output dengan recursive forecasting 360 bulan."""
    feat = make_features(monthly)
    fcols = feature_columns()

    # Time-series split: 80% train, 20% test.
    split = int(len(feat) * 0.80)
    train = feat.iloc[:split].copy()
    test = feat.iloc[split:].copy()

    X_train = train[fcols]
    y_train = train[TARGETS]
    X_test = test[fcols]
    y_test = test[TARGETS]

    # MinMaxScaler dipakai pada fitur dan target.
    sx = MinMaxScaler()
    sy = MinMaxScaler()
    Xtr = sx.fit_transform(X_train)
    ytr = sy.fit_transform(y_train)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    model.fit(Xtr, ytr)

    # Evaluasi pada data test
    pred_test_scaled = model.predict(sx.transform(X_test))
    pred_test = sy.inverse_transform(pred_test_scaled)

    metrics = []
    for i, target in enumerate(TARGETS):
        metrics.append({
            "Parameter": DISPLAY_COLUMNS[target],
            "RMSE": np.sqrt(mean_squared_error(y_test.iloc[:, i], pred_test[:, i])),
            "MAE": mean_absolute_error(y_test.iloc[:, i], pred_test[:, i]),
            "R²": r2_score(y_test.iloc[:, i], pred_test[:, i]),
        })
    metrics_df = pd.DataFrame(metrics)

    # Recursive forecast.
    history = monthly[TARGETS].copy().reset_index(drop=True)
    last_date = monthly["DATE"].iloc[-1]
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=years * 12,
        freq="MS",
    )

    future_rows = []
    for date in future_dates:
        row = {}

        def lag(col, k):
            return float(history[col].iloc[-k])

        row["RR_lag1"] = lag("RR", 1)
        row["RR_lag2"] = lag("RR", 2)
        row["RR_lag3"] = lag("RR", 3)

        row["TN_lag1"] = lag("TN", 1)
        row["TN_lag2"] = lag("TN", 2)
        row["TN_lag3"] = lag("TN", 3)

        row["TX_lag1"] = lag("TX", 1)
        row["TX_lag2"] = lag("TX", 2)
        row["TX_lag3"] = lag("TX", 3)

        for c in ["TAVG", "RH_AVG", "SS", "FF_X", "FF_AVG"]:
            row[f"{c}_lag1"] = lag(c, 1)

        m = date.month
        row["month_sin"] = np.sin(2 * np.pi * m / 12)
        row["month_cos"] = np.cos(2 * np.pi * m / 12)

        Xf = pd.DataFrame([[row[c] for c in fcols]], columns=fcols)
        pred_scaled = model.predict(sx.transform(Xf))
        pred = sy.inverse_transform(pred_scaled)[0]

        # RR tidak boleh negatif
        pred[TARGETS.index("RR")] = max(0, pred[TARGETS.index("RR")])

        future_rows.append([date] + list(pred))
        history.loc[len(history)] = pred

    future_df = pd.DataFrame(future_rows, columns=["DATE"] + TARGETS)
    return model, metrics_df, test[["DATE"] + TARGETS], pred_test, future_df


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Menu Navigasi")
page = st.sidebar.radio(
    "Pilih Tampilan:",
    ["🏠 Dashboard", "📊 Validasi & Evaluasi", "👤 Profil Peneliti"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Wilayah Pesisir")

selected_station = st.sidebar.selectbox(
    "Pilih wilayah:",
    list(STATIONS.keys())
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
st.sidebar.subheader("📁 Data NASA POWER")
up1 = st.sidebar.file_uploader(
    "Minangkabau (.csv)",
    type=["csv"],
    key="minang"
)
up2 = st.sidebar.file_uploader(
    "Pesawaran (.csv/.xlsx)",
    type=["csv", "xlsx"],
    key="pesawaran"
)
up3 = st.sidebar.file_uploader(
    "Maritim Panjang (.csv)",
    type=["csv"],
    key="maritim"
)

st.sidebar.caption(
    "Periode historis penelitian: 1985–2025. "
    "Forecast 30 tahun menghasilkan periode 2026–2055."
)

# -----------------------------
# LOAD FILES
# -----------------------------
files = {
    "minangkabau": up1,
    "pesawaran": up2,
    "maritim_panjang": up3,
}

datasets = {}
errors = []

for station_key, uploaded in files.items():
    if uploaded is not None:
        try:
            datasets[station_key] = read_uploaded_file(uploaded, station_key)
        except Exception as e:
            errors.append(f"{station_key}: {e}")

if errors:
    for e in errors:
        st.sidebar.error(e)

# Demo/in-repo paths, useful when files are placed beside app.py.
default_paths = {
    "minangkabau": "minang kabau data nasa.csv",
    "pesawaran": "pesawaran data nasa DAN bmkg(1).xlsx",
    "maritim_panjang": "maritim panjang data nasa.csv",
}
for key, path in default_paths.items():
    if key not in datasets:
        try:
            if Path(path).exists():
                class LocalFile:
                    def __init__(self, path):
                        self.name = Path(path).name
                        self._path = path
                    def read(self):
                        return open(self._path, "rb").read()
                # read directly to avoid Streamlit uploader dependency
                if path.endswith(".xlsx"):
                    raw = pd.read_excel(path, header=None)
                    header_row = next(
                        i for i in range(min(len(raw), 100))
                        if "YEAR" in raw.iloc[i].astype(str).str.strip().tolist()
                    )
                    df0 = pd.read_excel(path, header=header_row)
                else:
                    df0 = pd.read_csv(path, skiprows=16)
                df0.columns = [str(c).strip() for c in df0.columns]
                df0 = df0.replace(-999, np.nan)
                df0["DATE"] = pd.to_datetime(
                    df0["YEAR"].astype("Int64").astype(str) + "-01-01",
                    errors="coerce"
                ) + pd.to_timedelta(pd.to_numeric(df0["DOY"]) - 1, unit="D")
                df0 = df0.dropna(subset=["DATE"])
                df0 = df0[(df0["DATE"].dt.year >= 1985) & (df0["DATE"].dt.year <= 2025)]
                df0 = df0.rename(columns={
                    "T2M_MIN": "TN", "T2M_MAX": "TX", "T2M": "TAVG",
                    "RH2M": "RH_AVG", "PRECTOTCORR": "RR",
                    "ALLSKY_SFC_SW_DWN": "SS", "WS10M_MAX": "FF_X",
                    "WS2M": "FF_AVG"
                })
                datasets[key] = df0[["DATE"] + TARGETS].copy()
        except Exception as e:
            errors.append(f"File lokal {path}: {e}")

if selected_station_key := STATIONS.get(selected_station):
    daily = datasets.get(selected_station_key)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="main-title">DASHBOARD MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM WILAYAH PESISIR PANTAI PULAU SUMATERA</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025</div>',
    unsafe_allow_html=True
)

if daily is None:
    st.info(
        "Silakan upload ketiga dataset pada sidebar. "
        "Untuk Pesawaran, file Excel NASA POWER yang Anda gunakan dapat langsung diunggah."
    )
    st.stop()

# Filter periode
daily_filtered = daily[
    (daily["DATE"].dt.date >= start_date) &
    (daily["DATE"].dt.date <= end_date)
].copy()
monthly = to_monthly(daily_filtered)

# -----------------------------
# PAGE: DASHBOARD
# -----------------------------
if page == "🏠 Dashboard":
    st.markdown('<div class="section-title">👤 Profil Peneliti & Akademik</div>', unsafe_allow_html=True)

   col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background-color:#e7f1ff;
        padding:18px;
        border-radius:8px;
        min-height:150px;
    ">
    <b>Identitas Peneliti</b>
    <br><br>
    • <b>Nama Peneliti:</b> Huriyatul Firdausi
    <br>
    • <b>NIM:</b> 06111382328074
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background-color:#fffde7;
        padding:18px;
        border-radius:8px;
        min-height:150px;
    ">
    <b>Dosen Pembimbing</b>
    <br><br>
    • Dr. Melly Ariska, S.Pd., M.Sc.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background-color:#e8f8ed;
        padding:18px;
        border-radius:8px;
        min-height:150px;
    ">
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

    st.markdown('<div class="section-title">🛠️ Metadata Konfigurasi Model</div>', unsafe_allow_html=True)
    meta = pd.DataFrame({
        "Konfigurasi": [
            "Wilayah", "Model", "Window", "Forecast Horizon",
            "Periode Historis", "Parameter", "Scaling", "Evaluation Metrics"
        ],
        "Nilai": [
            selected_station, "Random Forest", "6 bulan",
            "30 Tahun (360 bulan)", "1985–2025",
            "TN, TX, TAVG, RH_AVG, RR, SS, FF_X, FF_AVG + lag features + sin/cos bulan",
            "MinMaxScaler", "RMSE, MAE, R²"
        ]
    })
    st.dataframe(meta, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">📈 Data Historis Bulanan</div>', unsafe_allow_html=True)

    parameter = st.selectbox(
        "Pilih parameter yang ditampilkan:",
        TARGETS,
        format_func=lambda x: DISPLAY_COLUMNS[x]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["DATE"],
        y=monthly[parameter],
        mode="lines",
        name=DISPLAY_COLUMNS[parameter],
    ))
    fig.update_layout(
        height=430,
        xaxis_title="Waktu",
        yaxis_title=f"{DISPLAY_COLUMNS[parameter]} ({UNITS[parameter]})",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📊 Statistik Data</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Jumlah Data Harian", f"{len(daily_filtered):,}")
    s2.metric("Jumlah Data Bulanan", f"{len(monthly):,}")
    s3.metric("Rata-rata", f"{monthly[parameter].mean():.2f}")
    s4.metric("Maksimum", f"{monthly[parameter].max():.2f}")

    st.markdown('<div class="section-title">🧠 Ringkasan Parameter Penelitian</div>', unsafe_allow_html=True)
    st.write(
        "Model menggunakan delapan parameter iklim utama, fitur lag 1–3 bulan "
        "untuk variabel terpilih, serta representasi siklik bulan menggunakan "
        "sinus dan cosinus. Random Forest digunakan untuk menangkap hubungan "
        "nonlinier antarvariabel dan pola temporal."
    )

# -----------------------------
# PAGE: VALIDASI
# -----------------------------
elif page == "📊 Validasi & Evaluasi":
    st.markdown('<div class="section-title">📊 Validasi & Evaluasi Model Random Forest</div>', unsafe_allow_html=True)

    st.write(
        "Klik tombol berikut untuk melatih model pada data historis wilayah yang dipilih. "
        "Pembagian data menggunakan urutan waktu 80% untuk pelatihan dan 20% untuk pengujian."
    )

    if len(monthly) < 60:
        st.warning("Data bulanan terlalu sedikit untuk pelatihan model.")
        st.stop()

    if st.button("🚀 Latih Model & Buat Prediksi 30 Tahun", type="primary"):
        with st.spinner("Melatih Random Forest dan membuat forecast 360 bulan..."):
            model, metrics_df, test_df, pred_test, future_df = recursive_forecast(
                monthly, years=30, window=6, n_estimators=300
            )
        st.session_state["model_result"] = {
            "model": model,
            "metrics": metrics_df,
            "test": test_df,
            "pred_test": pred_test,
            "future": future_df,
        }

    result = st.session_state.get("model_result")

    if result is None:
        st.info("Belum ada hasil. Tekan tombol **Latih Model & Buat Prediksi 30 Tahun**.")
    else:
        metrics_df = result["metrics"]
        future_df = result["future"]

        st.subheader("Hasil Evaluasi")
        st.dataframe(
            metrics_df.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Prediksi 30 Tahun (2026–2055)")
        forecast_parameter = st.selectbox(
            "Parameter prediksi:",
            TARGETS,
            format_func=lambda x: DISPLAY_COLUMNS[x],
            key="forecast_parameter"
        )

        hist_for_plot = monthly[["DATE", forecast_parameter]].copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist_for_plot["DATE"],
            y=hist_for_plot[forecast_parameter],
            mode="lines",
            name="Historis"
        ))
        fig2.add_trace(go.Scatter(
            x=future_df["DATE"],
            y=future_df[forecast_parameter],
            mode="lines",
            name="Prediksi 2026–2055"
        ))
        fig2.update_layout(
            height=480,
            xaxis_title="Waktu",
            yaxis_title=f"{DISPLAY_COLUMNS[forecast_parameter]} ({UNITS[forecast_parameter]})",
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Tabel Forecast")
        shown = future_df.copy()
        shown["Tahun"] = shown["DATE"].dt.year
        annual = shown.groupby("Tahun")[TARGETS].mean().reset_index()
        st.dataframe(
            annual.style.format({c: "{:.3f}" for c in TARGETS}),
            use_container_width=True,
            hide_index=True
        )

        csv = future_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download hasil prediksi CSV",
            csv,
            "prediksi_iklim_2026_2055.csv",
            "text/csv"
        )

# -----------------------------
# PAGE: PROFIL
# -----------------------------
else:
    st.markdown('<div class="section-title">👤 Profil Peneliti & Akademik</div>', unsafe_allow_html=True)
    st.markdown("""
    **Nama Peneliti:** Huriyatul Firdausi  
    **NIM:** 06111382328074  

    **Dosen Pembimbing:** Dr. Melly Ariska, S.Pd., M.Sc.

    **Program Studi:** Pendidikan Fisika  
    **Fakultas:** Keguruan dan Ilmu Pendidikan  
    **Universitas:** Universitas Sriwijaya  
    **Tahun:** 2026
    """)

    st.markdown('<div class="section-title">📚 Judul Penelitian</div>', unsafe_allow_html=True)
    st.write(
        "**MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM "
        "WILAYAH PESISIR PANTAI PULAU SUMATERA**"
    )
    st.caption("Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025")

    st.markdown('<div class="section-title">🌍 Wilayah Penelitian</div>', unsafe_allow_html=True)
    st.write(
        "Dashboard menyediakan tiga wilayah pesisir: **Stasiun Minangkabau, "
        "Stasiun Pesawaran, dan Stasiun Maritim Panjang**."
    )

    st.markdown('<div class="section-title">⚠️ Catatan Data</div>', unsafe_allow_html=True)
    st.warning(
        "Kolom NASA POWER ALLSKY_SFC_SW_DWN secara teknis merupakan "
        "radiasi gelombang pendek permukaan (MJ/m²/hari), bukan lama penyinaran "
        "matahari dalam jam. Jika dalam skripsi variabel tersebut disebut SS "
        "sebagai lama penyinaran, definisi dan satuannya sebaiknya diseragamkan "
        "dengan sumber data sebelum seminar/ujian."
    )
