import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# 1. HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Prediksi Iklim Sumatera",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. STYLE
# =========================================================
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #123b67;
    font-size: 30px;
    font-weight: 700;
    line-height: 1.35;
    margin-bottom: 4px;
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
    margin-top: 18px;
    margin-bottom: 12px;
}
.profile-box {
    padding: 18px;
    border-radius: 8px;
    min-height: 150px;
    border: 1px solid #d9dfe7;
}
.blue-box { background: #e7f1ff; }
.yellow-box { background: #fffde7; }
.green-box { background: #e8f8ed; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. KONFIGURASI PENELITIAN
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

STATIONS = {
    "Stasiun Minangkabau": "minang kabau data nasa.csv",
    "Stasiun Pesawaran": "pesawaran data nasa DAN bmkg(1).xlsx",
    "Stasiun Maritim Panjang": "maritim panjang data nasa.csv",
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
# 4. PEMBACAAN DATA NASA POWER
# =========================================================
REQUIRED = [
    "YEAR", "DOY", "T2M_MIN", "T2M_MAX", "T2M", "RH2M",
    "PRECTOTCORR", "WS10M_MAX", "WS2M", "ALLSKY_SFC_SW_DWN"
]


def find_csv_header(path):
    """Cari baris header NASA POWER tanpa mengandalkan nomor baris."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            fields = [x.strip().upper() for x in line.strip().split(",")]
            if len(fields) >= 2 and fields[0] == "YEAR" and fields[1] == "DOY":
                return i
    raise ValueError("Header YEAR,DOY tidak ditemukan pada CSV.")


def find_excel_header(path):
    """Cari baris header YEAR/DOY pada Excel."""
    raw = pd.read_excel(path, header=None)
    for i in range(len(raw)):
        fields = [str(x).strip().upper() for x in raw.iloc[i].tolist()]
        if "YEAR" in fields and "DOY" in fields:
            return i
    raise ValueError("Header YEAR/DOY tidak ditemukan pada Excel.")


def standardize(df):
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Kolom NASA POWER tidak lengkap: " + ", ".join(missing)
        )

    df = df[REQUIRED].copy()

    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # NASA POWER menggunakan -999 sebagai nilai missing.
    df = df.replace(-999, np.nan)

    df["DATE"] = (
        pd.to_datetime(
            df["YEAR"].astype("Int64").astype(str) + "-01-01",
            errors="coerce"
        )
        + pd.to_timedelta(df["DOY"] - 1, unit="D")
    )

    df = df.dropna(subset=["DATE"])

    # Periode historis penelitian
    df = df[
        (df["DATE"] >= pd.Timestamp("1985-01-01")) &
        (df["DATE"] <= pd.Timestamp("2025-12-31"))
    ].copy()

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

    return df[["DATE"] + TARGETS].sort_values("DATE").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_station(station_name):
    path = BASE_DIR / STATIONS[station_name]

    if not path.exists():
        raise FileNotFoundError(
            f"File '{path.name}' tidak ditemukan di repository GitHub."
        )

    if path.suffix.lower() == ".csv":
        header = find_csv_header(path)
        df = pd.read_csv(path, skiprows=header)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        header = find_excel_header(path)
        df = pd.read_excel(path, header=header)
    else:
        raise ValueError("Format file tidak didukung.")

    return standardize(df)


# =========================================================
# 5. DATA BULANAN
# =========================================================
@st.cache_data(show_spinner=False)
def monthly_data(daily):
    x = daily.set_index("DATE")

    monthly = pd.DataFrame(index=x.resample("MS").size().index)

    monthly["TN"] = x["TN"].resample("MS").min()
    monthly["TX"] = x["TX"].resample("MS").max()
    monthly["TAVG"] = x["TAVG"].resample("MS").mean()
    monthly["RH_AVG"] = x["RH_AVG"].resample("MS").mean()

    # RR NASA POWER = mm/hari, sehingga dijumlahkan menjadi mm/bulan.
    monthly["RR"] = x["RR"].resample("MS").sum()

    monthly["SS"] = x["SS"].resample("MS").mean()
    monthly["FF_X"] = x["FF_X"].resample("MS").max()
    monthly["FF_AVG"] = x["FF_AVG"].resample("MS").mean()

    return monthly.reset_index().rename(columns={"index": "DATE"})


# =========================================================
# 6. FEATURE ENGINEERING
# =========================================================
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


def create_features(monthly):
    d = monthly.copy()

    for lag in [1, 2, 3]:
        d[f"RR_lag{lag}"] = d["RR"].shift(lag)

    for col in ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "FF_AVG"]:
        d[f"{col}_lag1"] = d[col].shift(1)

    d["MONTH"] = d["DATE"].dt.month
    d["month_sin"] = np.sin(2 * np.pi * d["MONTH"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["MONTH"] / 12)

    return d.dropna().reset_index(drop=True)


# =========================================================
# 7. RANDOM FOREST + EVALUASI
# =========================================================
@st.cache_data(show_spinner=False)
def train_random_forest(monthly):
    data = create_features(monthly)

    if len(data) < 60:
        raise ValueError(
            "Data bulanan kurang dari 60 baris. "
            "Gunakan rentang historis yang lebih panjang."
        )

    # 80% train dan 20% test, urutan waktu dipertahankan.
    split = int(len(data) * 0.80)

    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    X_train = train[FEATURES]
    y_train = train[TARGETS]
    X_test = test[FEATURES]
    y_test = test[TARGETS]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    y_train_s = scaler_y.fit_transform(y_train)

    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_s, y_train_s)

    pred_s = model.predict(scaler_x.transform(X_test))
    pred = scaler_y.inverse_transform(pred_s)

    metrics = []

    for i, target in enumerate(TARGETS):
        actual = y_test[target].values
        prediction = pred[:, i]

        metrics.append({
            "Parameter": DISPLAY_COLUMNS[target],
            "RMSE": np.sqrt(mean_squared_error(actual, prediction)),
            "MAE": mean_absolute_error(actual, prediction),
            "R²": r2_score(actual, prediction),
        })

    return (
        model,
        scaler_x,
        scaler_y,
        pd.DataFrame(metrics),
    )


# =========================================================
# 8. FORECAST 2026–2055
# =========================================================
def forecast_360(monthly, model, scaler_x, scaler_y):
    history = monthly[TARGETS].copy().reset_index(drop=True)

    last_date = monthly["DATE"].max()

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=360,
        freq="MS",
    )

    results = []

    for date in future_dates:
        row = {
            "RR_lag1": history["RR"].iloc[-1],
            "RR_lag2": history["RR"].iloc[-2],
            "RR_lag3": history["RR"].iloc[-3],
            "TN_lag1": history["TN"].iloc[-1],
            "TX_lag1": history["TX"].iloc[-1],
            "TAVG_lag1": history["TAVG"].iloc[-1],
            "RH_AVG_lag1": history["RH_AVG"].iloc[-1],
            "SS_lag1": history["SS"].iloc[-1],
            "FF_X_lag1": history["FF_X"].iloc[-1],
            "FF_AVG_lag1": history["FF_AVG"].iloc[-1],
            "month_sin": np.sin(2 * np.pi * date.month / 12),
            "month_cos": np.cos(2 * np.pi * date.month / 12),
        }

        X = pd.DataFrame(
            [[row[c] for c in FEATURES]],
            columns=FEATURES,
        )

        pred_s = model.predict(scaler_x.transform(X))
        pred = scaler_y.inverse_transform(pred_s)[0]

        # RR tidak boleh negatif.
        pred[TARGETS.index("RR")] = max(
            0, pred[TARGETS.index("RR")]
        )

        results.append([date] + list(pred))

        # Recursive forecasting: prediksi bulan ini menjadi input bulan berikutnya.
        history.loc[len(history)] = pred

    return pd.DataFrame(
        results,
        columns=["DATE"] + TARGETS,
    )


# =========================================================
# 9. SIDEBAR
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
    key="start_date",
)

end_date = st.sidebar.date_input(
    "Selesai",
    value=pd.Timestamp("2025-12-31").date(),
    min_value=pd.Timestamp("1985-01-01").date(),
    max_value=pd.Timestamp("2025-12-31").date(),
    format="YYYY/MM/DD",
    key="end_date",
)

if start_date > end_date:
    st.sidebar.error("Tanggal mulai harus lebih awal dari tanggal selesai.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Data historis: 1985–2025")
st.sidebar.caption("Forecast: 2026–2055 (360 bulan)")


# =========================================================
# 10. LOAD DATA
# =========================================================
try:
    daily = load_station(selected_station)
except Exception as e:
    st.error(f"Data untuk {selected_station} belum dapat dibaca.")
    st.exception(e)
    st.stop()

monthly_full = monthly_data(daily)

# Data untuk tampilan mengikuti date picker.
monthly_view = monthly_full[
    (monthly_full["DATE"].dt.date >= start_date) &
    (monthly_full["DATE"].dt.date <= end_date)
].copy()

if monthly_view.empty:
    st.warning("Tidak ada data bulanan pada rentang yang dipilih.")
    st.stop()


# =========================================================
# 11. HEADER
# =========================================================
st.markdown(
    '<div class="main-title">'
    'MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM '
    'WILAYAH PESISIR PANTAI PULAU SUMATERA'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 12. DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="profile-box blue-box">
        <b>Identitas Peneliti</b>
        <br><br>
        • <b>Nama Peneliti:</b> Huriyatul Firdausi
        <br>
        • <b>NIM:</b> 06111382328074
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="profile-box yellow-box">
        <b>Dosen Pembimbing</b>
        <br><br>
        • Dr. Melly Ariska, S.Pd., M.Sc.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="profile-box green-box">
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
            "TN, TX, TAVG, RH_AVG, RR, SS, FF_X, FF_AVG + lag 1–3 + sin/cos bulan",
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
        "Pilih Parameter Iklim:",
        TARGETS,
        format_func=lambda x: DISPLAY_COLUMNS[x],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_view["DATE"],
            y=monthly_view[parameter],
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

    a, b, c, d = st.columns(4)

    a.metric("Data Harian", f"{len(daily):,}")
    b.metric("Data Bulanan", f"{len(monthly_view):,}")
    c.metric("Rata-rata", f"{monthly_view[parameter].mean():.2f}")
    d.metric("Maksimum", f"{monthly_view[parameter].max():.2f}")


# =========================================================
# 13. VALIDASI & EVALUASI
# =========================================================
elif page == "📊 Validasi & Evaluasi":

    st.markdown(
        '<div class="section-title">📊 Validasi & Evaluasi Random Forest</div>',
        unsafe_allow_html=True,
    )

    st.info(
        f"Wilayah aktif: **{selected_station}**. "
        "Model menggunakan data historis lengkap 1985–2025 untuk pelatihan "
        "dan evaluasi. Date picker digunakan untuk tampilan data historis."
    )

    if st.button(
        f"🚀 Latih Model {selected_station} & Buat Prediksi 30 Tahun",
        type="primary",
    ):

        with st.spinner(
            f"Melatih Random Forest untuk {selected_station}..."
        ):
            try:
                model, scaler_x, scaler_y, metrics = train_random_forest(
                    monthly_full
                )

                future = forecast_360(
                    monthly_full,
                    model,
                    scaler_x,
                    scaler_y,
                )

                # Hasil disimpan bersama nama wilayah.
                st.session_state["result"] = {
                    "station": selected_station,
                    "metrics": metrics,
                    "future": future,
                }

            except Exception as e:
                st.error("Terjadi kesalahan saat melatih model.")
                st.exception(e)

    result = st.session_state.get("result")

    # Jangan tampilkan hasil wilayah lama.
    if result is not None and result.get("station") != selected_station:
        result = None
        st.session_state.pop("result", None)

    if result is None:

        st.warning(
            f"Belum ada hasil prediksi untuk **{selected_station}**. "
            "Klik tombol di atas untuk menjalankan Random Forest."
        )

    else:

        metrics = result["metrics"]
        future = result["future"]

        st.success(
            f"Hasil model yang ditampilkan adalah khusus **{selected_station}**."
        )

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
            key=f"forecast_parameter_{selected_station}",
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_full["DATE"],
                y=monthly_full[forecast_parameter],
                mode="lines",
                name="Historis 1985–2025",
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
            height=480,
            xaxis_title="Waktu",
            yaxis_title=(
                f"{DISPLAY_COLUMNS[forecast_parameter]} "
                f"({UNITS[forecast_parameter]})"
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------
        # TABEL FORECAST TAHUNAN
        # ---------------------------------------------
        st.subheader("Tabel Forecast Tahunan")

        annual = future.copy()
        annual["Tahun"] = annual["DATE"].dt.year

        # Untuk parameter klimatologis:
        # mean = TN, TX, TAVG, RH_AVG, SS, FF_X, FF_AVG
        # sum  = RR karena RR adalah akumulasi curah hujan bulanan.
        annual_mean = annual.groupby("Tahun")[
            ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "FF_AVG"]
        ].mean()

        annual_rr = annual.groupby("Tahun")["RR"].sum()

        annual_table = annual_mean.copy()
        annual_table["RR"] = annual_rr
        annual_table = annual_table.reset_index()

        annual_table = annual_table[
            ["Tahun", "TN", "TX", "TAVG", "RH_AVG", "RR", "SS", "FF_X", "FF_AVG"]
        ]

        st.dataframe(
            annual_table.style.format({
                "TN": "{:.3f}",
                "TX": "{:.3f}",
                "TAVG": "{:.3f}",
                "RH_AVG": "{:.3f}",
                "RR": "{:.3f}",
                "SS": "{:.3f}",
                "FF_X": "{:.3f}",
                "FF_AVG": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv = future.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv,
            file_name=(
                "forecast_"
                + selected_station.lower().replace(" ", "_")
                + "_2026_2055.csv"
            ),
            mime="text/csv",
        )


# =========================================================
# 14. PROFIL PENELITI
# =========================================================
else:

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="profile-box blue-box">
        <b>Identitas Peneliti</b>
        <br><br>
        • <b>Nama Peneliti:</b> Huriyatul Firdausi
        <br>
        • <b>NIM:</b> 06111382328074
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="profile-box yellow-box">
        <b>Dosen Pembimbing</b>
        <br><br>
        • Dr. Melly Ariska, S.Pd., M.Sc.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="profile-box green-box">
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
        "Penelitian mencakup tiga wilayah pesisir: Stasiun Minangkabau, "
        "Stasiun Pesawaran, dan Stasiun Maritim Panjang."
    )

    st.markdown(
        '<div class="section-title">📌 Konfigurasi Penelitian</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Model: Random Forest | Window: 6 bulan | Forecast Horizon: "
        "30 tahun (360 bulan) | Data historis: 1985–2025 | "
        "Prediksi: 2026–2055 | Scaling: MinMaxScaler | "
        "Evaluasi: RMSE, MAE, R²."
    )

    st.markdown(
        '<div class="section-title">⚠️ Keterangan Variabel SS</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Pada NASA POWER, ALLSKY_SFC_SW_DWN merupakan radiasi gelombang "
        "pendek permukaan (MJ/m²/hari). Variabel ini bukan lama penyinaran "
        "matahari dalam jam. Definisi dan satuannya sebaiknya diseragamkan "
        "dengan metodologi penelitian."
    )
