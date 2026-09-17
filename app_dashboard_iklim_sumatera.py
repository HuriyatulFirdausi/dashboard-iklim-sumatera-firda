import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Dashboard Iklim Sumatera",
    page_icon="🌍",
    layout="wide",
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.main-title{
    text-align:center;color:#123b67;font-size:30px;
    font-weight:700;line-height:1.35;margin-bottom:4px;
}
.sub-title{
    text-align:center;color:#666;font-size:14px;margin-bottom:25px;
}
.section-title{
    color:#173f6b;font-size:21px;font-weight:650;
    border-bottom:1px solid #d9dfe7;padding-bottom:7px;
    margin-top:18px;margin-bottom:12px;
}
.profile-box{
    padding:18px;border-radius:8px;min-height:155px;
    border:1px solid #d9dfe7;
}
.blue-box{background:#e7f1ff;}
.yellow-box{background:#fffde7;}
.green-box{background:#e8f8ed;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
FORECAST_DIR = BASE_DIR / "hasil_prediksi"

STATIONS = {
    "Stasiun Minangkabau": "minang kabau data FIX.xlsx",
    "Stasiun Pesawaran": "pesawaran data FIX.xlsx",
    "Stasiun Maritim Panjang": "Maritim panjang data FIX.xlsx",
}

TARGETS = ["TN","TX","TAVG","RH_AVG","RR","SS","FF_X","FF_AVG"]

DISPLAY_COLUMNS = {
    "TN":"Temperatur Minimum (TN)",
    "TX":"Temperatur Maksimum (TX)",
    "TAVG":"Temperatur Rata-rata (TAVG)",
    "RH_AVG":"Kelembapan Relatif Rata-rata (RH_AVG)",
    "RR":"Curah Hujan (RR)",
    "SS":"Lama Penyinaran Matahari (SS)",
    "FF_X":"Kecepatan Angin Maksimum (FF_X)",
    "FF_AVG":"Kecepatan Angin Rata-rata (FF_AVG)",
}

UNITS = {
    "TN":"°C","TX":"°C","TAVG":"°C","RH_AVG":"%",
    "RR":"mm","SS":"jam","FF_X":"m/s","FF_AVG":"m/s"
}

FEATURES = [
    "RR_lag1","RR_lag2","RR_lag3",
    "TN_lag1","TX_lag1","TAVG_lag1","RH_AVG_lag1",
    "SS_lag1","FF_X_lag1","FF_AVG_lag1",
    "month_sin","month_cos"
]

# =========================================================
# BACA DATASET FINAL
# =========================================================
@st.cache_data(show_spinner=False)
def load_final_data(station):
    path = BASE_DIR / STATIONS[station]

    if not path.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {path.name}"
        )

    raw = pd.read_excel(path)
    raw.columns = [str(c).strip().upper() for c in raw.columns]

    required = ["YEAR","DOY"] + TARGETS
    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(
            "Kolom dataset tidak lengkap. Kolom yang belum ditemukan: "
            + ", ".join(missing)
        )

    raw = raw[required].copy()

    for c in required:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.replace([-999, -99.9, -99], np.nan)

    raw["DATE"] = (
        pd.to_datetime(
            raw["YEAR"].astype("Int64").astype(str) + "-01-01",
            errors="coerce"
        )
        + pd.to_timedelta(raw["DOY"] - 1, unit="D")
    )

    raw = raw.dropna(subset=["DATE"])

    raw = raw[
        (raw["DATE"] >= "1985-01-01") &
        (raw["DATE"] <= "2025-12-31")
    ].copy()

    raw = raw.dropna(subset=TARGETS)

    return raw[["DATE"] + TARGETS].sort_values("DATE").reset_index(drop=True)

# =========================================================
# DATA BULANAN
# =========================================================
@st.cache_data(show_spinner=False)
def make_monthly(daily):
    x = daily.set_index("DATE")

    monthly = pd.DataFrame()
    monthly["TN"] = x["TN"].resample("MS").mean()
    monthly["TX"] = x["TX"].resample("MS").mean()
    monthly["TAVG"] = x["TAVG"].resample("MS").mean()
    monthly["RH_AVG"] = x["RH_AVG"].resample("MS").mean()

    # RR = akumulasi curah hujan dalam bulan
    monthly["RR"] = x["RR"].resample("MS").sum()

    monthly["SS"] = x["SS"].resample("MS").mean()
    monthly["FF_X"] = x["FF_X"].resample("MS").mean()
    monthly["FF_AVG"] = x["FF_AVG"].resample("MS").mean()

    monthly = monthly.dropna().reset_index()
    return monthly

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def make_features(monthly):
    d = monthly.copy()

    for lag in [1,2,3]:
        d[f"RR_lag{lag}"] = d["RR"].shift(lag)

    for c in ["TN","TX","TAVG","RH_AVG","SS","FF_X","FF_AVG"]:
        d[f"{c}_lag1"] = d[c].shift(1)

    d["MONTH"] = d["DATE"].dt.month
    d["month_sin"] = np.sin(2*np.pi*d["MONTH"]/12)
    d["month_cos"] = np.cos(2*np.pi*d["MONTH"]/12)

    return d.dropna().reset_index(drop=True)

# =========================================================
# RANDOM FOREST
# =========================================================
@st.cache_data(show_spinner=False)
def train_model(monthly):
    d = make_features(monthly)

    split = int(len(d) * 0.80)
    train = d.iloc[:split]
    test = d.iloc[split:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_x.fit_transform(train[FEATURES])
    X_test = scaler_x.transform(test[FEATURES])

    y_train = scaler_y.fit_transform(train[TARGETS])

    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = scaler_y.inverse_transform(model.predict(X_test))

    metrics = []
    for i,c in enumerate(TARGETS):
        actual = test[c].values
        p = pred[:,i]
        metrics.append({
            "Parameter": DISPLAY_COLUMNS[c],
            "RMSE": np.sqrt(mean_squared_error(actual,p)),
            "MAE": mean_absolute_error(actual,p),
            "R²": r2_score(actual,p)
        })

    return model, scaler_x, scaler_y, pd.DataFrame(metrics)

# =========================================================
# FORECAST 360 BULAN
# =========================================================
def make_forecast(monthly, model, scaler_x, scaler_y):
    history = monthly[TARGETS].copy().reset_index(drop=True)

    future_dates = pd.date_range(
        start="2026-01-01",
        periods=360,
        freq="MS"
    )

    out = []

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
            "month_sin": np.sin(2*np.pi*date.month/12),
            "month_cos": np.cos(2*np.pi*date.month/12)
        }

        X = pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)
        p = scaler_y.inverse_transform(
            model.predict(scaler_x.transform(X))
        )[0]

        p[TARGETS.index("RR")] = max(0,p[TARGETS.index("RR")])

        out.append([date] + list(p))
        history.loc[len(history)] = p

    return pd.DataFrame(out,columns=["DATE"]+TARGETS)

# =========================================================
# FILE FORECAST PERMANEN
# =========================================================
def forecast_file(station):
    safe = station.lower().replace(" ","_")
    return FORECAST_DIR / f"forecast_{safe}_2026_2055.csv"

@st.cache_data(show_spinner=False)
def load_saved_forecast(station):
    path = forecast_file(station)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 📌 Menu Navigasi")

page = st.sidebar.radio(
    "Pilih Tampilan:",
    ["🏠 Dashboard","📊 Validasi & Evaluasi","👤 Profil Peneliti"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Wilayah Pesisir")

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
    format="YYYY/MM/DD"
)

end_date = st.sidebar.date_input(
    "Selesai",
    value=pd.Timestamp("2025-12-31").date(),
    min_value=pd.Timestamp("1985-01-01").date(),
    max_value=pd.Timestamp("2025-12-31").date(),
    format="YYYY/MM/DD"
)

if start_date > end_date:
    st.sidebar.error("Tanggal mulai harus sebelum tanggal selesai.")
    st.stop()

# =========================================================
# LOAD AKTIF
# =========================================================
try:
    daily = load_final_data(selected_station)
    monthly = make_monthly(daily)
except Exception as e:
    st.error(f"Gagal membaca {selected_station}.")
    st.exception(e)
    st.stop()

monthly_view = monthly[
    (monthly["DATE"].dt.date >= start_date) &
    (monthly["DATE"].dt.date <= end_date)
]

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">MACHINE LEARNING UNTUK MEMPREDIKSI '
    'PERUBAHAN IKLIM WILAYAH PESISIR PANTAI PULAU SUMATERA</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Analisis Temporal Jangka Panjang '
    'Berbasis Random Forest — 1985–2025</div>',
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
        unsafe_allow_html=True
    )

    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="profile-box blue-box">
        <b>Identitas Peneliti</b><br><br>
        • <b>Nama:</b> Huriyatul Firdausi<br>
        • <b>NIM:</b> 06111382328074
        </div>
        """,unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="profile-box yellow-box">
        <b>Dosen Pembimbing</b><br><br>
        • Dr. Melly Ariska, S.Pd., M.Sc.
        </div>
        """,unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="profile-box green-box">
        <b>Informasi Akademik</b><br><br>
        • <b>Program Studi:</b> Pendidikan Fisika<br>
        • <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan<br>
        • <b>Universitas:</b> Universitas Sriwijaya<br>
        • <b>Tahun:</b> 2026
        </div>
        """,unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🛠️ Metadata Konfigurasi Model</div>',
        unsafe_allow_html=True
    )

    meta = pd.DataFrame({
        "Konfigurasi":[
            "Wilayah","Model","Window","Forecast Horizon",
            "Periode Historis","Parameter","Scaling","Evaluation Metrics"
        ],
        "Nilai":[
            selected_station,
            "Random Forest",
            "6 bulan",
            "30 Tahun (360 bulan)",
            "1985–2025",
            "TN, TX, TAVG, RH_AVG, RR, SS, FF_X, FF_AVG + lag 1–3 + sin/cos bulan",
            "MinMaxScaler",
            "RMSE, MAE, R²"
        ]
    })

    st.dataframe(meta,use_container_width=True,hide_index=True)

    st.markdown(
        '<div class="section-title">📊 Parameter Data Iklim</div>',
        unsafe_allow_html=True
    )

    param_table = pd.DataFrame({
        "Kode":["TN","TX","TAVG","RH_AVG","RR","SS","FF_X","FF_AVG"],
        "Parameter":[
            "Temperatur Minimum",
            "Temperatur Maksimum",
            "Temperatur Rata-rata",
            "Kelembapan Relatif Rata-rata",
            "Curah Hujan",
            "Lama Penyinaran Matahari",
            "Kecepatan Angin Maksimum",
            "Kecepatan Angin Rata-rata"
        ]
    })

    st.dataframe(param_table,use_container_width=True,hide_index=True)

    st.markdown(
        '<div class="section-title">📈 Data Historis Bulanan</div>',
        unsafe_allow_html=True
    )

    parameter = st.selectbox(
        "Pilih Parameter Iklim:",
        TARGETS,
        format_func=lambda x: DISPLAY_COLUMNS[x]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_view["DATE"],
        y=monthly_view[parameter],
        mode="lines",
        name=DISPLAY_COLUMNS[parameter]
    ))
    fig.update_layout(
        height=430,
        xaxis_title="Waktu",
        yaxis_title=f"{DISPLAY_COLUMNS[parameter]} ({UNITS[parameter]})",
        hovermode="x unified"
    )
    st.plotly_chart(fig,use_container_width=True)

# =========================================================
# VALIDASI & EVALUASI
# =========================================================
elif page == "📊 Validasi & Evaluasi":

    st.markdown(
        '<div class="section-title">📊 Validasi & Evaluasi Random Forest</div>',
        unsafe_allow_html=True
    )

    saved = load_saved_forecast(selected_station)

    if saved is not None:
        st.success(
            f"Forecast permanen untuk **{selected_station}** ditemukan. "
            "Model tidak perlu dilatih ulang."
        )
        future = saved
        metrics = None

    else:
        st.warning(
            f"Belum ada file forecast permanen untuk **{selected_station}**."
        )

        if st.button(
            f"🚀 Buat Forecast {selected_station} 2026–2055",
            type="primary"
        ):
            with st.spinner("Melatih Random Forest dan membuat 360 bulan forecast..."):
                model, sx, sy, metrics = train_model(monthly)
                future = make_forecast(monthly,model,sx,sy)

                st.session_state[f"future_{selected_station}"] = future
                st.session_state[f"metrics_{selected_station}"] = metrics

            st.success(
                "Forecast berhasil dibuat. "
                "Download CSV ini lalu masukkan ke folder "
                "`hasil_prediksi` di GitHub agar tersimpan permanen."
            )

        future = st.session_state.get(f"future_{selected_station}")
        metrics = st.session_state.get(f"metrics_{selected_station}")

    if future is not None:

        if metrics is not None:
            st.subheader("Hasil Evaluasi Model")
            st.dataframe(
                metrics.style.format({
                    "RMSE":"{:.4f}",
                    "MAE":"{:.4f}",
                    "R²":"{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("Prediksi 2026–2055")

        p = st.selectbox(
            "Pilih parameter prediksi:",
            TARGETS,
            format_func=lambda x: DISPLAY_COLUMNS[x],
            key=f"pred_{selected_station}"
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly["DATE"],
            y=monthly[p],
            mode="lines",
            name="Historis 1985–2025"
        ))

        fig.add_trace(go.Scatter(
            x=future["DATE"],
            y=future[p],
            mode="lines",
            name="Prediksi 2026–2055"
        ))

        fig.update_layout(
            height=470,
            xaxis_title="Waktu",
            yaxis_title=f"{DISPLAY_COLUMNS[p]} ({UNITS[p]})",
            hovermode="x unified"
        )

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Tabel Forecast Tahunan")

        annual = future.copy()
        annual["Tahun"] = annual["DATE"].dt.year

        annual_table = annual.groupby("Tahun")[
            ["TN","TX","TAVG","RH_AVG","SS","FF_X","FF_AVG"]
        ].mean()

        annual_table["RR"] = annual.groupby("Tahun")["RR"].sum()
        annual_table = annual_table.reset_index()

        annual_table = annual_table[
            ["Tahun","TN","TX","TAVG","RH_AVG","RR","SS","FF_X","FF_AVG"]
        ]

        st.dataframe(
            annual_table.style.format({
                "TN":"{:.3f}","TX":"{:.3f}","TAVG":"{:.3f}",
                "RH_AVG":"{:.3f}","RR":"{:.3f}","SS":"{:.3f}",
                "FF_X":"{:.3f}","FF_AVG":"{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        csv = future.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv,
            file_name=(
                "forecast_"
                + selected_station.lower().replace(" ","_")
                + "_2026_2055.csv"
            ),
            mime="text/csv"
        )

# =========================================================
# PROFIL PENELITI
# =========================================================
else:

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
        unsafe_allow_html=True
    )

    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="profile-box blue-box">
        <b>Identitas Peneliti</b><br><br>
        • <b>Nama:</b> Huriyatul Firdausi<br>
        • <b>NIM:</b> 06111382328074
        </div>
        """,unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="profile-box yellow-box">
        <b>Dosen Pembimbing</b><br><br>
        • Dr. Melly Ariska, S.Pd., M.Sc.
        </div>
        """,unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="profile-box green-box">
        <b>Informasi Akademik</b><br><br>
        • <b>Program Studi:</b> Pendidikan Fisika<br>
        • <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan<br>
        • <b>Universitas:</b> Universitas Sriwijaya<br>
        • <b>Tahun:</b> 2026
        </div>
        """,unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📚 Judul Penelitian</div>',
        unsafe_allow_html=True
    )

    st.write(
        "**MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM "
        "WILAYAH PESISIR PANTAI PULAU SUMATERA**"
    )
    st.caption(
        "Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025"
    )

    st.markdown(
        '<div class="section-title">📊 Parameter Data Iklim</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        pd.DataFrame({
            "Kode":["TN","TX","TAVG","RH_AVG","RR","SS","FF_X","FF_AVG"],
            "Parameter":[
                "Temperatur Minimum",
                "Temperatur Maksimum",
                "Temperatur Rata-rata",
                "Kelembapan Relatif Rata-rata",
                "Curah Hujan",
                "Lama Penyinaran Matahari",
                "Kecepatan Angin Maksimum",
                "Kecepatan Angin Rata-rata"
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">🌍 Wilayah Penelitian</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Penelitian mencakup Stasiun Minangkabau, "
        "Stasiun Pesawaran, dan Stasiun Maritim Panjang."
    )

    st.markdown(
        '<div class="section-title">⚙️ Konfigurasi Penelitian</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Model: Random Forest | Window: 6 bulan | "
        "Forecast Horizon: 30 tahun (360 bulan) | "
        "Data historis: 1985–2025 | Prediksi: 2026–2055 | "
        "Scaling: MinMaxScaler | Evaluasi: RMSE, MAE, R²."
    )
