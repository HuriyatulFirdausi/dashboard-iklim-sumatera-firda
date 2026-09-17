import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Dashboard Iklim Pesisir Pulau Sumatera",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

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

st.markdown("""
<style>
.main-title {font-size: 30px; font-weight: 700; margin-bottom: 0;}
.sub-title {font-size: 17px; color: #555; margin-top: 2px;}
</style>
""", unsafe_allow_html=True)


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper().replace(" ", "_").replace("-", "_")


def clean_columns(df):
    aliases = {
        "TAHUN": "YEAR", "THN": "YEAR",
        "HARI_KE": "DOY", "HARIKE": "DOY", "DAY_OF_YEAR": "DOY",
        "TANGGAL": "DATE", "DATE": "DATE",
        "TN": "TN", "TX": "TX", "TAVG": "TAVG",
        "RH_AVG": "RH_AVG", "RHAVG": "RH_AVG",
        "RR": "RR", "SS": "SS",
        "FF_X": "FF_X", "FFX": "FF_X",
        "FF_AVG": "FF_AVG", "FFAVG": "FF_AVG",
    }
    out = {}
    for c in df.columns:
        n = norm(c)
        out[c] = aliases.get(n, n)
    return df.rename(columns=out).loc[:, ~df.rename(columns=out).columns.duplicated()]


def read_final_excel(path):
    """Baca dataset FIX dengan header pada baris Excel ke-17 (index 16).
    Tetap punya fallback otomatis jika struktur sheet berubah.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    # Struktur file FIX: baris header data berada pada baris Excel ke-17.
    try:
        df = pd.read_excel(path, header=16, engine="openpyxl")
        df = clean_columns(df)
        score = len(set(TARGETS).intersection(df.columns))
        if score < 6:
            raise ValueError("Header utama tidak cocok")
    except Exception:
        raw = pd.read_excel(path, header=None, engine="openpyxl")
        best = None
        best_score = -1
        for i in range(min(40, len(raw))):
            vals = {norm(v) for v in raw.iloc[i].tolist()}
            score = len(vals.intersection(set(TARGETS + ["YEAR", "DOY", "DATE", "TAHUN"])))
            if score > best_score:
                best_score, best = score, i
        if best is None or best_score < 6:
            raise ValueError("Header dataset FIX tidak ditemukan.")
        df = raw.iloc[best + 1:].copy()
        df.columns = [norm(v) for v in raw.iloc[best].tolist()]
        df = clean_columns(df)

    # YEAR/DOY atau DATE
    for c in TARGETS + ["YEAR", "DOY"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "DATE" in df.columns:
        date = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=True)
    elif "YEAR" in df.columns and "DOY" in df.columns:
        year = pd.to_numeric(df["YEAR"], errors="coerce")
        doy = pd.to_numeric(df["DOY"], errors="coerce")
        date = pd.to_datetime(year.astype("Int64").astype(str) + "-01-01", errors="coerce") + pd.to_timedelta(doy - 1, unit="D")
    else:
        # Beberapa file BMKG mempunyai kolom tahun/hari dengan variasi nama.
        cols = {norm(c): c for c in df.columns}
        if "YEAR" not in cols and "TAHUN" in cols:
            df["YEAR"] = pd.to_numeric(df[cols["TAHUN"]], errors="coerce")
        if "DOY" not in cols:
            for candidate in ["HARI_KE", "HARIKE", "DAY_OF_YEAR"]:
                if candidate in cols:
                    df["DOY"] = pd.to_numeric(df[cols[candidate]], errors="coerce")
                    break
        if "YEAR" in df.columns and "DOY" in df.columns:
            date = pd.to_datetime(df["YEAR"].astype("Int64").astype(str) + "-01-01", errors="coerce") + pd.to_timedelta(df["DOY"] - 1, unit="D")
        else:
            raise ValueError("Kolom tanggal YEAR/DOY atau DATE tidak ditemukan.")

    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise ValueError("Kolom parameter tidak lengkap: " + ", ".join(missing))

    out = df[TARGETS].copy()
    out.insert(0, "DATE", date)
    out = out.dropna(subset=["DATE"])
    out = out[(out.DATE >= HIST_START) & (out.DATE <= HIST_END)]
    out = out.sort_values("DATE").drop_duplicates("DATE")
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=3)
def load_station(filename):
    return read_final_excel(str(BASE_DIR / filename))


@st.cache_data(show_spinner=False, max_entries=3)
def make_monthly(filename, start_date, end_date):
    daily = load_station(filename)
    daily = daily[(daily.DATE >= pd.Timestamp(start_date)) & (daily.DATE <= pd.Timestamp(end_date))]
    x = daily.set_index("DATE")[TARGETS]
    idx = pd.date_range(x.index.min().to_period("M").start_time, x.index.max().to_period("M").start_time, freq="MS")
    m = pd.DataFrame(index=idx)
    m["RR"] = x["RR"].resample("MS").sum(min_count=1)
    for c in TARGETS:
        if c != "RR":
            m[c] = x[c].resample("MS").mean()
    return m.reset_index(names="DATE")


def supervised(monthly):
    d = monthly.copy()
    for lag in range(1, WINDOW + 1):
        for c in TARGETS:
            d[f"{c}_lag{lag}"] = d[c].shift(lag)
    month = d.DATE.dt.month
    d["month_sin"] = np.sin(2 * np.pi * month / 12)
    d["month_cos"] = np.cos(2 * np.pi * month / 12)
    features = [f"{c}_lag{lag}" for lag in range(1, WINDOW + 1) for c in TARGETS] + ["month_sin", "month_cos"]
    d = d.dropna(subset=features + TARGETS).reset_index(drop=True)
    return d, features


@st.cache_resource(show_spinner=False, max_entries=3)
def train_model(station_name, start_date, end_date):
    filename = STATIONS[station_name]
    monthly = make_monthly(filename, start_date, end_date)
    data, features = supervised(monthly)

    if len(data) < 100:
        raise ValueError(f"Data bulanan setelah preprocessing hanya {len(data)} baris.")

    cut = int(len(data) * 0.80)
    train = data.iloc[:cut]
    test = data.iloc[cut:]

    sx = MinMaxScaler()
    sy = MinMaxScaler()
    Xtr = sx.fit_transform(train[features])
    Ytr = sy.fit_transform(train[TARGETS])
    Xte = sx.transform(test[features])
    Yte = test[TARGETS].to_numpy()

    # GridSearch sangat kecil agar cocok untuk Streamlit Cloud.
    base = RandomForestRegressor(random_state=42, n_jobs=1)
    grid = GridSearchCV(
        base,
        {"n_estimators": [40], "max_depth": [12]},
        cv=TimeSeriesSplit(n_splits=2),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(Xtr, Ytr)
    model = grid.best_estimator_

    pred = sy.inverse_transform(model.predict(Xte))
    metrics = []
    for i, c in enumerate(TARGETS):
        actual = Yte[:, i]
        p = pred[:, i]
        metrics.append({
            "Parameter": c,
            "RMSE": np.sqrt(mean_squared_error(actual, p)),
            "MAE": mean_absolute_error(actual, p),
            "R²": r2_score(actual, p) if len(np.unique(actual)) > 1 else np.nan,
        })

    # Forecast 360 bulan tanpa concat berulang.
    history = monthly[TARGETS].tail(WINDOW).copy().reset_index(drop=True)
    future = []
    dates = pd.date_range(FORECAST_START, periods=FORECAST_MONTHS, freq="MS")
    for date in dates:
        row = {}
        for lag in range(1, WINDOW + 1):
            src = history.iloc[-lag]
            for c in TARGETS:
                row[f"{c}_lag{lag}"] = float(src[c])
        row["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * date.month / 12)
        xf = pd.DataFrame([row])[features]
        yp = sy.inverse_transform(model.predict(sx.transform(xf)))[0]
        vals = {c: float(yp[i]) for i, c in enumerate(TARGETS)}
        future.append({"DATE": date, **vals})
        history = pd.concat([history, pd.DataFrame([vals])], ignore_index=True).tail(WINDOW)

    forecast = pd.DataFrame(future)
    return {
        "monthly": monthly,
        "train_n": len(train),
        "test_n": len(test),
        "daily_n": len(load_station(filename)),
        "metrics": pd.DataFrame(metrics),
        "forecast": forecast,
        "best_params": grid.best_params_,
    }


def header():
    st.markdown('<div class="main-title">MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM WILAYAH PESISIR PANTAI PULAU SUMATERA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analisis Temporal Jangka Panjang Berbasis Random Forest - 1985 - 2025</div>', unsafe_allow_html=True)
    st.divider()


def annual_table(forecast):
    x = forecast.copy()
    x["Tahun"] = x.DATE.dt.year
    out = []
    for year, g in x.groupby("Tahun"):
        row = {"Tahun": int(year)}
        for c in TARGETS:
            row[c] = g[c].sum() if c == "RR" else g[c].mean()
        out.append(row)
    return pd.DataFrame(out)


def dashboard(station, start_date, end_date):
    header()
    try:
        with st.spinner(f"Memproses {station}... (hanya pertama kali)"):
            r = train_model(station, start_date, end_date)
    except Exception as e:
        st.error(f"Gagal memproses {station}.")
        st.exception(e)
        return

    st.success(f"{station} berhasil diproses. Hasil akan memakai cache saat halaman direrun.")
    a, b, c, d = st.columns(4)
    a.metric("Data Harian", f"{r['daily_n']:,}")
    b.metric("Data Bulanan", f"{len(r['monthly']):,}")
    c.metric("Training", f"{r['train_n']:,}")
    d.metric("Forecast", "2026–2055")

    param = st.selectbox("Pilih parameter", TARGETS, format_func=lambda z: f"{z} — {PARAMETER_INFO[z][0]}")
    chart = r["monthly"].set_index("DATE")[[param]]
    chart.columns = [f"{param} ({PARAMETER_INFO[param][1]})"]
    st.line_chart(chart)

    st.subheader("Hasil Prediksi Tahunan 2026–2055")
    annual = annual_table(r["forecast"])
    st.dataframe(annual.round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Hasil Prediksi CSV",
        annual.to_csv(index=False).encode("utf-8"),
        f"prediksi_{station.lower().replace(' ', '_')}_2026_2055.csv",
        "text/csv",
    )
    st.caption(f"GridSearchCV: {r['best_params']}")


def validation(station, start_date, end_date):
    header()
    st.subheader("Validasi & Evaluasi Model")
    try:
        with st.spinner("Memuat evaluasi..."):
            r = train_model(station, start_date, end_date)
    except Exception as e:
        st.error("Evaluasi belum dapat ditampilkan.")
        st.exception(e)
        return
    st.dataframe(r["metrics"].round(4), use_container_width=True, hide_index=True)
    st.info("Data dibagi secara kronologis: 80% training dan 20% testing. GridSearchCV menggunakan TimeSeriesSplit 2-fold untuk menjaga urutan waktu.")


def profile():
    header()
    st.subheader("Profil Peneliti")
    x, y = st.columns(2)
    with x:
        st.markdown("**Nama:** Huriyatul Firdausi  \n**NIM:** 06111382328074  \n**Program Studi:** Pendidikan Fisika  \n**Fakultas:** Keguruan dan Ilmu Pendidikan  \n**Universitas:** Universitas Sriwijaya  \n**Tahun:** 2026")
    with y:
        st.markdown("**Dosen Pembimbing:** Dr. Melly Ariska, S.Pd., M.Sc.  \n**Metode:** Random Forest Regressor  \n**Periode historis:** 1985–2025  \n**Periode prediksi:** 2026–2055  \n**Window:** 6 bulan  \n**Evaluasi:** RMSE, MAE, R²")
    st.divider()
    st.subheader("Parameter Dataset")
    st.dataframe(pd.DataFrame([{"Kode": c, "Parameter": PARAMETER_INFO[c][0], "Satuan": PARAMETER_INFO[c][1]} for c in TARGETS]), use_container_width=True, hide_index=True)


st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih halaman", ["Dashboard", "Validasi & Evaluasi", "Profil Peneliti"])
st.sidebar.divider()
st.sidebar.subheader("Konfigurasi Data")
station = st.sidebar.selectbox("Pilih stasiun", list(STATIONS))
start_date = st.sidebar.date_input("Mulai", HIST_START.date(), min_value=HIST_START.date(), max_value=HIST_END.date())
end_date = st.sidebar.date_input("Selesai", HIST_END.date(), min_value=HIST_START.date(), max_value=HIST_END.date())
if start_date > end_date:
    st.sidebar.error("Tanggal mulai harus lebih kecil atau sama dengan tanggal selesai.")
    st.stop()
st.sidebar.caption("Periode historis: 1985–2025")
st.sidebar.caption("Forecast: 2026–2055")
st.sidebar.caption("Random Forest + GridSearchCV ringan")

if menu == "Dashboard":
    dashboard(station, start_date, end_date)
elif menu == "Validasi & Evaluasi":
    validation(station, start_date, end_date)
else:
    profile()
