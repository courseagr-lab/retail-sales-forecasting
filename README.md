
# Retail Daily Sales Forecasting

## Latar Belakang Masalah

Bisnis retail membutuhkan estimasi penjualan harian yang akurat untuk perencanaan stok, penjadwalan staf, dan strategi promosi. Tanpa forecast yang baik, keputusan operasional sering dibuat berdasarkan intuisi atau rata-rata historis sederhana, yang rentan meleset terutama saat ada faktor musiman, hari libur, dan aktivitas promosi yang tidak selalu berulang dengan pola tetap. Project ini bertujuan membangun sistem forecasting penjualan harian yang mempertimbangkan tren jangka panjang, musiman mingguan/tahunan, serta efek eksternal seperti promo dan hari libur.

## Dataset

Data penjualan harian mencakup periode 3 tahun (2023-2025), dengan kolom:

| Kolom          | Keterangan                                    |
| -------------- | --------------------------------------------- |
| `date`       | Tanggal harian                                |
| `sales`      | Total penjualan pada hari tersebut (target)   |
| `promo`      | Indikator apakah ada promo pada hari tersebut |
| `is_holiday` | Indikator hari libur besar                    |

Data mentah memerlukan proses cleaning: format angka yang tidak konsisten, entri yang hilang, tanggal duplikat, gap pada urutan kalender, serta outlier ekstrem yang perlu ditangani sebelum analisis dan modeling.

## Solusi / Pendekatan

Project ini dibangun sebagai pipeline modular dengan tahapan berikut:

1. **Data Cleaning** — parsing tanggal, pembersihan format angka, penggabungan tanggal duplikat, pengisian gap kalender, deteksi outlier berbasis rolling median (bukan threshold tetap, karena baseline "normal" bergantung pada konteks musiman), dan interpolasi nilai hilang.
2. **Exploratory Data Analysis** — dekomposisi time series (trend, seasonal, residual), analisis pola mingguan dan bulanan, uji autokorelasi (ACF/PACF), dan uji stasioneritas (ADF test).
3. **Feature Engineering** — fitur kalender (hari, bulan, weekend), lag features (1, 7, 14, 30 hari), dan rolling statistics (mean dan std 7 dan 30 hari), dengan penanganan ketat terhadap data leakage.
4. **Modeling** — empat pendekatan forecasting dibandingkan secara head-to-head menggunakan train-test split berbasis waktu (bukan random split), untuk mensimulasikan kondisi forecasting yang sesungguhnya.
5. **Hyperparameter Tuning** — pencarian parameter optimal untuk model berbasis machine learning menggunakan Optuna (Bayesian optimization).
6. **Experiment Tracking** — seluruh eksperimen (parameter dan metrik tiap model) dicatat menggunakan MLflow untuk memudahkan perbandingan dan audit.
7. **Deployment** — model terbaik disajikan melalui dashboard interaktif untuk visualisasi forecast dan simulasi prediksi.

Seluruh path data, threshold cleaning, dan hyperparameter dieksternalisasi ke file konfigurasi terpusat, menjaga pipeline tetap environment-agnostic dan mudah dikonfigurasi ulang.

## Model

Empat model dibandingkan, mewakili pendekatan yang berbeda-beda:

- **Naive Seasonal** — baseline sederhana yang memprediksi nilai hari ini sama dengan nilai 7 hari sebelumnya, memanfaatkan pola mingguan yang sudah dikonfirmasi lewat EDA.
- **SARIMA** — model statistik time series klasik dengan komponen seasonal periode mingguan, dipilih berdasarkan hasil ACF/PACF dan uji stasioneritas.
- **Prophet** — model dekomposisi additif yang secara eksplisit memasukkan `promo` dan `is_holiday` sebagai regressor eksternal, selain komponen trend dan seasonality bawaan.
- **XGBoost** — model berbasis tree yang memanfaatkan fitur tabular hasil feature engineering (lag, rolling, kalender), dituning menggunakan Optuna dengan validasi `TimeSeriesSplit`.

Evaluasi menggunakan MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), dan MAPE (Mean Absolute Percentage Error), diukur pada holdout 90 hari terakhir.

## Hasil

| Model             | MAE             | RMSE            | MAPE            |
| ----------------- | --------------- | --------------- | --------------- |
| **Prophet** | **14.34** | **20.23** | **3.61%** |
| Naive Seasonal    | 27.31           | 37.90           | 6.60%           |
| SARIMA            | 37.76           | 47.10           | 9.02%           |
| XGBoost (tuned)   | 38.87           | 44.18           | 9.50%           |

Prophet unggul jauh di seluruh metrik, dengan error kurang lebih setengah dari Naive Seasonal dan sepertiga dari SARIMA/XGBoost.

## Interpretasi

**Prophet menang karena informasi eksternal, bukan karena model yang lebih kompleks.** Keunggulannya berasal dari kemampuan memasukkan `promo` dan `is_holiday` sebagai regressor eksplisit — model tahu kapan lonjakan permintaan akan terjadi, alih-alih harus menebaknya semata dari pola historis.

**SARIMA justru kalah dari baseline naive.** Tanpa informasi promo, SARIMA hanya bisa mengandalkan pola musiman rutin, sehingga forecast-nya cenderung diratakan dan gagal menangkap lonjakan tidak berpola. Naive seasonal "menang" bukan karena canggih, tapi karena secara implisit mewarisi efek promo dari data aktual minggu sebelumnya.

**XGBoost, meski sudah dituning dan diberi fitur promo secara eksplisit, tetap gagal menangkap lonjakan tinggi.** Root cause-nya struktural: model berbasis tree memprediksi berdasarkan rata-rata nilai target pada kelompok data serupa yang pernah dilihat, sehingga tidak dapat mengekstrapolasi di luar rentang nilai training. Karena data memiliki tren naik yang kuat, XGBoost secara sistematis under-predict pada periode dengan nilai tinggi. Feature importance mengonfirmasi rolling mean dan lag mingguan sebagai sinyal terkuat, namun ini tidak cukup mengatasi keterbatasan ekstrapolasi tersebut.

**Kesimpulan utama:** untuk data dengan tren kuat dan efek eksternal yang diketahui, model yang secara eksplisit memodelkan tren dan menerima informasi eksternal (seperti Prophet) lebih unggul dibandingkan model yang mengandalkan pola historis murni (SARIMA) maupun model tabular berbasis tree (XGBoost), terlepas dari seberapa matang proses tuning yang dilakukan.

## Arsitektur

retail-sales-forecasting/
├── data/
│   ├── raw/              # data mentah
│   ├── interim/           # hasil cleaning
│   ├── processed/          # hasil feature engineering, siap modeling
│   └── external/
├── notebooks/
│   ├── 01-data-cleaning.ipynb
│   ├── 02-eda-timeseries.ipynb
│   ├── 03-feature-engineering.ipynb
│   ├── 04-modeling-baseline.ipynb
│   ├── 05-modeling-xgboost-tuning.ipynb
│   └── 06-model-comparison.ipynb
├── src/
│   ├── data/                # cleaning pipeline
│   ├── features/            # lag, rolling, calendar features
│   ├── models/               # baseline, Prophet, XGBoost, evaluasi, MLflow logging
│   └── visualization/
├── app/
│   └── streamlit_app.py      # dashboard forecast interaktif
├── models/                  # model artifact final
├── reports/
│   └── figures/               # hasil visualisasi EDA
├── tests/
└── config/
    └── config.yaml            # path, threshold cleaning, hyperparameter, konfigurasi MLflow

## Eksperimen Tracking dan Deployment

Seluruh eksperimen (naive, SARIMA, Prophet, XGBoost) dicatat menggunakan MLflow, mencakup parameter dan metrik tiap run, sehingga dapat dibandingkan secara terpusat tanpa perlu pencatatan manual. Model final (Prophet) disajikan melalui dashboard interaktif yang menampilkan perbandingan actual vs forecast beserta interval kepercayaan, dekomposisi komponen trend/seasonality, serta simulasi prediksi untuk kombinasi tanggal, promo, dan hari libur tertentu.

## Keterbatasan dan Pengembangan Selanjutnya

- Efek promo dimodelkan sebagai variabel biner sederhana; belum mempertimbangkan intensitas atau jenis promo yang berbeda-beda.
- SARIMA dan XGBoost dapat dieksplorasi lebih lanjut dengan menambahkan regressor eksternal secara eksplisit (misalnya melalui SARIMAX atau fitur promo yang di-lag), untuk melihat apakah ini menutup gap performa terhadap Prophet.
- Evaluasi menggunakan satu periode holdout tunggal (90 hari); validasi lebih lanjut menggunakan multiple rolling windows dapat memberikan estimasi performa yang lebih robust.
