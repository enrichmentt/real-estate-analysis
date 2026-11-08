"""Воспроизводимый анализ регионального потребления домашних хозяйств.

Модуль содержит полный расчётный процесс, который используется как при запуске
из командной строки, так и в Jupyter Notebook. Все денежные значения выражены
в номинальных рублях на человека. Корректировка на инфляцию не выполняется,
поскольку в предоставленном наборе данных отсутствует дефлятор.
"""

from __future__ import annotations

import json
import re
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.stattools import acf, adfuller, pacf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "rsdocs_1491005410458_wide_appendix_2025.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "summary.json"
RANDOM_STATE = 42
N_CLUSTERS = 4
TEST_SIZE = 5
FORECAST_HORIZON = 3
MODEL_NAMES = {"naive": "Наивный", "holt": "Holt", "arima": "ARIMA"}
FEATURE_COLUMNS = [
    "mean_value",
    "coefficient_of_variation",
    "cagr",
    "total_growth",
    "max_annual_decline",
    "negative_years_share",
]
FEATURE_LABELS = {
    "mean_value": "Среднее, руб.",
    "coefficient_of_variation": "Коэффициент вариации",
    "cagr": "Среднегодовой темп прироста",
    "total_growth": "Прирост за 2000–2023",
    "max_annual_decline": "Максимальное годовое падение",
    "negative_years_share": "Доля лет снижения",
}
FEDERAL_DISTRICTS = {
    "Центральный федеральный округ",
    "Северо-Западный федеральный округ",
    "Южный федеральный округ",
    "Северо-Кавказский федеральный округ",
    "Приволжский федеральный округ",
    "Уральский федеральный округ",
    "Сибирский федеральный округ",
    "Дальневосточный федеральный округ",
}


def configure_plots() -> None:
    """Задаёт единый стиль графиков и шрифт с поддержкой кириллицы."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 110,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def _year_columns(columns: pd.Index) -> list[str]:
    return [str(column) for column in columns if re.fullmatch(r"\d{4}", str(column))]


def load_and_prepare_data(
    path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает, очищает, фильтрует и преобразует широкий набор данных Росстата.

    Возвращает исходные данные, все строки субъектов, полные ряды субъектов,
    полные наблюдения в длинном формате и реестр исключённых территорий.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    raw = raw.dropna(axis=1, how="all")
    raw.columns = raw.columns.str.strip()
    raw["region_name"] = raw["region_name"].str.strip()
    # В названии региона источник содержит латинскую "p"; исправляем опечатку.
    raw["region_name"] = raw["region_name"].str.replace(
        "Калинингpадская область", "Калининградская область", regex=False
    )
    years = _year_columns(raw.columns)
    if years != [str(year) for year in range(2000, 2024)]:
        raise ValueError(f"Unexpected year columns: {years}")

    numeric = raw[years].apply(
        lambda column: pd.to_numeric(
            column.str.replace("\u00a0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace({"…": pd.NA, "...": pd.NA, "—": pd.NA, "": pd.NA}),
            errors="coerce",
        )
    )
    raw[years] = numeric

    national_mask = raw["region_name"].str.contains("– всего", regex=False)
    district_mask = raw["region_name"].isin(FEDERAL_DISTRICTS)
    duplicate_mask = raw["region_name"].str.contains("без автономн", regex=False)
    aggregate_mask = national_mask | district_mask | duplicate_mask

    excluded_rows: list[dict[str, Any]] = []
    for index, row in raw.loc[aggregate_mask].iterrows():
        if national_mask.loc[index]:
            reason = "Агрегат по Российской Федерации"
        elif district_mask.loc[index]:
            reason = "Федеральный округ"
        else:
            reason = "Дублирующая агрегированная запись"
        excluded_rows.append(
            {"region_name": row["region_name"], "missing_years": int(row[years].isna().sum()), "reason": reason}
        )

    subjects = raw.loc[~aggregate_mask].copy().reset_index(drop=True)
    missing_counts = subjects[years].isna().sum(axis=1)
    for _, row in subjects.loc[missing_counts > 0].iterrows():
        excluded_rows.append(
            {
                "region_name": row["region_name"],
                "missing_years": int(row[years].isna().sum()),
                "reason": "Неполный ряд за 2000–2023; интерполяция не применялась",
            }
        )

    complete = subjects.loc[missing_counts == 0].copy().reset_index(drop=True)
    long = complete.melt(
        id_vars=["region_name", "region_code", "indicator_name", "measure_unit"],
        value_vars=years,
        var_name="year",
        value_name="consumption",
    )
    long["year"] = long["year"].astype(int)
    long["consumption"] = long["consumption"].astype(float)
    long = long.sort_values(["region_name", "year"]).reset_index(drop=True)
    excluded = pd.DataFrame(excluded_rows)
    return raw, subjects, complete, long, excluded


def get_series(complete: pd.DataFrame, region_name: str) -> pd.Series:
    years = _year_columns(complete.columns)
    row = complete.loc[complete["region_name"] == region_name]
    if len(row) != 1:
        raise KeyError(f"Expected one complete series for {region_name!r}")
    values = row[years].iloc[0].astype(float).to_numpy()
    return pd.Series(values, index=pd.Index([int(year) for year in years], name="year"), name=region_name)


def extract_features(complete: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает признаки масштаба и относительной динамики каждого полного ряда."""
    years = _year_columns(complete.columns)
    records: list[dict[str, Any]] = []
    for _, row in complete.iterrows():
        values = row[years].astype(float).to_numpy()
        growth = pd.Series(values).pct_change().dropna().to_numpy()
        records.append(
            {
                "region_name": row["region_name"],
                "mean_value": values.mean(),
                "std_value": values.std(ddof=1),
                "coefficient_of_variation": values.std(ddof=1) / values.mean(),
                "cagr": (values[-1] / values[0]) ** (1 / (len(values) - 1)) - 1,
                "total_growth": values[-1] / values[0] - 1,
                "max_annual_decline": growth.min(),
                "negative_years_share": (growth < 0).mean(),
            }
        )
    return pd.DataFrame(records).set_index("region_name")


def cluster_regions(
    features: pd.DataFrame, n_clusters: int = N_CLUSTERS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, np.ndarray]:
    """Сравнивает K от 2 до 4 и обучает выбранную интерпретируемую модель K-means."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[FEATURE_COLUMNS])
    evaluation_rows = []
    fitted: dict[int, KMeans] = {}
    for k in range(2, 5):
        model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        fitted[k] = model
        evaluation_rows.append(
            {
                "n_clusters": k,
                "silhouette_score": silhouette_score(scaled, labels),
                "inertia": model.inertia_,
                "min_cluster_size": int(np.bincount(labels).min()),
            }
        )
    evaluation = pd.DataFrame(evaluation_rows)

    model = fitted[n_clusters]
    labels = model.labels_
    result = features.copy()
    result["cluster"] = labels
    result["distance_to_centroid"] = np.linalg.norm(
        scaled - model.cluster_centers_[labels], axis=1
    )
    profiles = result.groupby("cluster")[FEATURE_COLUMNS].mean()
    profiles["n_regions"] = result.groupby("cluster").size()
    profiles["regions"] = result.groupby("cluster").apply(
        lambda frame: ", ".join(sorted(frame.index)), include_groups=False
    )
    return result, evaluation, profiles, scaler, scaled


def select_representatives(
    clustered: pd.DataFrame, complete: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    """Выбирает регионы вблизи центроидов трёх наиболее различных кластеров."""
    candidates = (
        clustered.sort_values(["cluster", "distance_to_centroid"])
        .groupby("cluster")
        .head(1)
        .sort_values("cluster")
    )
    if len(candidates) > 3:
        scaled = StandardScaler().fit_transform(clustered[FEATURE_COLUMNS])
        positions = {region: clustered.index.get_loc(region) for region in candidates.index}
        best_combination = max(
            combinations(candidates.index, 3),
            key=lambda combo: sum(
                np.linalg.norm(scaled[positions[first]] - scaled[positions[second]])
                for first, second in combinations(combo, 2)
            ),
        )
        selected = candidates.loc[list(best_combination)].sort_values("cluster")
    else:
        selected = candidates
    selected_names = selected.index.tolist()
    stats_rows = []
    for region in selected_names:
        series = get_series(complete, region)
        slope, _ = np.polyfit(series.index.to_numpy(), series.to_numpy(), 1)
        stats_rows.append(
            {
                "region_name": region,
                "cluster": int(clustered.loc[region, "cluster"]),
                "distance_to_centroid": clustered.loc[region, "distance_to_centroid"],
                "mean_value": series.mean(),
                "min_value": series.min(),
                "min_year": int(series.idxmin()),
                "max_value": series.max(),
                "max_year": int(series.idxmax()),
                "linear_slope_rub_per_year": slope,
                "relative_slope_percent": 100 * slope / series.mean(),
            }
        )
    return selected_names, pd.DataFrame(stats_rows)


def _adf_record(region: str, transformation: str, values: np.ndarray) -> dict[str, Any]:
    statistic, p_value, used_lag, n_obs, critical_values, _ = adfuller(
        values, regression="c", autolag="AIC", result_object=False
    )
    return {
        "region_name": region,
        "transformation": transformation,
        "adf_statistic": statistic,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_1_percent": critical_values["1%"],
        "critical_5_percent": critical_values["5%"],
        "stationary_at_5_percent": p_value < 0.05,
    }


def _significant_lags(values: np.ndarray, kind: str, n_lags: int = 7) -> list[int]:
    function = acf if kind == "acf" else pacf
    kwargs = {"method": "ywm"} if kind == "pacf" else {}
    coefficients, confidence = function(
        values, nlags=n_lags, alpha=0.05, result_object=False, **kwargs
    )
    lower, upper = confidence[:, 0] - coefficients, confidence[:, 1] - coefficients
    return [lag for lag in range(1, n_lags + 1) if lower[lag] > 0 or upper[lag] < 0]


def analyze_selected_series(
    selected_names: list[str], complete: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Оценивает HP-тренды, проводит ADF-тесты и формирует гипотезы ARIMA по ACF/PACF."""
    adf_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    for number, region in enumerate(selected_names, start=1):
        series = get_series(complete, region)
        cycle, trend = hpfilter(series.to_numpy(), lamb=6.25)
        cycle_series = pd.Series(cycle, index=series.index)
        cycle_rows.append(
            {
                "region_name": region,
                "cycle_std_rub": cycle_series.std(),
                "largest_positive_cycle_year": int(cycle_series.idxmax()),
                "largest_positive_cycle_rub": cycle_series.max(),
                "largest_negative_cycle_year": int(cycle_series.idxmin()),
                "largest_negative_cycle_rub": cycle_series.min(),
            }
        )
        adf_rows.append(_adf_record(region, "level", series.to_numpy()))
        adf_rows.append(_adf_record(region, "first_difference", series.diff().dropna().to_numpy()))

        differences = series.diff().dropna()
        acf_lags = _significant_lags(differences.to_numpy(), "acf")
        pacf_lags = _significant_lags(differences.to_numpy(), "pacf")
        hypothesis_rows.append(
            {
                "region_name": region,
                "significant_acf_lags": ", ".join(map(str, acf_lags)) or "нет",
                "significant_pacf_lags": ", ".join(map(str, pacf_lags)) or "нет",
                "initial_p_hypothesis": min(max(pacf_lags, default=0), 2),
                "initial_q_hypothesis": min(max(acf_lags, default=0), 2),
                "note": "Гипотеза по первым разностям; окончательный порядок выбирается по AIC на train",
            }
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(series.index, series.values, marker="o", label="Исходный ряд")
        axes[0].plot(series.index, trend, linewidth=2.5, label="HP-тренд (λ = 6,25)")
        axes[0].set_title(f"{region}: динамика и тренд")
        axes[0].set_ylabel("Рубли на человека")
        axes[0].legend()
        axes[1].bar(cycle_series.index, cycle_series.values, color="#607d8b")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Циклическая составляющая HP-фильтра")
        axes[1].set_xlabel("Год")
        axes[1].set_ylabel("Отклонение, руб.")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"03_region_{number}_trend.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        plot_acf(differences, lags=7, zero=False, alpha=0.05, ax=axes[0])
        plot_pacf(differences, lags=7, zero=False, alpha=0.05, method="ywm", ax=axes[1])
        axes[0].set_title("ACF первых разностей")
        axes[1].set_title("PACF первых разностей")
        for ax in axes:
            ax.set_xlabel("Лаг, лет")
            ax.set_ylabel("Корреляция")
        fig.suptitle(f"{region}: автокорреляционный анализ")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"04_region_{number}_acf_pacf.png")
        plt.close(fig)

    return pd.DataFrame(adf_rows), pd.DataFrame(cycle_rows), pd.DataFrame(hypothesis_rows)


def select_arima_order(train: pd.Series) -> tuple[tuple[int, int, int], pd.DataFrame]:
    """Выбирает экономную спецификацию ARIMA только по AIC обучающей выборки."""
    rows: list[dict[str, Any]] = []
    for d in range(2):
        for p in range(3):
            for q in range(3):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted = ARIMA(
                            train.to_numpy(dtype=float),
                            order=(p, d, q),
                            trend="c" if d == 0 else None,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit()
                    rows.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "aic": fitted.aic,
                            "converged": bool(fitted.mle_retvals.get("converged", True)),
                        }
                    )
                except (ValueError, np.linalg.LinAlgError):
                    rows.append({"p": p, "d": d, "q": q, "aic": np.inf, "converged": False})
    grid = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    valid = grid[np.isfinite(grid["aic"]) & grid["converged"]]
    if valid.empty:
        valid = grid[np.isfinite(grid["aic"])]
    if valid.empty:
        raise RuntimeError("No ARIMA specification could be estimated")
    best = valid.iloc[0]
    return (int(best["p"]), int(best["d"]), int(best["q"])), grid


def fit_predict(
    model_name: str,
    train: pd.Series,
    horizon: int,
    arima_order: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Обучает одну модель и возвращает числовой прогноз."""
    if model_name == "naive":
        sktime_train = pd.Series(
            train.to_numpy(dtype=float),
            index=pd.period_range(start=str(train.index[0]), periods=len(train), freq="Y"),
        )
        forecaster = NaiveForecaster(strategy="last")
        forecaster.fit(sktime_train)
        fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
        return np.asarray(forecaster.predict(fh), dtype=float)
    if model_name == "holt":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = Holt(train.to_numpy(dtype=float), initialization_method="estimated").fit(
                optimized=True
            )
        return np.asarray(fitted.forecast(horizon), dtype=float)
    if model_name == "arima":
        if arima_order is None:
            arima_order, _ = select_arima_order(train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(
                train.to_numpy(dtype=float),
                order=arima_order,
                trend="c" if arima_order[1] == 0 else None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        return np.asarray(fitted.forecast(horizon), dtype=float)
    raise ValueError(f"Unknown model: {model_name}")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }


def event_hypothesis(year: int) -> str:
    hypotheses = {
        2019: "До пандемийного шока; по одному ряду причину отклонения идентифицировать нельзя",
        2020: "Пандемия COVID-19 и ограничения могли нарушить прежнюю траекторию потребления",
        2021: "Восстановление после шока 2020 года и изменение номинальных цен могли усилить рост",
        2022: "Санкционный шок, перестройка рынков и инфляция могли изменить динамику",
        2023: "Адаптация экономики и номинальный ценовой рост могли сохранить отклонение от тренда",
    }
    return hypotheses.get(year, "Возможный структурный или ценовой шок; требуется проверка на дополнительных данных")


def forecast_selected_regions(
    selected_names: list[str], complete: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Оценивает три модели, выполняет одношаговую rolling-проверку и прогноз на 2024–2026 годы."""
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    rolling_metric_rows: list[dict[str, Any]] = []
    future_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []

    for number, region in enumerate(selected_names, start=1):
        series = get_series(complete, region)
        train = series.iloc[:-TEST_SIZE]
        test = series.iloc[-TEST_SIZE:]
        arima_order, grid = select_arima_order(train)
        grid.insert(0, "region_name", region)
        order_frames.append(grid)

        test_predictions: dict[str, np.ndarray] = {}
        for model_name in MODEL_NAMES:
            prediction = fit_predict(model_name, train, len(test), arima_order)
            test_predictions[model_name] = prediction
            metrics = calculate_metrics(test.to_numpy(), prediction)
            metric_rows.append(
                {
                    "region_name": region,
                    "model": model_name,
                    "model_label": MODEL_NAMES[model_name],
                    **metrics,
                    "train_start": int(train.index[0]),
                    "train_end": int(train.index[-1]),
                    "test_start": int(test.index[0]),
                    "test_end": int(test.index[-1]),
                    "arima_order": str(arima_order) if model_name == "arima" else "—",
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "region_name": region,
                        "model": model_name,
                        "year": test.index,
                        "actual": test.to_numpy(),
                        "predicted": prediction,
                    }
                )
            )

        region_metrics = pd.DataFrame([row for row in metric_rows if row["region_name"] == region])
        best_model = region_metrics.sort_values(["rmse", "mape_percent"]).iloc[0]["model"]

        rolling_predictions = []
        for offset, actual_year in enumerate(test.index):
            # Окно фиксированной длины сдвигается на год: старые наблюдения исключаются из train.
            rolling_train = series.iloc[offset : offset + len(train)]
            predicted_value = fit_predict(best_model, rolling_train, 1, arima_order)[0]
            actual_value = float(series.loc[actual_year])
            rolling_predictions.append(predicted_value)
            rolling_frames.append(
                {
                    "region_name": region,
                    "model": best_model,
                    "year": int(actual_year),
                    "train_start": int(rolling_train.index[0]),
                    "train_end": int(rolling_train.index[-1]),
                    "actual": actual_value,
                    "predicted": predicted_value,
                    "error": actual_value - predicted_value,
                    "absolute_error": abs(actual_value - predicted_value),
                    "absolute_percentage_error": abs(actual_value - predicted_value) / actual_value * 100,
                    "event_hypothesis": event_hypothesis(int(actual_year)),
                }
            )
        rolling_metrics = calculate_metrics(test.to_numpy(), np.asarray(rolling_predictions))
        fixed_best_metrics = region_metrics.loc[region_metrics["model"] == best_model].iloc[0]
        rolling_metric_rows.append(
            {
                "region_name": region,
                "model": best_model,
                "model_label": MODEL_NAMES[best_model],
                "fixed_rmse": fixed_best_metrics["rmse"],
                "fixed_mape_percent": fixed_best_metrics["mape_percent"],
                "rolling_rmse": rolling_metrics["rmse"],
                "rolling_mape_percent": rolling_metrics["mape_percent"],
            }
        )

        final_order = select_arima_order(series)[0] if best_model == "arima" else arima_order
        future_prediction = fit_predict(best_model, series, FORECAST_HORIZON, final_order)
        future_years = np.arange(series.index[-1] + 1, series.index[-1] + 1 + FORECAST_HORIZON)
        last_value = float(series.iloc[-1])
        for year, value in zip(future_years, future_prediction):
            future_frames.append(
                {
                    "region_name": region,
                    "model": best_model,
                    "model_label": MODEL_NAMES[best_model],
                    "year": int(year),
                    "forecast": value,
                    "growth_from_2023_percent": (value / last_value - 1) * 100,
                    "final_arima_order": str(final_order) if best_model == "arima" else "—",
                }
            )

        region_errors = pd.DataFrame(
            [row for row in rolling_frames if row["region_name"] == region]
        )
        error_frames.append(
            region_errors.sort_values("absolute_error", ascending=False)
            .head(5)
            .assign(error_rank=lambda frame: range(1, len(frame) + 1))
        )

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(train.index, train.values, color="#263238", label="Обучающая выборка")
        ax.plot(test.index, test.values, color="#263238", marker="o", label="Факт (test)")
        colors = {"naive": "#78909c", "holt": "#1565c0", "arima": "#6a1b9a"}
        for model_name, prediction in test_predictions.items():
            ax.plot(
                test.index,
                prediction,
                linestyle="--",
                marker=".",
                color=colors[model_name],
                label=f"{MODEL_NAMES[model_name]} (test)",
            )
        ax.plot(
            np.concatenate(([series.index[-1]], future_years)),
            np.concatenate(([last_value], future_prediction)),
            color="#c62828",
            linewidth=2.5,
            marker="o",
            label=f"Прогноз: {MODEL_NAMES[best_model]}",
        )
        ax.axvline(test.index[0] - 0.5, color="gray", linestyle=":", label="Граница train/test")
        ax.set_title(f"{region}: проверка моделей и прогноз до {future_years[-1]}")
        ax.set_xlabel("Год")
        ax.set_ylabel("Рубли на человека")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"05_region_{number}_forecast.png")
        plt.close(fig)

    return {
        "metrics": pd.DataFrame(metric_rows),
        "test_predictions": pd.concat(prediction_frames, ignore_index=True),
        "rolling_predictions": pd.DataFrame(rolling_frames),
        "rolling_metrics": pd.DataFrame(rolling_metric_rows),
        "future_forecasts": pd.DataFrame(future_frames),
        "top_errors": pd.concat(error_frames, ignore_index=True),
        "arima_aic_grid": pd.concat(order_frames, ignore_index=True),
    }


def create_cluster_plots(
    clustered: pd.DataFrame, evaluation: pd.DataFrame, scaled: np.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(evaluation["n_clusters"], evaluation["silhouette_score"], marker="o", linewidth=2)
    best_row = evaluation.loc[evaluation["n_clusters"] == N_CLUSTERS].iloc[0]
    ax.scatter([N_CLUSTERS], [best_row["silhouette_score"]], s=120, color="#c62828", zorder=3)
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("Число кластеров")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Сравнение решений K-means")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_silhouette.png")
    plt.close(fig)

    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=clustered["cluster"], cmap="viridis", s=65
    )
    for region in clustered.sort_values(["cluster", "distance_to_centroid"]).groupby("cluster").head(1).index:
        position = clustered.index.get_loc(region)
        ax.annotate(region, coordinates[position], xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Первая главная компонента")
    ax.set_ylabel("Вторая главная компонента")
    ax.set_title("Регионы в пространстве стандартизированных признаков (PCA)")
    legend = ax.legend(*scatter.legend_elements(), title="Кластер", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_clusters_pca.png")
    plt.close(fig)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def save_outputs(
    raw: pd.DataFrame,
    subjects: pd.DataFrame,
    complete: pd.DataFrame,
    long: pd.DataFrame,
    excluded: pd.DataFrame,
    features: pd.DataFrame,
    clustered: pd.DataFrame,
    evaluation: pd.DataFrame,
    profiles: pd.DataFrame,
    selected_stats: pd.DataFrame,
    adf_results: pd.DataFrame,
    cycles: pd.DataFrame,
    hypotheses: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(TABLES_DIR / "clean_long_data.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(TABLES_DIR / "excluded_territories.csv", index=False, encoding="utf-8-sig")
    features.to_csv(TABLES_DIR / "region_features.csv", encoding="utf-8-sig")
    clustered.to_csv(TABLES_DIR / "clustered_regions.csv", encoding="utf-8-sig")
    evaluation.to_csv(TABLES_DIR / "cluster_evaluation.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(TABLES_DIR / "cluster_profiles.csv", encoding="utf-8-sig")
    selected_stats.to_csv(TABLES_DIR / "selected_region_statistics.csv", index=False, encoding="utf-8-sig")
    adf_results.to_csv(TABLES_DIR / "adf_results.csv", index=False, encoding="utf-8-sig")
    cycles.to_csv(TABLES_DIR / "cycle_statistics.csv", index=False, encoding="utf-8-sig")
    hypotheses.to_csv(TABLES_DIR / "arima_hypotheses.csv", index=False, encoding="utf-8-sig")
    for name, frame in forecasts.items():
        frame.to_csv(TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    best_rows = []
    for region in selected_stats["region_name"]:
        region_metrics = forecasts["metrics"].loc[forecasts["metrics"]["region_name"] == region]
        best = region_metrics.sort_values(["rmse", "mape_percent"]).iloc[0]
        best_rows.append(
            {
                "region_name": region,
                "model": best["model"],
                "model_label": best["model_label"],
                "rmse": best["rmse"],
                "mape_percent": best["mape_percent"],
            }
        )
    summary = {
        "data": {
            "raw_rows": len(raw),
            "raw_columns": len(raw.columns),
            "subject_rows_before_missing_filter": len(subjects),
            "complete_regions": len(complete),
            "excluded_rows": len(excluded),
            "years": [2000, 2023],
            "long_rows": len(long),
            "raw_dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
        },
        "clustering": {
            "algorithm": "KMeans",
            "selected_n_clusters": N_CLUSTERS,
            "features": FEATURE_COLUMNS,
            "evaluation": _json_records(evaluation),
        },
        "selected_regions": _json_records(selected_stats),
        "adf": _json_records(adf_results),
        "best_models": best_rows,
        "forecast_metrics": _json_records(forecasts["metrics"]),
        "rolling_metrics": _json_records(forecasts["rolling_metrics"]),
        "future_forecasts": _json_records(forecasts["future_forecasts"]),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline() -> dict[str, Any]:
    """Выполняет весь проект и сохраняет каждую таблицу и каждый график."""
    configure_plots()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw, subjects, complete, long, excluded = load_and_prepare_data()
    features = extract_features(complete)
    clustered, evaluation, profiles, _, scaled = cluster_regions(features)
    selected_names, selected_stats = select_representatives(clustered, complete)
    adf_results, cycles, hypotheses = analyze_selected_series(selected_names, complete)
    forecasts = forecast_selected_regions(selected_names, complete)
    create_cluster_plots(clustered, evaluation, scaled)
    save_outputs(
        raw,
        subjects,
        complete,
        long,
        excluded,
        features,
        clustered,
        evaluation,
        profiles,
        selected_stats,
        adf_results,
        cycles,
        hypotheses,
        forecasts,
    )
    return {
        "raw": raw,
        "subjects": subjects,
        "complete": complete,
        "long": long,
        "excluded": excluded,
        "features": features,
        "clustered": clustered,
        "evaluation": evaluation,
        "profiles": profiles,
        "selected_names": selected_names,
        "selected_stats": selected_stats,
        "adf_results": adf_results,
        "cycles": cycles,
        "hypotheses": hypotheses,
        **forecasts,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Регионов с полными рядами: {len(results['complete'])}")
    print("Выбранные регионы:", ", ".join(results["selected_names"]))
    print("\nОценка кластеризации:\n", results["evaluation"].to_string(index=False))
    print("\nМетрики прогнозов:\n", results["metrics"].to_string(index=False))
    print("\nРезультаты сохранены в каталоге outputs")
