"""Формирует полный PDF-отчёт на русском языке из сохранённых результатов."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
REPORT_PATH = ROOT / "reports" / "regional_consumption_report.pdf"


def register_fonts() -> None:
    regular = os.environ.get("DAIMON_CJK_FONT_REGULAR")
    bold = os.environ.get("DAIMON_CJK_FONT_BOLD")
    if not regular or not bold:
        # Используем системные шрифты, если специальные настройки не заданы.
        candidates = [
            (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
            (
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ),
        ]
        for regular_path, bold_path in candidates:
            if regular_path.exists() and bold_path.exists():
                regular, bold = str(regular_path), str(bold_path)
                break
    if not regular or not bold:
        raise RuntimeError("A TrueType font with Cyrillic support is required")
    pdfmetrics.registerFont(TTFont("DaimonCJK", regular))
    pdfmetrics.registerFont(TTFont("DaimonCJK-Bold", bold))
    pdfmetrics.registerFontFamily(
        "DaimonCJK",
        normal="DaimonCJK",
        bold="DaimonCJK-Bold",
        italic="DaimonCJK",
        boldItalic="DaimonCJK-Bold",
    )


def rub(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def pct(value: float, digits: int = 2) -> str:
    return f"{num(value, digits)}%"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv", encoding="utf-8-sig")


register_fonts()
styles = getSampleStyleSheet()
body = ParagraphStyle(
    "RussianBody",
    parent=styles["BodyText"],
    fontName="DaimonCJK",
    fontSize=10.2,
    leading=15.2,
    alignment=TA_JUSTIFY,
    spaceAfter=7,
    textColor=colors.HexColor("#263238"),
)
heading1 = ParagraphStyle(
    "RussianHeading1",
    parent=styles["Heading1"],
    fontName="DaimonCJK-Bold",
    fontSize=16,
    leading=21,
    spaceBefore=8,
    spaceAfter=12,
    keepWithNext=True,
    textColor=colors.HexColor("#17365D"),
)
heading2 = ParagraphStyle(
    "RussianHeading2",
    parent=styles["Heading2"],
    fontName="DaimonCJK-Bold",
    fontSize=12.5,
    leading=17,
    spaceBefore=10,
    spaceAfter=8,
    keepWithNext=True,
    textColor=colors.HexColor("#244A70"),
)
caption = ParagraphStyle(
    "Caption",
    parent=body,
    fontSize=8.8,
    leading=12,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"),
    spaceAfter=10,
)
small = ParagraphStyle("Small", parent=body, fontSize=7.8, leading=10, spaceAfter=0)
small_bold = ParagraphStyle("SmallBold", parent=small, fontName="DaimonCJK-Bold")
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, firstLineIndent=-9, spaceAfter=5)

story: list = []
figure_number = 0
table_number = 0


def add_text(text: str, style: ParagraphStyle = body) -> None:
    story.append(Paragraph(text, style))


def add_heading(text: str, level: int = 1) -> None:
    story.append(Paragraph(text, heading1 if level == 1 else heading2))


def add_bullet(text: str) -> None:
    story.append(Paragraph("• " + text, bullet))


def add_table(title: str, headers: list[str], rows: Iterable[Iterable], widths: list[float] | None = None) -> None:
    global table_number
    table_number += 1
    add_text(f"Таблица {table_number}. {title}", caption)
    data = [[Paragraph(str(cell), small_bold) for cell in headers]]
    data.extend([[Paragraph(str(cell), small) for cell in row] for row in rows])
    table = Table(data, colWidths=[value * cm for value in widths] if widths else None, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor("#263238")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#263238")),
                ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.HexColor("#263238")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))


def add_figure(filename: str, title: str, max_height: float = 10.5) -> None:
    global figure_number
    figure_number += 1
    path = FIGURES / filename
    with PILImage.open(path) as image:
        width_px, height_px = image.size
    width = 16.8
    height = width * height_px / width_px
    if height > max_height:
        height = max_height
        width = height * width_px / height_px
    story.append(Image(str(path), width=width * cm, height=height * cm))
    add_text(f"Рисунок {figure_number}. {title}", caption)


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFont("DaimonCJK", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, height - 1.25 * cm, "Анализ регионального потребления домашних хозяйств")
    canvas.drawRightString(width - 2 * cm, 1.1 * cm, f"Страница {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#AAAAAA"))
    canvas.line(2 * cm, height - 1.4 * cm, width - 2 * cm, height - 1.4 * cm)
    canvas.restoreState()


def first_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#17365D"))
    canvas.setLineWidth(1.5)
    canvas.line(3 * cm, 3 * cm, A4[0] - 3 * cm, 3 * cm)
    canvas.restoreState()


# Загружаем только результаты, полученные после выполнения расчётного процесса.
excluded = read("excluded_territories")
evaluation = read("cluster_evaluation")
profiles = read("cluster_profiles")
selected = read("selected_region_statistics")
adf = read("adf_results")
cycles = read("cycle_statistics").set_index("region_name")
hypotheses = read("arima_hypotheses").set_index("region_name")
metrics = read("metrics")
rolling_metrics = read("rolling_metrics").set_index("region_name")
future = read("future_forecasts")
errors = read("top_errors")
long = read("clean_long_data")

# Титульный лист.
story.append(Spacer(1, 4.2 * cm))
add_text(
    "Анализ и прогнозирование социально-экономических показателей регионов России",
    ParagraphStyle(
        "CoverTitle",
        parent=heading1,
        fontSize=22,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=22,
    ),
)
add_text(
    "Фактическое конечное потребление домашних хозяйств на душу населения",
    ParagraphStyle("CoverSubtitle", parent=body, fontSize=14, leading=21, alignment=TA_CENTER),
)
story.append(Spacer(1, 1.2 * cm))
add_text("Итоговый проект по дисциплине «Временные ряды»", ParagraphStyle("CoverCourse", parent=body, alignment=TA_CENTER))
add_text("Период данных: 2000–2023 годы<br/>Горизонт прогноза: 2024–2026 годы", ParagraphStyle("CoverYears", parent=body, alignment=TA_CENTER))
story.append(PageBreak())

add_heading("Введение")
add_text(
    "Цель работы — выполнить полный цикл анализа годовых временных рядов по субъектам Российской Федерации: подготовить данные, выделить группы сходной динамики, выбрать представителей, сравнить статистические модели и сформулировать рекомендации. Исходный показатель — фактическое конечное потребление домашних хозяйств на душу населения в текущих рыночных ценах, рубли."
)
add_text(
    "Поскольку ряд номинальный, его рост нельзя напрямую интерпретировать как рост реального благосостояния: на него влияют и физическое потребление, и цены. Все прогнозы и рекомендации ниже относятся к номинальным значениям. Источник данных — приложенный файл Росстата <i>rsdocs_1491005410458_wide_appendix_2025.csv</i>."
)

add_heading("Этап 1. Загрузка и подготовка данных")
add_text(
    "Исходный файл содержит 96 записей и 24 годовых столбца (2000–2023), а также четыре служебных поля. Пустой технический столбец после завершающей запятой удалён. В исходном чтении годовые поля являются текстовыми из-за десятичных запятых, пустых ячеек и символа «…». После замены десятичного разделителя и специальных обозначений значения преобразованы в числа."
)
add_text(
    "Из анализа исключены общероссийский итог, восемь федеральных округов и две дублирующие агрегированные записи. Из оставшихся 85 субъектов шесть имеют неполную историю. Длительные фрагменты не восстанавливались интерполяцией или экстраполяцией. Итоговый набор включает <b>79 регионов и 1 896 наблюдений</b> в длинном формате «регион — год»; пропусков в нём нет."
)
add_table(
    "Исключённые территориальные записи",
    ["Территория", "Пропусков", "Причина"],
    [
        [row.region_name, int(row.missing_years), row.reason]
        for row in excluded.itertuples()
    ],
    [7.2, 1.7, 8.0],
)
add_text("В источнике обнаружена латинская буква в названии «Калининградская область»; она нормализована без изменения значений показателя.")

add_heading("Этап 2. Кластерный анализ регионов")
add_heading("2.1. Признаки и алгоритм", 2)
add_text(
    "Для каждого региона рассчитаны шесть признаков: среднее значение; коэффициент вариации; среднегодовой темп прироста CAGR; относительный прирост за весь период; максимальное годовое падение; доля лет с отрицательной динамикой. Среднее сохраняет информацию об уровне, а пять остальных характеристик описывают относительную динамику и волатильность."
)
add_text(
    "Применён K-means: центроиды удобно интерпретировать, а качество решений можно сравнить по silhouette score. Все признаки стандартизированы (среднее 0, стандартное отклонение 1). Для устойчивости использованы 50 начальных инициализаций и фиксированное случайное состояние. Рассмотрены K = 2, 3 и 4."
)
add_table(
    "Сравнение числа кластеров",
    ["K", "Silhouette score", "Inertia", "Минимальный размер кластера"],
    [
        [int(row.n_clusters), num(row.silhouette_score, 4), num(row.inertia, 2), int(row.min_cluster_size)]
        for row in evaluation.itertuples()
    ],
    [2.0, 4.0, 4.0, 5.0],
)
add_text(
    "Выбрано четыре кластера: silhouette score <b>0,3354</b> — наибольший среди рассмотренных. Решение также различает устойчивый рост, периодические снижения и резкую структурную аномалию. При K = 2 и K = 3 качество ниже, а часть содержательных различий исчезает."
)
add_figure("01_silhouette.png", "Silhouette score для решений K-means", 8)
add_figure("02_clusters_pca.png", "Проекция регионов на две главные компоненты", 11)

add_heading("2.2. Содержательная интерпретация кластеров", 2)
add_table(
    "Средние характеристики кластеров",
    ["Кластер", "N", "Среднее, руб.", "CV", "CAGR", "Прирост", "Макс. падение", "Доля снижений"],
    [
        [
            int(row.cluster),
            int(row.n_regions),
            rub(row.mean_value),
            num(row.coefficient_of_variation, 3),
            pct(row.cagr * 100),
            pct(row.total_growth * 100, 1),
            pct(row.max_annual_decline * 100, 1),
            pct(row.negative_years_share * 100, 1),
        ]
        for row in profiles.itertuples()
    ],
    [1.2, 1.0, 2.5, 1.3, 1.7, 2.0, 2.2, 2.2],
)
cluster_descriptions = {
    0: "Умеренный рост и устойчивая динамика: средний CAGR 14,19%, средний уровень 240 359 руб.; максимальные годовые падения в среднем близки к нулю.",
    1: "Наиболее быстрый устойчивый рост: средний CAGR 16,01%, суммарный прирост около 30 раз; в большинстве рядов отрицательные годы отсутствуют.",
    2: "Более неоднородная динамика: CAGR 16,01%, но выше доля лет снижения и заметнее отдельные падения, чем в кластере 1.",
    3: "Особая траектория Красноярского края: скачок 2004 года и снижение на 53,29% в 2005 году. Наблюдения не исправлялись без независимого подтверждения ошибки.",
}
for cluster, row in profiles.set_index("cluster").iterrows():
    add_text(f"<b>Кластер {cluster}.</b> {cluster_descriptions[cluster]}")
    add_text(f"Состав ({int(row['n_regions'])}): {row['regions']}.", small)

add_heading("Этап 3. Выбор регионов для детального прогнозирования")
add_text(
    "В каждом кластере выбран ближайший к центроиду регион. Для перехода от четырёх групп к трём прогнозируемым случаям рассмотрены тройки представителей из разных кластеров и выбрана тройка с максимальной суммой попарных расстояний в стандартизированном признаковом пространстве. Это формализует требование максимального разнообразия динамики."
)
add_table(
    "Базовые статистики выбранных регионов",
    ["Регион", "Кластер", "Среднее", "Минимум (год)", "Максимум (год)", "Наклон, руб./год", "Отн. наклон"],
    [
        [
            row.region_name,
            int(row.cluster),
            rub(row.mean_value),
            f"{rub(row.min_value)} ({int(row.min_year)})",
            f"{rub(row.max_value)} ({int(row.max_year)})",
            rub(row.linear_slope_rub_per_year),
            pct(row.relative_slope_percent),
        ]
        for row in selected.itertuples()
    ],
    [3.4, 1.1, 2.1, 2.5, 2.6, 2.4, 1.8],
)
add_bullet("<b>Республика Коми:</b> ближайший к центру представитель кластера 0, высокий средний уровень и умеренная динамика.")
add_bullet("<b>Курская область:</b> ближайший к центру представитель кластера 1; относительный наклон 10,25% — наибольший среди выбранных.")
add_bullet("<b>Красноярский край:</b> единственный представитель кластера 3, крайний случай, важный для проверки чувствительности моделей к структурному разрыву.")

add_heading("Этап 4. Анализ выбранных временных рядов")
add_text(
    "Тренд выделен HP-фильтром с λ = 6,25 для годовых данных. Остаток фильтра использован как условная циклическая составляющая. Для уровней и первых разностей проведён расширенный тест Дики — Фуллера с константой и автоматическим выбором лага по AIC. Нулевая гипотеза — нестационарность; порог значимости — 5%."
)
add_text(
    "Сезонная декомпозиция не применяется: одно наблюдение в год не содержит внутригодовых повторений. ACF и PACF построены для первых разностей, чтобы снизить влияние доминирующего тренда. Значимые лаги определяются по 95% доверительным границам."
)
region_names = selected["region_name"].tolist()
for index, region in enumerate(region_names, start=1):
    add_heading(f"4.{index}. {region}", 2)
    add_figure(f"03_region_{index}_trend.png", f"{region}: исходный ряд, HP-тренд и цикл", 11)
    region_adf = adf.loc[adf["region_name"] == region]
    add_table(
        f"ADF-тест: {region}",
        ["Ряд", "ADF", "p-value", "Лаг", "Крит. 5%", "Стационарен"],
        [
            [
                "Уровень" if row.transformation == "level" else "Первая разность",
                num(row.adf_statistic, 3),
                num(row.p_value, 4),
                int(row.used_lag),
                num(row.critical_5_percent, 3),
                "Да" if row.stationary_at_5_percent else "Нет",
            ]
            for row in region_adf.itertuples()
        ],
        [3.4, 2.0, 2.0, 1.2, 2.2, 2.5],
    )
    cycle = cycles.loc[region]
    add_text(
        f"Стандартное отклонение циклической составляющей — <b>{rub(cycle['cycle_std_rub'])} руб.</b> "
        f"Минимум цикла наблюдается в {int(cycle['largest_negative_cycle_year'])} году ({rub(cycle['largest_negative_cycle_rub'])} руб.), "
        f"максимум — в {int(cycle['largest_positive_cycle_year'])} году ({rub(cycle['largest_positive_cycle_rub'])} руб.)."
    )
    add_figure(f"04_region_{index}_acf_pacf.png", f"{region}: ACF и PACF первых разностей", 8.5)
    hypothesis = hypotheses.loc[region]
    add_text(
        f"Значимые лаги ACF: {hypothesis['significant_acf_lags']}; PACF: {hypothesis['significant_pacf_lags']}. "
        f"Исходная гипотеза — p = {int(hypothesis['initial_p_hypothesis'])}, q = {int(hypothesis['initial_q_hypothesis'])}. "
        "Она сравнивается далее с автоматическим подбором по AIC, а не используется как жёсткое ограничение."
    )

add_heading("4.4. Сравнение регионов", 2)
add_text(
    "У Республики Коми наибольший абсолютный линейный наклон — 22 570 руб./год, тогда как у Курской области он равен 19 576, у Красноярского края — 20 211 руб./год. После нормировки на средний уровень лидирует Курская область (10,25%). По стандартному отклонению HP-цикла Красноярский край наиболее волатилен (19 618 руб.), затем Коми (11 718 руб.) и Курская область (5 656 руб.)."
)
add_text(
    "ADF не отклоняет нулевую гипотезу ни для одного уровня или первой разности на уровне 5%. Следовательно, формально подтверждённой стационарности нет; это согласуется с меняющимися темпами номинального роста и структурными шоками. Длина ряда мала, а автоматический выбор до девяти лагов сокращает эффективное число наблюдений, поэтому мощность теста ограничена. Значимых лагов ACF/PACF первых разностей не обнаружено ни у одного региона."
)

add_heading("Этап 5. Прогнозирование")
add_heading("5.1. Схема проверки и модели", 2)
add_text(
    "Обучающая выборка — 2000–2018 годы (19 наблюдений, 79,2%), тестовая — 2019–2023 годы (5 наблюдений). Сравниваются наивный прогноз последним значением (<i>NaiveForecaster(strategy=\"last\")</i>), линейное экспоненциальное сглаживание Холта и ARIMA. Для ARIMA перебираются p, q = 0…2 и d = 0…1; порядок выбирается по AIC только на train. Метрики: RMSE в рублях и MAPE в процентах. Основной критерий выбора — минимальный RMSE."
)
metric_rows = []
for row in metrics.itertuples():
    metric_rows.append(
        [row.region_name, row.model_label, row.arima_order, rub(row.rmse), pct(row.mape_percent)]
    )
add_table("Качество моделей на фиксированной тестовой выборке", ["Регион", "Модель", "ARIMA", "RMSE, руб.", "MAPE"], metric_rows, [4.1, 2.3, 2.3, 3.1, 2.1])
add_text(
    "ARIMA оказалась лучшей для всех регионов. Наивный прогноз систематически уступает: он не учитывает тренд, поэтому на растущих номинальных рядах занижает многошаговый прогноз. Holt учитывает тренд, но уступает ARIMA, которая дополнительно описывает зависимость изменений и ошибок."
)

add_heading("5.2. Графики и прогноз до 2026 года", 2)
for index, region in enumerate(region_names, start=1):
    add_figure(f"05_region_{index}_forecast.png", f"{region}: тестовые и будущие прогнозы", 10)

add_table(
    "Прогноз лучших моделей",
    ["Регион", "Модель", "Год", "Прогноз, руб.", "К 2023 году", "Итоговый ARIMA"],
    [
        [row.region_name, row.model_label, int(row.year), rub(row.forecast), pct(row.growth_from_2023_percent), row.final_arima_order]
        for row in future.itertuples()
    ],
    [3.5, 1.8, 1.2, 2.8, 2.2, 2.6],
)
add_text(
    "К 2026 году базовый прогноз составляет 738 676 руб. для Коми (+24,23% к 2023), 622 462 руб. для Курской области (+30,22%) и 668 443 руб. для Красноярского края (+20,15%). Это точечные номинальные сценарии, а не гарантированные значения."
)

add_heading("5.3. Проверка устойчивости rolling forecast", 2)
add_text(
    "Для лучшей модели каждого региона использовано скользящее окно длиной 19 лет. Перед каждым годом 2019–2023 окно сдвигается на один шаг, параметры переоцениваются, строится одношаговый прогноз. Порядок ARIMA зафиксирован по исходному train, чтобы проверка сравнивала обновление параметров, а не повторный выбор спецификации."
)
add_table(
    "Фиксированный и скользящий прогноз лучших моделей",
    ["Регион", "Модель", "RMSE fixed", "RMSE rolling", "MAPE fixed", "MAPE rolling"],
    [
        [row.region_name, row.model_label, rub(row.fixed_rmse), rub(row.rolling_rmse), pct(row.fixed_mape_percent), pct(row.rolling_mape_percent)]
        for row in rolling_metrics.reset_index().itertuples()
    ],
    [3.8, 1.8, 2.6, 2.7, 2.3, 2.4],
)
for region in region_names:
    row = rolling_metrics.loc[region]
    reduction = (1 - row["rolling_rmse"] / row["fixed_rmse"]) * 100
    add_text(
        f"<b>{region}:</b> RMSE снизился с {rub(row['fixed_rmse'])} до {rub(row['rolling_rmse'])} руб. ({pct(reduction, 1)}). "
        "Обновление данных и одношаговый горизонт улучшают точность."
    )
add_text(
    "Сравнение не является полностью равнозначным по горизонту: fixed RMSE относится к пятилетнему прогнозу из одного начала, rolling RMSE — к пяти одношаговым прогнозам. Поэтому устойчивость подтверждается в рамках доступного теста, но ошибку трёхлетнего будущего прогноза нельзя приравнивать к rolling RMSE."
)

add_heading("Этап 6. Анализ ошибок")
add_text(
    "Для каждого региона отобраны пять наибольших по модулю ошибок скользящего прогноза лучшей модели. Поскольку тест содержит пять лет, представлены все они в порядке убывания абсолютной ошибки. Связь с экономическими событиями рассматривается как гипотеза по совпадению во времени, а не как доказанная причинность."
)
for region in region_names:
    add_heading(region, 2)
    region_errors = errors.loc[errors["region_name"] == region]
    add_table(
        f"Топ-5 ошибок: {region}",
        ["Ранг", "Год", "Факт", "Прогноз", "Абс. ошибка", "APE"],
        [
            [int(row.error_rank), int(row.year), rub(row.actual), rub(row.predicted), rub(row.absolute_error), pct(row.absolute_percentage_error)]
            for row in region_errors.itertuples()
        ],
        [1.2, 1.4, 3.0, 3.0, 3.0, 2.0],
    )
    largest = region_errors.iloc[0]
    add_text(
        f"Максимальная ошибка — <b>{rub(largest['absolute_error'])} руб. в {int(largest['year'])} году</b> "
        f"({pct(largest['absolute_percentage_error'])} фактического значения)."
    )
    for row in region_errors.itertuples():
        add_bullet(f"{int(row.year)}: {row.event_hypothesis}.")

add_heading("Этап 7. Индивидуальные рекомендации")
for region in region_names:
    add_heading(region, 2)
    region_future = future.loc[future["region_name"] == region].sort_values("year")
    value_2024 = region_future.iloc[0]["forecast"]
    value_2025 = region_future.iloc[1]["forecast"]
    value_2026 = region_future.iloc[2]["forecast"]
    growth_2026 = region_future.iloc[2]["growth_from_2023_percent"]
    rolling_rmse = rolling_metrics.loc[region, "rolling_rmse"]
    add_bullet(
        f"В номинальном планировании мер поддержки потребления использовать базовый ориентир <b>{rub(value_2026)} руб. на человека к 2026 году</b> "
        f"({pct(growth_2026)} к факту 2023 года), с промежуточными значениями {rub(value_2024)} руб. в 2024-м и {rub(value_2025)} руб. в 2025-м."
    )
    add_bullet(
        f"Ежегодно обновлять модель после публикации фактических данных. Для осторожного сценарного резерва на 2026 год использовать рабочий диапазон "
        f"<b>{rub(value_2026 - rolling_rmse)}–{rub(value_2026 + rolling_rmse)} руб.</b> (точечный прогноз ± один rolling RMSE = {rub(rolling_rmse)} руб.). "
        "Это эмпирический запас по ошибке, а не статистический доверительный интервал."
    )
    if region == "Республика Коми":
        add_bullet("Отдельно контролировать 2026-й как год возможного замедления: прогнозный прирост снижается с 8,05% в 2024 году до 7,07% в 2026-м; наибольшее rolling-отклонение допущено в 2023 году (12,71%).")
    elif region == "Курская область":
        add_bullet("При межрегиональных сопоставлениях учитывать наибольший относительный наклон 10,25% среднего уровня в год: абсолютные лимиты следует пересматривать вместе с быстро растущим номинальным рядом, а не фиксировать на период до 2026 года.")
    else:
        add_bullet("До утверждения долгосрочного ориентира проверить скачок 2004 года и падение 2005-го по первоисточнику. При методологическом разрыве перестроить сопоставимый ряд: именно аномалия выделила Красноярский край в отдельный кластер.")
add_text("Для оценки реального уровня жизни все перечисленные номинальные ориентиры должны быть дополнены региональными индексами потребительских цен.")

add_heading("Этап 8. Общие выводы, ограничения и направления улучшения")
add_text(
    "Кластеризация выделила четыре группы: 19 регионов с умеренной устойчивой динамикой, 45 преимущественно непрерывно растущих рядов, 14 регионов с более заметными снижениями и отдельную аномальную траекторию Красноярского края. Использование относительных признаков позволило учесть не только масштаб, но и различия темпов и волатильности."
)
add_text(
    "Для Республики Коми, Курской области и Красноярского края ARIMA оказалась лучше Holt и наивного бейзлайна по RMSE и MAPE. Скользящая переоценка уменьшила ошибки всех лучших моделей. Точность выше там, где свежая динамика ближе к предшествующему тренду; ошибки растут в годы резких изменений 2021–2023 годов. Наивный прогноз непригоден как основной на трендовых номинальных рядах, но полезен как обязательный ориентир."
)
add_text("<b>Ограничения исследования:</b>")
add_bullet("Только 24 годовых наблюдения; сложные порядки ARIMA и ADF с большим числом лагов имеют ограниченную статистическую надёжность.")
add_bullet("Показатель не дефлирован: прогноз отражает номинальную стоимость и не отделяет инфляцию от реального потребления.")
add_bullet("Модели одномерные и не учитывают доходы, цены, занятость, демографию, отраслевую структуру и будущие шоки.")
add_bullet("Аномалия Красноярского края может быть методологическим разрывом; без первоисточника её нельзя корректно устранить.")
add_bullet("Rolling-проверка одношаговая, тогда как итоговый прогноз трёхлетний; снижение RMSE нельзя механически переносить на больший горизонт.")
add_text("<b>Направления улучшения:</b> добавить региональные дефляторы и объясняющие переменные; проверить сопоставимость рядов; рассмотреть модели с интервенциями для 2009, 2015, 2020 и 2022 годов; оценить несколько горизонтов на расширяющихся окнах; строить интервальные прогнозы; ежегодно обновлять спецификацию и параметры.")
add_text(
    "Таким образом, построенные модели пригодны как воспроизводимый базовый сценарий номинальной динамики, но практические решения требуют регулярной переоценки, сценарного резерва и дополнительной экономической информации."
)

add_heading("Источник данных")
add_text("Федеральная служба государственной статистики. Фактическое конечное потребление домашних хозяйств на душу населения на территории субъектов Российской Федерации, 2000–2023 годы. Файл: rsdocs_1491005410458_wide_appendix_2025.csv, предоставлен в задании.")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(REPORT_PATH),
    pagesize=A4,
    rightMargin=2 * cm,
    leftMargin=2 * cm,
    topMargin=2 * cm,
    bottomMargin=1.8 * cm,
    title="Анализ и прогнозирование регионального потребления домашних хозяйств",
    author="",
    subject="Итоговый проект по временным рядам",
)
doc.build(story, onFirstPage=first_page, onLaterPages=header_footer)
print("Отчёт сохранён: reports/regional_consumption_report.pdf")
