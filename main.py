# -*- coding: utf-8 -*-
"""RWH Predictor —— 屋面雨水回收经济性预测 App（Flet 跨平台）。"""
import os
import joblib
import numpy as np
import pandas as pd
import flet as ft
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

APP_NAME = "RWH Predictor"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _asset(name):
    for p in (os.path.join(_HERE, "assets", name), os.path.join(_HERE, name)):
        if os.path.exists(p):
            return p
    return os.path.join(_HERE, "assets", name)


# ============ 1. 加载模型与历史数据 ============
bundle = joblib.load(_asset("rf_bcr_model.joblib"))
model, features = bundle["model"], bundle["features"]

df = pd.read_excel(_asset("clean_data.xlsx"))
X = df[features].astype(float)
y = df["BCR"].astype(float)

# 与训练完全相同的划分：3:1，random_state=42（用于校验指标）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
_test_pred = model.predict(X_test)
METRICS = {
    "r2": r2_score(y_test, _test_pred),
    "mae": mean_absolute_error(y_test, _test_pred),
    "rmse": float(np.sqrt(mean_squared_error(y_test, _test_pred))),
}

STATS = {
    "min": X.min().to_dict(),
    "max": X.max().to_dict(),
    "p1": X.quantile(0.01).to_dict(),
    "p99": X.quantile(0.99).to_dict(),
    "mean": X.mean().to_dict(),
    "std": X.std().replace(0, 1).to_dict(),
}
HIST_X = X.values.tolist()
HIST_Y = y.values.tolist()

DEFAULTS = {
    "P 水价": 0.45, "R 降雨量": 1700, "CV 变异系数": 1.2,
    "CA 汇水面积": 800, "BA 总建筑面积": 6000, "N 人数": 120,
    "D 需水量": 12, "H 楼高": 18, "T 水箱容积": 40,
    "L 项目寿命": 30, "E 电价": 0.10, "Dr 贴现率": 4.5,
}


# ============ 2. 推理逻辑 ============
def predict_bcr(x):
    return float(model.predict([x])[0])


def extrapolation_warnings(x):
    bad = []
    for i, f in enumerate(features):
        v = x[i]
        if v < STATS["p1"][f] or v > STATS["p99"][f]:
            bad.append(f"{f}={v:g}")
    return bad


def find_similar(x, k=3):
    mean = [STATS["mean"][f] for f in features]
    std = [STATS["std"][f] for f in features]
    zx = [(v - mean[j]) / std[j] for j, v in enumerate(x)]
    dists = []
    for i, row in enumerate(HIST_X):
        d = 0.0
        for j in range(len(features)):
            z = (row[j] - mean[j]) / std[j]
            d += (z - zx[j]) ** 2
        dists.append((float(np.sqrt(d)), HIST_Y[i], row))
    dists.sort(key=lambda r: r[0])
    return dists[:k]


# ============ 3. 界面 ============
def main(page: ft.Page):
    page.title = APP_NAME
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 16
    page.theme_mode = ft.ThemeMode.LIGHT

    fields = []
    input_rows = []
    for i, f in enumerate(features):
        tf = ft.TextField(
            label=f,
            value=str(DEFAULTS[f]),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text=f"历史范围 {STATS['min'][f]:g} ~ {STATS['max'][f]:g}",
            dense=True,
            expand=True,
        )
        fields.append(tf)
        if i % 2 == 0:
            input_rows.append(ft.Row([], spacing=10, expand=True))
        input_rows[-1].controls.append(tf)

    bcr_text = ft.Text("--", size=36, weight=ft.FontWeight.BOLD)
    verdict = ft.Container(
        content=ft.Text("等待预测", size=16, weight=ft.FontWeight.W_600),
        padding=ft.Padding(left=18, right=18, top=8, bottom=8),
        border_radius=20,
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
    )
    warn_text = ft.Text("", color=ft.Colors.ORANGE_800, size=13)
    similar_column = ft.Column(spacing=8)

    def render_similar(x):
        similar_column.controls.clear()
        for rank, (d, bcr, row) in enumerate(find_similar(x, 3), start=1):
            summary = "，".join(f"{features[j]} {row[j]:g}" for j in range(4))
            similar_column.controls.append(
                ft.Row([
                    ft.Text(f"#{rank}", weight=ft.FontWeight.W_600),
                    ft.Text(f"真实 BCR={bcr:.4f}", expand=True),
                    ft.Text(f"距离 {d:.2f}"),
                ], spacing=10)
            )
            similar_column.controls.append(
                ft.Text(summary, size=12, color=ft.Colors.OUTLINE)
            )

    def on_predict(e):
        try:
            x = [float(tf.value) for tf in fields]
        except (TypeError, ValueError):
            bcr_text.value = "输入无效"
            verdict.content.value = "请填写有效数字"
            warn_text.value = ""
            similar_column.controls.clear()
            page.update()
            return

        bcr = predict_bcr(x)
        bcr_text.value = f"{bcr:.4f}"
        feasible = bcr >= 1
        verdict.content.value = "经济可行" if feasible else "不可行"
        verdict.bgcolor = ft.Colors.GREEN_100 if feasible else ft.Colors.RED_100
        verdict.content.color = ft.Colors.GREEN_800 if feasible else ft.Colors.RED_800

        bad = extrapolation_warnings(x)
        warn_text.value = (
            "⚠ 外推风险：" + "；".join(bad) + "（超出历史 1–99 分位）"
            if bad else "输入处于历史数据范围内。"
        )
        render_similar(x)
        page.update()

    predict_btn = ft.FilledButton(
        content=ft.Text("开始预测", size=16),
        on_click=on_predict,
        expand=True,
        height=48,
    )

    metric_cells = ft.Row([
        _metric_cell("R²", f"{METRICS['r2']:.4f}"),
        _metric_cell("MAE", f"{METRICS['mae']:.4f}"),
        _metric_cell("RMSE", f"{METRICS['rmse']:.4f}"),
    ], spacing=10)

    page.add(
        ft.Card(content=ft.Container(content=ft.Column([
            ft.Text("项目输入参数", size=15, weight=ft.FontWeight.W_600),
            *input_rows, predict_btn,
        ], spacing=12), padding=16)),
        ft.Card(content=ft.Container(content=ft.Column([
            ft.Text("预测结果", size=15, weight=ft.FontWeight.W_600),
            bcr_text, verdict, warn_text,
        ], spacing=10), padding=16)),
        ft.Card(content=ft.Container(content=ft.Column([
            ft.Text("测试集校验（3:1，random_state=42）",
                    size=15, weight=ft.FontWeight.W_600),
            metric_cells,
        ], spacing=10), padding=16)),
        ft.Card(content=ft.Container(content=ft.Column([
            ft.Text("最相似的 3 个历史项目",
                    size=15, weight=ft.FontWeight.W_600),
            similar_column,
        ], spacing=10), padding=16)),
    )


def _metric_cell(label, value):
    return ft.Container(
        expand=True,
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
        border_radius=10,
        padding=14,
        content=ft.Column([
            ft.Text(value, size=18, weight=ft.FontWeight.W_700,
                    text_align=ft.TextAlign.CENTER),
            ft.Text(label, size=12, color=ft.Colors.OUTLINE,
                    text_align=ft.TextAlign.CENTER),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
    )


if __name__ == "__main__":
    ft.run(main)
