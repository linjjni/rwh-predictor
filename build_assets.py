# -*- coding: utf-8 -*-
"""
构建脚本：把训练好的随机森林（上级目录 rf_bcr_model.joblib）
序列化为 App 资源 src/assets/rf_trees.json，运行时仅用标准库即可推理，
无需在手机端安装 sklearn/pandas/numpy。
在电脑上运行：python build_assets.py
"""
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "..", "rf_bcr_model.joblib")
CLEAN = os.path.join(BASE, "..", "clean_data.xlsx")
OUT = os.path.join(BASE, "src", "assets", "rf_trees.json")

bundle = joblib.load(MODEL)
model, features = bundle["model"], bundle["features"]

df = pd.read_excel(CLEAN)
X = df[features].astype(float)

def serialize_tree(tree):
    n = tree.node_count
    return {
        "f": [int(v) for v in tree.feature],
        "t": [float(v) for v in tree.threshold],
        "l": [int(v) for v in tree.children_left],
        "r": [int(v) for v in tree.children_right],
        "v": [float(tree.value[i][0][0]) for i in range(n)],
    }

X_train, X_test, y_train, y_test = train_test_split(
    X, df["BCR"].astype(float), test_size=0.25, random_state=42
)

data = {
    "features": features,
    "trees": [serialize_tree(est.tree_) for est in model.estimators_],
    "history": {"X": X.values.tolist(), "y": df["BCR"].astype(float).values.tolist()},
    "test": {"X": X_test.values.tolist(), "y": y_test.values.tolist()},
    "stats": {
        "min": X.min().round(6).to_dict(),
        "max": X.max().round(6).to_dict(),
        "p1": X.quantile(0.01).round(6).to_dict(),
        "p99": X.quantile(0.99).round(6).to_dict(),
        "mean": X.mean().round(6).to_dict(),
        "std": X.std().replace(0, 1).round(6).to_dict(),
    },
    "defaults": {
        "P 水价": 0.45, "R 降雨量": 1700, "CV 变异系数": 1.2,
        "CA 汇水面积": 800, "BA 总建筑面积": 6000, "N 人数": 120,
        "D 需水量": 12, "H 楼高": 18, "T 水箱容积": 40,
        "L 项目寿命": 30, "E 电价": 0.10, "Dr 贴现率": 4.5,
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print("已生成：", os.path.abspath(OUT), "树数：", len(data["trees"]))
