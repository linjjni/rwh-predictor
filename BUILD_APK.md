# 打包 Android APK 说明

## 本机（电脑端）先跑通

在工程目录下：



```
flet run
```

界面有 12 个输入框、「开始预测」按钮、结果面板、测试集校验指标、最相似 3 个历史项目。

默认样本预测 BCR=0.0231，判定不可行；App 内重跑测试集 R²=0.6270、MAE=0.1139、RMSE=0.2313。

## 出 APK 的环境要求

Flet 官方平台矩阵：**APK 只能在 macOS 或 Linux 上构建**；原生 Windows 不支持打 APK

（Windows 上仅可走 WSL 内的 Linux 工具链）。构建前需装好：



* Flutter SDK（首次 `flet build` 会自动下载到 `~/flutter/<version>`）

* JDK 17

* Android SDK：cmdline-tools、platform-tools、build-tools、platform（API 35 左右），并已接受 license

## 出包命令（在 Linux /macOS/ WSL 内，工程目录下执行）



```
flet build apk --product "RWH Predictor" --org com.rwh --bundle-id com.rwh.predictor
```

产物在 `build/apk/` 目录下，把生成的 `.apk` 发到同学手机，开启「允许安装未知来源应用」后侧载安装。

## 没有 Linux/macOS 机器时：用 GitHub Actions 云端出包

仓库里放 `.github/workflows/build-apk.yml`：



```
name: build-apk

on: \[push, workflow\_dispatch]

jobs:

&#x20; apk:

&#x20;   runs-on: ubuntu-latest

&#x20;   steps:

&#x20;     - uses: actions/checkout@v4

&#x20;     - uses: astral-sh/setup-uv@v6

&#x20;     - run: uv sync

&#x20;     - run: uv run flet build apk --product "RWH Predictor" --yes

&#x20;     - uses: actions/upload-artifact@v4

&#x20;       with:

&#x20;         name: rwh-apk

&#x20;         path: build/apk
```

流水线跑完后在 Artifacts 里下载 APK。

## 说明



* 运行时为纯标准库推理：电脑端先跑 `python build_assets.py`，把上级目录的 `rf_bcr_model.joblib` 序列化为 `src/assets/rf_trees.json`（含树、统计量、历史、测试集、默认值）；App 只依赖 `flet`，手机端无需安装 sklearn/pandas/numpy，打包成功率高、体积小。

* iOS 上架 App Store 需付费 Apple Developer 账号，本轮不安排；本工程同样可在 macOS 上 `flet build ipa`。