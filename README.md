# TJJM 项目说明

本项目围绕中国省级面板数据，完成了三条主线工作：

1. 预处理与面板构建
2. STIRPAT + 残差学习模型训练与重构
3. 基于政策约束的 2024-2035 年情景预测与可视化

当前已经训练好的模型与结果都保存在 `Code/STIRPAT/output/` 下，便于复现和继续分析。

## 一、项目结构

- [Code/Preprocess/preprocess.py](Code/Preprocess/preprocess.py)
  - 读取原始数据，整理为统一的省级面板数据。
- [Code/STIRPAT/build_training_dataset.py](Code/STIRPAT/build_training_dataset.py)
  - 构建 STIRPAT 特征、计算残差目标，并生成 GRU 训练数据集。
- [Code/STIRPAT/train.py](Code/STIRPAT/train.py)
  - 训练残差学习网络 `EntityEmbeddingGRU`。
- [Code/STIRPAT/evaluate_reconstruction.py](Code/STIRPAT/evaluate_reconstruction.py)
  - 评估 STIRPAT 基线与混合模型的重构效果。
- [Code/STIRPAT/policy_scenario_forecast.py](Code/STIRPAT/policy_scenario_forecast.py)
  - 按政策约束生成 2024-2035 年三情景预测，并输出省级与全国结果。
- [Code/STIRPAT/plot_scenario_forecast.py](Code/STIRPAT/plot_scenario_forecast.py)
  - 对情景预测结果进行可视化。
- [Code/STIRPAT/plot_province_reconstruction.py](Code/STIRPAT/plot_province_reconstruction.py)
  - 绘制历史样本的原始值与预测值对比图。
- [Code/LMDI/](Code/LMDI/)
  - LMDI 分解、稳健性分析与可视化相关脚本。

## 二、模型与方法

本项目的 STIRPAT 部分采用两层结构：

1. 面板岭回归 STIRPAT
   - 用于拟合 `log_CO2` 的主结构项。
   - 代码中的核心特征列为：
     - `log_Population`
     - `log_pGDP`
     - `Industry`
     - `Urbanization`
     - `CoalShare`
     - `log_CarbonIntensity`
     - `log_Energy`
     - `log_PrivateCars`

2. 残差学习网络
   - 模型为 `EntityEmbeddingGRU`。
   - 输入包括动态特征窗口、历史残差以及省份嵌入。
   - 训练完成的模型文件位于 [Code/STIRPAT/output/model/best_ee_gru.pt](Code/STIRPAT/output/model/best_ee_gru.pt)

## 三、当前已完成的结果

### 1. 训练与重构结果

- 模型训练结果：
  - [Code/STIRPAT/output/model/train_metrics.json](Code/STIRPAT/output/model/train_metrics.json)
- 重构评估结果：
  - [Code/STIRPAT/output/model/reconstruction_metrics.json](Code/STIRPAT/output/model/reconstruction_metrics.json)
- 训练集与验证集重构明细：
  - [Code/STIRPAT/output/model/train_reconstruction_detail.csv](Code/STIRPAT/output/model/train_reconstruction_detail.csv)
  - [Code/STIRPAT/output/model/valid_reconstruction_detail.csv](Code/STIRPAT/output/model/valid_reconstruction_detail.csv)

### 2. 政策情景预测结果

预测区间：2024-2035 年。

三种情景：
- baseline
- low_carbon
- extensive

全国达峰结果如下：

- baseline：2035 年，13514.44，预测期内未达峰
- low_carbon：2035 年，12753.52，预测期内未达峰
- extensive：2035 年，14321.16，预测期内未达峰

对应汇总文件：
- [Code/STIRPAT/output/scenario_forecast/scenario_peak_summary.csv](Code/STIRPAT/output/scenario_forecast/scenario_peak_summary.csv)
- [Code/STIRPAT/output/scenario_forecast/scenario_peak_summary.json](Code/STIRPAT/output/scenario_forecast/scenario_peak_summary.json)

### 3. 整理后的集中结果包

为了避免文件分散，已经将结果统一整理到：

- [Code/STIRPAT/output/scenario_forecast/organized](Code/STIRPAT/output/scenario_forecast/organized)

该目录下文件含义如下：

- [Code/STIRPAT/output/scenario_forecast/organized/00_combined_long.csv](Code/STIRPAT/output/scenario_forecast/organized/00_combined_long.csv)
  - 统一长表，包含省级与全国结果
- [Code/STIRPAT/output/scenario_forecast/organized/01_province_yearly_detail.csv](Code/STIRPAT/output/scenario_forecast/organized/01_province_yearly_detail.csv)
  - 各省逐年逐情景预测明细
- [Code/STIRPAT/output/scenario_forecast/organized/02_province_peak_summary.csv](Code/STIRPAT/output/scenario_forecast/organized/02_province_peak_summary.csv)
  - 各省峰值年份与峰值结果
- [Code/STIRPAT/output/scenario_forecast/organized/03_national_yearly.csv](Code/STIRPAT/output/scenario_forecast/organized/03_national_yearly.csv)
  - 全国逐年加总结果
- [Code/STIRPAT/output/scenario_forecast/organized/04_national_peak_summary.csv](Code/STIRPAT/output/scenario_forecast/organized/04_national_peak_summary.csv)
  - 全国达峰汇总
- [Code/STIRPAT/output/scenario_forecast/organized/manifest.json](Code/STIRPAT/output/scenario_forecast/organized/manifest.json)
  - 文件清单说明

### 4. 可视化结果

- 全国三情景趋势图：
  - [Code/STIRPAT/output/scenario_forecast/scenario_forecast_national_trend.png](Code/STIRPAT/output/scenario_forecast/scenario_forecast_national_trend.png)
- 关键年份对比图：
  - [Code/STIRPAT/output/scenario_forecast/scenario_forecast_key_years.png](Code/STIRPAT/output/scenario_forecast/scenario_forecast_key_years.png)

## 四、关键输入变量与政策约束

情景预测中使用的 8 个未来路径变量如下：

- `log_Population`
- `log_pGDP`
- `log_Energy`
- `CoalShare`
- `Industry`
- `Urbanization`
- `log_CarbonIntensity`
- `log_PrivateCars`

其中：

- 对数变量采用年均增长率递推
- 原始值变量采用百分点变化递推
- 各情景差异主要体现在能源结构、产业结构和碳强度上

政策约束与阶段性参数已写入 [policy.md](policy.md)。

## 五、如何运行

### 1. 重新生成训练数据

```bash
D:/Anaconda3/python.exe Code/STIRPAT/build_training_dataset.py
```

### 2. 重新训练模型

```bash
D:/Anaconda3/python.exe Code/STIRPAT/train.py
```

### 3. 重新评估重构效果

```bash
D:/Anaconda3/python.exe Code/STIRPAT/evaluate_reconstruction.py
```

### 4. 重新生成政策情景预测

```bash
D:/Anaconda3/python.exe Code/STIRPAT/policy_scenario_forecast.py
```

### 5. 重新生成可视化

```bash
D:/Anaconda3/python.exe Code/STIRPAT/plot_scenario_forecast.py
```

## 六、数据与输出位置

主要数据文件：

- 原始面板数据：`Code/Preprocess/output/panel_master.csv`
- STIRPAT 数据集：`Code/STIRPAT/output/dataset/stirpat_ee_gru_dataset.npz`
- 残差面板：`Code/STIRPAT/output/dataset/panel_with_residual.csv`
- 已训练模型：`Code/STIRPAT/output/model/best_ee_gru.pt`

主要结果目录：

- [Code/STIRPAT/output/model](Code/STIRPAT/output/model)
- [Code/STIRPAT/output/scenario_forecast](Code/STIRPAT/output/scenario_forecast)
- [Code/STIRPAT/output/scenario_forecast/organized](Code/STIRPAT/output/scenario_forecast/organized)

## 七、说明

- 当前预测脚本默认以 2023 年为基年，推演到 2035 年。
- 全国结果由各省结果逐年加总得到。
- 省级结果、全国结果和达峰结果已经统一归档，后续写论文或做图时建议优先读取 `organized` 目录中的文件。
