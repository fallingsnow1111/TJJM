# Kaya 预测框架

## 1. 预测目标

本文的预测部分不再直接回归 CO2，而是采用 Kaya 驱动预测，但在执行层面先用历史数据回归生成总能耗强度 $EI=Energy/GDP$，再闭合到 Kaya 恒等式中，从而避免结构项 $S$ 与强度项 $B$ 的重复计量：

$$
CO_2(t) = P_t \times A_t \times EI_t \times C_t
$$

其中，$P$ 为人口，$A$ 为人均 GDP，$EI$ 为总能耗强度，$C$ 为碳排放因子。$S$ 仍保留为结构解释变量，用于说明产业升级路径，但不再与 $B$ 在重构方程中同时相乘。

## 2. 预测思路

采用“两层结构”：

1. 第一层，对人口、人均 GDP、结构项和排放因子构建时间路径，并用历史 $A$、$S$ 回归生成 $EI$；
2. 第二层，按 Kaya 恒等式直接乘回，得到 CO2 预测值。

这里不再对 CO2 单独回归，因为那样会破坏前文的机制分解逻辑。

## 3. 政策约束

政策限制已单独记录在 [Paper/碳排放/十五五政策约束.md](十五五政策约束.md)，机器可读配置见 [Code/forecast_policy_15th_fyp.json](../../Code/forecast_policy_15th_fyp.json)。

核心做法是：

- 不再使用桥接期平滑峰值，避免把峰值位置写成调参结果；
- 用历史 $A$ 和 $S$ 的回归关系生成 $EI$，并以最近 4 年残差趋势作为默认外推项；
- 各情景通过乘数调整 A、S、C 的年均增速，$EI$ 由模型内部生成。

## 4. 情景设计

- `baseline`: 基准政策路径；
- `policy_strengthening`: 政策强化路径；
- `high_growth`: 高增长路径；
- `policy_lag`: 政策滞后路径。

当前默认设置下，基准情景的峰值会后移到 2030 年，且对残差窗口存在敏感性。若将窗口改成 3 年，峰值会前移到 2025；若改成 5 年，峰值会推迟到 2035。因此正文应把结果写成情景区间，并明确这是模型推导结果，而非手工校准结果。

## 5. 已生成结果

- 预测路径数据： [Code/output/kaya_forecast_paths.csv](../../Code/output/kaya_forecast_paths.csv)
- 预测摘要： [Code/output/kaya_forecast_summary.md](../../Code/output/kaya_forecast_summary.md)
- 预测图： [Code/output/figures/kaya_forecast_paths.png](../../Code/output/figures/kaya_forecast_paths.png)

当前脚本采用历史回归 + 残差趋势的生成式路径。这样可以避免把峰值位置写成桥接权重的函数，也更符合“预测由模型推导而非调参推出”的建模逻辑。如果论文需要更稳健的表述，应优先报告残差窗口敏感性，而不是回到把 S 与 B 同时乘回去的做法。

## 6. 写作建议

正文建议写成三段：

1. 预测方法：说明五因素 Kaya + 历史回归生成的 EI；
2. 情景结果：说明不同场景的峰值年份与 2030/2035 水平；
3. 机制解释：说明 A、S 和残差趋势如何共同决定达峰提前或推迟。
