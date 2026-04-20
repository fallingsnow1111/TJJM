# 省级面板模型主线与数据缺口

更新时间: 2026-04-20

## 1. 研究主线（必须统一）

本项目方法主线统一为：省级面板模型（province-year）。

约束规则：
- 凡是进入 LMDI 或回归模型的变量，必须全部为省级数据。
- 国家级数据只能用于背景描述、趋势对比、政策约束，不进入省级分解与回归方程。

## 2. 最低必要变量（入模硬约束）

以下 5 类变量是最低必要项，且必须为省级口径：
- CO2（碳排放）
- Energy（能源消费）
- GDP
- Population
- Industry（产业结构，优先第二产业占比）

可选增强变量（建议省级）：
- Urbanization（城镇化率）
- Energy mix（煤炭占比、清洁能源占比等）

## 3. 当前状态与急需补齐项

当前状态：
- 已对齐为省级：CO2
- 当前为国家级代理：GDP、Population、Energy、Industry、Urbanization

急需补齐（第一优先级）：
- 省级 GDP
- 省级 Population

随后补齐（第二优先级）：
- 省级 Energy
- 省级 Industry
- 省级 Urbanization

## 4. 统一主表目标结构

目标主表字段：
- province
- year
- CO2
- GDP
- Population
- Energy
- Industry
- Urbanization

目标时段：
- 2001-2022（保证高质量省级完整性）

扩展时段策略：
- 1990-2000 在满足同口径省级解释变量的前提下再扩展。

## 5. 执行顺序（落地）

- 第一步：补齐省级 GDP 与省级 Population。
- 第二步：补齐省级 Energy、Industry、Urbanization。
- 第三步：重建统一省级主表并做口径一致性审计。
- 第四步：运行 LMDI 分解。
- 第五步：运行省级面板回归（固定效应模型）。

## 6. 风险红线

禁止将国家级解释变量直接混入省级 LMDI 或省级回归模型。

原因：
- 会导致分解解释失真。
- 会产生聚合偏误（aggregation bias）。
- 会弱化或扭曲省际差异识别。
