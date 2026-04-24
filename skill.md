---
name: tjjm-project-memory
description: "TJJM 项目记忆，记录已处理数据、主表产物、目录边界与后续约束。"
---

# 项目记忆

更新时间: 2026-04-24

## 一、当前状态

- 原始 Excel 数据已经处理完成，skill.md 不再维护逐个文件的内容索引和表格摘要。
- 后续工作重点是基于已生成的面板产物继续建模和分析，而不是继续扫描原始表格。
- 省级面板主线保持不变：进入回归或解释分析的变量必须使用省级口径。

## 二、已生成的关键产物

- 主表与审计产物位于 [Code/Dataset/output](Code/Dataset/output)。
- LMDI 结果位于 [Code/LMDI/output](Code/LMDI/output)。
- STIRPAT 结果位于 [Code/Model/output](Code/Model/output)。
- 机器索引仍可通过 [Code/dataset_index_summary.json](Code/dataset_index_summary.json) 追溯，但不再把 Excel 目录清单写入 skill.md。

## 三、当前约束

- 国家级变量只用于背景描述或兜底，不直接混入省级回归。
- LMDI 与 STIRPAT 目录分离，互不写入对方输出目录。
- 以后如需追踪原始文件，优先看机器索引和已输出 CSV/JSON，而不是在 skill.md 里维护 Excel 清单。

## 四、本次改动

- 删除 skill.md 中按 Excel 分类的长索引、文件级摘要、缺失值清单和重复的建模建议。
- 将 skill.md 收缩为“已处理数据版”项目记忆，只保留当前仍有用的主线、产物和边界。
- 新建 [论文思路与方法顺序.md](论文思路与方法顺序.md)，单独记录 LMDI 与 STIRPAT 的论文叙事顺序。

## 五、踩坑

- 终端中文路径和中文常量匹配仍可能不稳定，继续依赖人工 Excel 清单收益已经很低。
- 原始数据目录继续扩展会让人工索引迅速过期，因此不再在 skill.md 里维护逐文件内容。

## 六、解决方法

- 用机器索引和已输出结果文件作为事实源。
- 将 skill.md 从数据目录清单改为流程记忆和边界说明。

## 七、验证结果

- 已确认 skill.md 不再包含 Excel 内容索引和逐表摘要。
- 已确认当前主要产物路径边界清晰，后续可直接基于输出文件继续分析。
