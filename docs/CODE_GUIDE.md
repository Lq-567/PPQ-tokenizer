# 代码导读

## PPQ

`tokenizer/tokenizer_core.py` 中的 `LearnablePQ` 包括输入向量的编码与解码、分组码本的最近邻量化和 EMA 码本更新；训练使用直通估计器。`tokenizer/tokenizer_runner.py` 负责按训练物品集合加载输入、读取初始化、训练及导出 SID。`tokenizer/prefix_objective.py` 的 `equal_weight_objective` 实现 `(完整重建误差 + 所有非完整累计前缀误差之和) / G + 0.25 × commitment`，每个前缀均按原始组顺序保留。

## FLASH

`src/flash_helpers.py` 中的 `FrozenOrderedTokenizer` 将 PPQ 导出的码映射到 FLASH 的 item token 格式；`SharedFlash.item_scores` 汇总各码位的对数概率，为完整商品目录打分。`vendor/Flash/genrec/models/Flash/model.py` 是所用 FLASH 快照。

`run.py` 从头构造 FLASH。为保持原实验的初始化随机数消耗，它先按照原代码构造文本投影器，然后将 Alignment 权重设为零、冻结投影器并注册禁止调用的检查。训练只用训练划分；验证和测试不回传梯度。梯度累积按 microbatch 的序列数量占比计算，不能换成有效标签数量占比，否则不再是同一个训练算术。

## 评估

入口保存 0、每 500 更新及终点的模型与 validation/test 预测。评估对每位用户的全部 item 打分，使用 item ID 对同分结果稳定排序。`metrics.json` 保存每个节点，`selections.json` 保存两种选择，`comparison.json` 将两个原实验选点的预测与打包参考数据对比。评估时保存并恢复随机数状态，避免改变后续训练流。

`configs/flash.json` 和 `configs/ppq.json` 保存实验配置；`run.py` 将输入和输出路径替换为当前项目路径。传入 `--sid-from` 时不会运行 PPQ 训练，且在 `sid_comparison.json` 标记复用。
