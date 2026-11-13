# RAG 评测集阶段 2.1 审计报告

## 审计范围

- 审计对象：`rag_eval_50.jsonl`、`rag_eval_50_review.csv`。
- 审计方法：逐题检查 PDF 原文、evidence、standard answer、evaluation_points、Chroma node_id 和 DocStore node_id。
- 未使用待测 LLM 自动判定标准答案。
- 未修改 PDF、Chroma、DocStore、Embedding、Reranker、LLM 或检索代码。

## 修改的问题 ID 和原因

### INSTALL-09

- 原因：Internet 要求确有原文支持，因此答案和评分点保留。
- 修改：evidence 补入独立原文句，避免评分点超出 evidence 覆盖范围。
- 原文证据：
  - `If you’re using your webcam with Windows® 8, the software required is downloaded automatically when you connect your webcam for the first time, and installed silently in the background.`
  - `Important: Make sure you’re connected to the Internet, so the software can be downloaded.`
- 来源：高清摄像头.pdf 物理页 5，page_label 5；node_id `4590a2b2-6410-43a6-99be-28219f708338`。

### PARAM-01

- 原因：答案同时包含“少量油或不用油”和“减少 90% 脂肪”，旧 evidence 只覆盖后者。
- 修改：补全完整原文证据；答案和评分点不变。
- 原文证据：`只需少量油或不用油即可炸制您喜爱的食 物，同时减少 90% 的脂肪量。`
- 来源：空气炸锅.pdf 物理页 6，page_label 6；node_id `89b02e4f-ff9f-4967-b6f5-54e6ba820882`。

### TROUBLE-04

- 原因：旧答案添加了原文没有明确规定的检查优先级。
- 修改：问题改为询问可能原因和对应处理建议；答案并列说明设备故障与低温存放两种原因，不再声称必须先检查低温。
- 原文证据：`设备已损坏或出现故障。 ... 您的 Airfryer 空气炸锅可能存放在一个太冷的位置。如果您的设备是在较低的环境温度下存放的，请将它升温到室温至少 15 分钟，然后重新插电。如果您的显示屏仍显示“E1”，请拨打飞利浦服务热线或联系您所在国家/地区的客户服务中心。`
- 来源：空气炸锅.pdf 物理页 21，page_label 21；node_id `96b838fd-1db1-4728-a97e-79a3be248713`。

### MODEL-01

- 原因：旧 evidence 只证明 `LA9039` 出现在规格页，不能单独证明它归属哪个产品。
- 修改：补充 OBD 产品封面证据，并将两个有效节点写入 expected_chunks。
- 原文证据 1：`Instruction Manual OBD II 4G/GPS Tracking Device`，来源物理页 1，page_label 1；node_id `5ef1eb63-d1f1-4918-bd75-5f2e8f21a8b2`。
- 原文证据 2：`17 LA9039 SPECIFICATIONS`，来源物理页 18，page_label 17；node_id `782b4213-034f-436c-9f88-ee4325dd9868`。

## 其余 46 条审计结果

- 未发现需要修改 question、standard answer 或 evaluation_points 的条目。
- 所有标准答案要点均能由对应当前 evidence 覆盖。
- 共用 expected_chunk 的题目均保留，未为了节点唯一性改写题目。
- 支持多个有效证据节点的 expected_chunks 命中规则。

## 50 条数据完整性检查

| 检查项 | 结果 |
|---|---|
| JSONL 条数 | 50 |
| 唯一 question_id | 50 |
| 参数查询 | 10 |
| 安装操作 | 10 |
| 故障排查 | 10 |
| 跨产品型号归属辨析 | 10 |
| 跨语言检索 | 10 |
| 必需字段完整 | PASS |
| 答案非空 | PASS |
| evaluation_points 非空 | PASS |
| evidence 页面匹配 | PASS |
| expected_chunks 在 Chroma/DocStore 中存在 | PASS |
| CSV 与 JSONL 同步 | PASS |

## 逐题语义审计记录

| question_id | evidence | node_id | evaluation_points 非空 | 答案要点覆盖 |
|---|---|---|---|---|
| PARAM-01 | PASS | PASS | PASS | PASS |
| PARAM-02 | PASS | PASS | PASS | PASS |
| PARAM-03 | PASS | PASS | PASS | PASS |
| PARAM-04 | PASS | PASS | PASS | PASS |
| PARAM-05 | PASS | PASS | PASS | PASS |
| PARAM-06 | PASS | PASS | PASS | PASS |
| PARAM-07 | PASS | PASS | PASS | PASS |
| PARAM-08 | PASS | PASS | PASS | PASS |
| PARAM-09 | PASS | PASS | PASS | PASS |
| PARAM-10 | PASS | PASS | PASS | PASS |
| INSTALL-01 | PASS | PASS | PASS | PASS |
| INSTALL-02 | PASS | PASS | PASS | PASS |
| INSTALL-03 | PASS | PASS | PASS | PASS |
| INSTALL-04 | PASS | PASS | PASS | PASS |
| INSTALL-05 | PASS | PASS | PASS | PASS |
| INSTALL-06 | PASS | PASS | PASS | PASS |
| INSTALL-07 | PASS | PASS | PASS | PASS |
| INSTALL-08 | PASS | PASS | PASS | PASS |
| INSTALL-09 | PASS | PASS | PASS | PASS |
| INSTALL-10 | PASS | PASS | PASS | PASS |
| TROUBLE-01 | PASS | PASS | PASS | PASS |
| TROUBLE-02 | PASS | PASS | PASS | PASS |
| TROUBLE-03 | PASS | PASS | PASS | PASS |
| TROUBLE-04 | PASS | PASS | PASS | PASS |
| TROUBLE-05 | PASS | PASS | PASS | PASS |
| TROUBLE-06 | PASS | PASS | PASS | PASS |
| TROUBLE-07 | PASS | PASS | PASS | PASS |
| TROUBLE-08 | PASS | PASS | PASS | PASS |
| TROUBLE-09 | PASS | PASS | PASS | PASS |
| TROUBLE-10 | PASS | PASS | PASS | PASS |
| MODEL-01 | PASS | PASS | PASS | PASS |
| MODEL-02 | PASS | PASS | PASS | PASS |
| MODEL-03 | PASS | PASS | PASS | PASS |
| MODEL-04 | PASS | PASS | PASS | PASS |
| MODEL-05 | PASS | PASS | PASS | PASS |
| MODEL-06 | PASS | PASS | PASS | PASS |
| MODEL-07 | PASS | PASS | PASS | PASS |
| MODEL-08 | PASS | PASS | PASS | PASS |
| MODEL-09 | PASS | PASS | PASS | PASS |
| MODEL-10 | PASS | PASS | PASS | PASS |
| CROSS-01 | PASS | PASS | PASS | PASS |
| CROSS-02 | PASS | PASS | PASS | PASS |
| CROSS-03 | PASS | PASS | PASS | PASS |
| CROSS-04 | PASS | PASS | PASS | PASS |
| CROSS-05 | PASS | PASS | PASS | PASS |
| CROSS-06 | PASS | PASS | PASS | PASS |
| CROSS-07 | PASS | PASS | PASS | PASS |
| CROSS-08 | PASS | PASS | PASS | PASS |
| CROSS-09 | PASS | PASS | PASS | PASS |
| CROSS-10 | PASS | PASS | PASS | PASS |

## 仍然存在的质量限制

- “跨产品型号归属辨析”不是相邻型号混淆测试，因为现有资料没有足够的相邻型号说明书。
- OBD PDF 的封面前两个物理页对应同一个 page_label，数据通过 source_page 与 source_page_label 同时记录以避免混淆。
- TROUBLE-01 使用两个重叠切片作为有效证据节点，Recall@3 应按任意命中。
- MODEL-01 需要两个页面共同支持，完整信息已在 evidence 和审计报告中记录。
- 本次只审计评测集事实和字段完整性，未运行三组检索实验，也未产生 Recall@3、耗时或成本结果。
- 答案正确率仍须由人工评分，不由待测 LLM 自评。
