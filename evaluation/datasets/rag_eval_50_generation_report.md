# RAG 50 条评测集生成报告

## 阶段 2.1 人工审核修正

- 已完成 4 条定点修正：INSTALL-09、PARAM-01、TROUBLE-04、MODEL-01。
- 其余 46 条问题文本和标准答案未改变。
- 五类各 10 条、50 个 question_id 和全部字段结构保持不变。
- 未修改 PDF、Chroma、DocStore、Embedding、Reranker、LLM 或检索代码。

## 索引前置检查

- 云台 PDF 单文件入库完成。
- 首次切片尝试因 NLTK `stopwords` 和 `punkt_tab` 数据缺失而中止，未写入节点；补齐官方 NLTK 数据后原样重试成功。
- Chroma 与 DocStore 均为 240 个节点，node_id 完全一致。
- 原有 211 个节点保留，云台新增 29 个节点。
- 现有多文件入库函数在单文件追加时会用空 StorageContext 覆盖 DocStore；已从 Chroma 中恢复全部节点的正文和 metadata，Chroma 未删除或重建。
- 五份 PDF 均通过现有向量存储探测，且五份文档均通过 BM25 检索探测。
- 未删除知识库、未重建索引、未改变切片参数。

## 每类实际生成数量

| 类别 | 数量 |
|---|---:|
| 参数查询 | 10 |
| 安装操作 | 10 |
| 故障排查 | 10 |
| 跨产品型号归属辨析 | 10 |
| 跨语言检索 | 10 |

## 每份 PDF 的题目分布

| PDF | 题数 |
|---|---:|
| OBD.pdf | 8 |
| 云台.pdf | 13 |
| 小米路由器.pdf | 10 |
| 空气炸锅.pdf | 12 |
| 高清摄像头.pdf | 7 |

## 中英文题目分布

| 题目语言 | 数量 |
|---|---:|
| 中文 | 45 |
| 英文 | 5 |

跨语言检索题包含：

- 中文问题检索英文说明书 5 条：CROSS-01、CROSS-02、CROSS-03、CROSS-04、CROSS-05
- 英文问题检索中文说明书 5 条：CROSS-06、CROSS-07、CROSS-08、CROSS-09、CROSS-10

## 每道题的证据来源和 node_id 检查

| question_id | PDF | 物理页 | page_label | 原文证据 | 节点ID存在 | node_id |
|---|---|---:|---:|---|---|---|
| PARAM-01 | 空气炸锅.pdf | 6 | 6 | PASS | PASS | `89b02e4f-ff9f-4967-b6f5-54e6ba820882` |
| PARAM-02 | 空气炸锅.pdf | 8 | 8 | PASS | PASS | `99771fe1-d400-4bca-bf4c-bac2c5884c07` |
| PARAM-03 | 空气炸锅.pdf | 15 | 15 | PASS | PASS | `d7f91704-e19e-413b-afac-7b13cf19fc6b` |
| PARAM-04 | 云台.pdf | 6 | 6 | PASS | PASS | `4d2d60f4-f403-41dd-9593-c53582e50431` |
| PARAM-05 | 云台.pdf | 21 | 21 | PASS | PASS | `d3fd0c6b-85bc-4f98-8278-34e8e7af940c` |
| PARAM-06 | 云台.pdf | 21 | 21 | PASS | PASS | `d3fd0c6b-85bc-4f98-8278-34e8e7af940c` |
| PARAM-07 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `009f5888-c752-4fe0-acc3-8965a9ad3078` |
| PARAM-08 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `009f5888-c752-4fe0-acc3-8965a9ad3078` |
| PARAM-09 | 高清摄像头.pdf | 5 | 5 | PASS | PASS | `4590a2b2-6410-43a6-99be-28219f708338` |
| PARAM-10 | OBD.pdf | 3 | 2 | PASS | PASS | `97e3e4fe-5743-4a20-a3d1-c645a356a388` |
| INSTALL-01 | 空气炸锅.pdf | 8 | 8 | PASS | PASS | `a777b9c0-368b-4b12-b447-a2d05395d313` |
| INSTALL-02 | 空气炸锅.pdf | 10 | 10 | PASS | PASS | `11b8c2db-677a-4852-a2a1-8fc8e9fac450` |
| INSTALL-03 | 空气炸锅.pdf | 12 | 12 | PASS | PASS | `c839acb9-2271-482b-b422-886b33da81bd` |
| INSTALL-04 | 云台.pdf | 6 | 6 | PASS | PASS | `4d2d60f4-f403-41dd-9593-c53582e50431` |
| INSTALL-05 | 云台.pdf | 7 | 7 | PASS | PASS | `c8df40d8-c29a-4fab-a8f1-d76e01c8300c` |
| INSTALL-06 | 云台.pdf | 8 | 8 | PASS | PASS | `b6966c02-8924-4120-9506-a2590fe7d557` |
| INSTALL-07 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `009f5888-c752-4fe0-acc3-8965a9ad3078` |
| INSTALL-08 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `66e7e9cc-3be2-44dd-927e-d48bd9e2ff31` |
| INSTALL-09 | 高清摄像头.pdf | 5 | 5 | PASS | PASS | `4590a2b2-6410-43a6-99be-28219f708338` |
| INSTALL-10 | OBD.pdf | 4 | 3 | PASS | PASS | `8f265280-4b48-44de-8562-da11e6041ae4` |
| TROUBLE-01 | 空气炸锅.pdf | 20 | 20 | PASS | PASS | `01cc9fa7-6fab-46b2-9dff-73a6d7034e84, a9df3d2a-bb5e-417e-b556-03c7663c0498` |
| TROUBLE-02 | 空气炸锅.pdf | 20 | 20 | PASS | PASS | `a9df3d2a-bb5e-417e-b556-03c7663c0498` |
| TROUBLE-03 | 空气炸锅.pdf | 21 | 21 | PASS | PASS | `678f6975-6fdb-445d-aaf3-28f0dd5c3077` |
| TROUBLE-04 | 空气炸锅.pdf | 21 | 21 | PASS | PASS | `96b838fd-1db1-4728-a97e-79a3be248713` |
| TROUBLE-05 | 云台.pdf | 7 | 7 | PASS | PASS | `c8df40d8-c29a-4fab-a8f1-d76e01c8300c` |
| TROUBLE-06 | 云台.pdf | 20 | 20 | PASS | PASS | `f6e0bd3c-12ce-44bf-a465-6a7fef84d7af` |
| TROUBLE-07 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `3f051004-da70-47f4-8bbb-e82699010bf0` |
| TROUBLE-08 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `3f051004-da70-47f4-8bbb-e82699010bf0` |
| TROUBLE-09 | 高清摄像头.pdf | 7 | 7 | PASS | PASS | `fb90776c-d0f2-49d8-a3ea-cd8a14a66a81` |
| TROUBLE-10 | OBD.pdf | 7 | 6 | PASS | PASS | `b639b80f-6866-4f91-a955-ce9ef19c2f99` |
| MODEL-01 | OBD.pdf | 1,18 | 1,17 | PASS | PASS | `5ef1eb63-d1f1-4918-bd75-5f2e8f21a8b2, 782b4213-034f-436c-9f88-ee4325dd9868` |
| MODEL-02 | 高清摄像头.pdf | 1 | 1 | PASS | PASS | `1db4fabe-863d-4c14-8886-ea3c9c7d39cc` |
| MODEL-03 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `7c21911d-5320-46cc-8fba-b9e1a86f52f1` |
| MODEL-04 | 空气炸锅.pdf | 1 | 1 | PASS | PASS | `46400c15-91bb-4c4f-8859-bf09b90cfd0c` |
| MODEL-05 | 云台.pdf | 3 | 3 | PASS | PASS | `a6be8df9-077b-4876-a47f-5cd5ca6dbffd` |
| MODEL-06 | OBD.pdf | 3 | 2 | PASS | PASS | `97e3e4fe-5743-4a20-a3d1-c645a356a388` |
| MODEL-07 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `009f5888-c752-4fe0-acc3-8965a9ad3078` |
| MODEL-08 | 高清摄像头.pdf | 3 | 3 | PASS | PASS | `0deffe19-08ee-49c3-bb81-55d1a855664c` |
| MODEL-09 | 云台.pdf | 5 | 5 | PASS | PASS | `27d1be4a-a078-4d78-938f-eb8cdb3acd27` |
| MODEL-10 | 云台.pdf | 10 | 10 | PASS | PASS | `0075fe06-cfd8-4948-83a9-ee2bad80c6af` |
| CROSS-01 | OBD.pdf | 2 | 1 | PASS | PASS | `389869fb-9141-43ff-a991-223f24d8346f` |
| CROSS-02 | OBD.pdf | 6 | 5 | PASS | PASS | `c5f10482-1996-4789-8a45-ff3b5d3aec79` |
| CROSS-03 | 高清摄像头.pdf | 5 | 5 | PASS | PASS | `4590a2b2-6410-43a6-99be-28219f708338` |
| CROSS-04 | 高清摄像头.pdf | 6 | 6 | PASS | PASS | `cbb7d19e-edb4-4501-8136-3c3934365983` |
| CROSS-05 | OBD.pdf | 18 | 17 | PASS | PASS | `782b4213-034f-436c-9f88-ee4325dd9868` |
| CROSS-06 | 云台.pdf | 15 | 15 | PASS | PASS | `db052eaf-9f80-4d91-9473-70e750081279` |
| CROSS-07 | 空气炸锅.pdf | 14 | 14 | PASS | PASS | `ff02e247-456b-4a29-9392-f09c20c5e6ea` |
| CROSS-08 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `686b5cfc-10e4-4e23-875c-8eea8499732f` |
| CROSS-09 | 小米路由器.pdf | 1 | 1 | PASS | PASS | `c1c1e33c-3a2d-437e-ba04-7959807500db` |
| CROSS-10 | 云台.pdf | 21 | 21 | PASS | PASS | `3d8cd0d6-de1a-4036-87d9-b60bc0151e20` |

## 无法生成的题目及原因

无。50 条题目均能从现有 PDF 原文和现有索引节点中找到直接证据。所有答案均未引入 PDF 原文之外的事实。

## 重复问题检查

- 重复 question 文本：0 条。
- 重复 evidence 文本：0 条。
- 不同问题可共用 expected_chunk；没有为了节点唯一性修改有效题目。

## 质量边界

- “跨产品型号归属辨析”仅验证不同产品、型号、参数和功能的归属，不代表相邻型号混淆测试。
- OBD 物理页与 page_label 存在封面/印刷页码偏移，数据同时记录两者。
- MODEL-01 使用两个证据节点：物理页 1 的产品封面和物理页 18 的 LA9039 规格页。
- TROUBLE-01 的两个 expected_chunks 为重叠切片，任一命中均可。
- 自动化检查确认 evidence 文本与 node_id 存在性；答案正确率仍需人工评分。
