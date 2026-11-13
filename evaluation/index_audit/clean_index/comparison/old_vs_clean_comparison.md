# Old vs Clean Index Node Comparison

## Conclusion

- 旧索引节点数：240
- 新索引节点数：240
- node_id 交集：0
- 旧索引独有 ID：240
- 新索引独有 ID：240
- 正文精确 SHA-256 交集：233
- 旧索引独有正文 SHA-256：7
- 新索引独有正文 SHA-256：7
- 可确定性配对节点：240
- 配对正文完全相同：233
- 配对正文发生变化：7
- 切片边界发生变化：7
- 节点集合数量与文档页覆盖一致，但两次构建不是逐 node 的完全相同快照。

结论：两次构建产生了相同的节点总数、相同的 PDF 节点数和物理页覆盖，
但 node_id 全部重新生成；233 个节点的正文逐字相同，7 个节点的正文因切片
边界调整而改变。因此应视为“结构对齐的新构建”，而不是可复用旧 node_id
的完全相同索引。

## Classification

| 分类 | 数量 |
|---|---:|
| `identical` | 0 |
| `metadata_only_changed` | 0 |
| `content_changed` | 7 |
| `id_changed_but_content_same` | 233 |
| `new_node` | 0 |
| `missing_node` | 0 |
| `unmatched` | 0 |

## Matching policy

1. 先按相同 node_id 配对。
2. 对同一物理页且两边 chunk 数相同的节点，按 `(chunk_start_char, chunk_end_char)`
   排序后的 chunk 序号一一配对。
3. 对无法按序号配对的剩余节点，只接受同页且起止字符完全相同的边界。
4. 最后只接受在两边都唯一的精确正文 SHA-256。
5. 不使用文本相似度阈值强行匹配。

## Document and page coverage

| 文档 | 旧节点 | 新节点 | 旧有节点页 | 新有节点页 | ID 变正文同 | 正文变化 | 边界变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OBD.pdf | 24 | 24 | 20 | 20 | 24 | 0 | 0 |
| 云台.pdf | 29 | 29 | 22 | 22 | 27 | 2 | 2 |
| 小米路由器.pdf | 12 | 12 | 1 | 1 | 11 | 1 | 1 |
| 空气炸锅.pdf | 34 | 34 | 22 | 22 | 34 | 0 | 0 |
| 高清摄像头.pdf | 141 | 141 | 116 | 116 | 137 | 4 | 4 |

所有 PDF 的节点数和有节点物理页集合均一致。各页 chunk 数量也一致；没有
`new_node` 或 `missing_node`。

## Body and slice-boundary differences

精确正文相同的配对：233。

发生正文变化且切片边界变化的节点：

| 文档 | 页 | 旧边界 | 新边界 | 旧字符数 | 新字符数 |
|---|---:|---:|---:|---:|---:|
| 云台.pdf | 5 | `0-468` | `0-478` | 468 | 478 |
| 云台.pdf | 5 | `410-604` | `429-604` | 194 | 175 |
| 小米路由器.pdf | 1 | `3559-4026` | `3559-4068` | 467 | 509 |
| 高清摄像头.pdf | 15 | `2-1544` | `2-1694` | 1542 | 1692 |
| 高清摄像头.pdf | 15 | `1344-1971` | `1545-1971` | 627 | 426 |
| 高清摄像头.pdf | 87 | `0-1583` | `0-1653` | 1583 | 1653 |
| 高清摄像头.pdf | 87 | `1405-1791` | `1528-1791` | 386 | 263 |

这 7 个节点的来源文档、物理页和 page_label 均未改变。变化原因是实际切片
边界随 metadata-aware chunking 的 metadata 内容调整，不是 PDF 源文件变化。

## Metadata comparison

| 指标 | 结果 |
|---|---:|
| 完整语义 metadata 相同的配对 | 0/240 |
| 10 个旧 canonical source 字段完整相同 | 0/240 |
| 排除路径和上传时间的稳定来源字段相同 | 240/240 |
| filename 相同 | 240/240 |
| 原始 document_id 字符串相同 | 0/240 |
| 规范化 document_id 相同 | 240/240 |
| physical_page 相同 | 240/240 |
| page_label 相同 | 240/240 |

逐字段差异：

| 字段 | 差异节点数 |
|---|---:|
| `doc_id` | 240 |
| `document_id` | 240 |
| `file_path` | 240 |
| `ref_doc_id` | 240 |
| `uploaded_at` | 240 |

主要差异是：

- 旧索引 `file_path` 使用 Windows 分隔符，云台部分节点为绝对路径；新索引统一为
  项目相对 POSIX 路径。
- 旧索引 `uploaded_at` 为浮点时间戳；新索引为 ISO 字符串。
- 新索引增加 `file_sha256`、`source_page`、`page_index`。
- 旧、新 `document_id` 的原始字符串均体现物理页，但旧值通常带
  `data\uploads\` 前缀，新值直接使用 `<stored_filename>_part_<N-1>`。
- `file_path` 和 `uploaded_at` 的变化导致 10 个旧 canonical source 字段整体
  不相同；排除二者后，其余稳定来源字段 240/240 一致。

完整字段值保存在 `node_diff.csv` 的 `old_metadata_json` 与
`new_metadata_json` 列，逐字段差异保存在 `metadata_differences_json`。
`_node_content` 和 `_node_type` 属于存储层内部序列化字段，不参与语义 metadata
判定。

## Short-content correspondence

新索引 15 个短内容节点中，15 个在旧索引中存在确定性对应；
其中正文相同 15 个。

| 新短节点 | 文档 | 页 | 旧对应节点 | 正文相同 | 旧短候选 |
|---|---|---:|---|---|---|
| `0c633190-24f9-41ab-96f9-a98bdc3844b6` | 高清摄像头.pdf | 107 | `8e666f28-b6f1-4f5f-92a5-57c2029bc5b3` | True | True |
| `23fe1cf9-5ce1-4d61-b687-df3538fb66c0` | 高清摄像头.pdf | 110 | `07e989a5-2502-4e95-b8c7-21fcec0001a3` | True | True |
| `310d75d6-98bd-484e-baf8-22a18b21afae` | 空气炸锅.pdf | 1 | `46400c15-91bb-4c4f-8859-bf09b90cfd0c` | True | True |
| `49167b8f-06ac-4878-b80a-a9f6a4200893` | 高清摄像头.pdf | 109 | `b34e69e7-ee26-44ff-98fd-13725a417a14` | True | True |
| `51e08a1e-6967-4969-aacd-7dc21b5f4b18` | 高清摄像头.pdf | 112 | `a04c9c0e-16c8-411e-a3d9-ecc97b0504bd` | True | True |
| `72c99279-3f18-4e39-9bbd-1660a2b42999` | 空气炸锅.pdf | 7 | `8bc1c4a7-9fb4-46ba-bc25-d964bcbc388e` | True | True |
| `76793c81-7932-4052-9bb3-b03da0ef803b` | 空气炸锅.pdf | 22 | `3badff71-f38d-4c52-b626-bc809a674875` | True | False |
| `b3b17f5b-ad64-451c-8101-5ab3237e7ff5` | 空气炸锅.pdf | 17 | `7bee0196-1b3f-42e1-8d62-3f1adbb8f9d8` | True | True |
| `be20761a-557b-4872-aa26-53a531e3095e` | OBD.pdf | 1 | `5ef1eb63-d1f1-4918-bd75-5f2e8f21a8b2` | True | False |
| `ccb04cd0-5352-4c05-8da4-1177595c3011` | 高清摄像头.pdf | 111 | `7185bf94-c68b-484b-8f0b-3e1d9d3c7e1b` | True | False |
| `daab2b84-e25f-4fed-bcfd-8fd278b9d6b9` | 高清摄像头.pdf | 113 | `7309217d-0078-4eaa-9872-a230860b48a0` | True | True |
| `daefc31a-246c-4d45-a8c6-6e9d1b9ff8b6` | 高清摄像头.pdf | 114 | `acf4a586-4ce1-4e7c-807d-d155d4c5d669` | True | True |
| `eadba531-1712-4003-b063-bf5aedc028c2` | 高清摄像头.pdf | 108 | `fd528915-b687-4d8e-b150-0435feb16aae` | True | True |
| `f342ae73-e02e-4511-8cbb-8d4a52fdd93c` | 高清摄像头.pdf | 1 | `1db4fabe-863d-4c14-8886-ea3c9c7d39cc` | True | False |
| `f88e35b1-bb50-4c39-8b3b-6a6ecf90f558` | 云台.pdf | 1 | `ef2c351e-b957-4125-aaf4-d72db26a5f35` | True | True |

旧索引 14 个短内容候选中，14 个在新索引中存在确定性对应；
其中正文相同 14 个。

| 旧短候选 | 文档 | 页 | 新对应节点 | 正文相同 | 新短候选 |
|---|---|---:|---|---|---|
| `07e989a5-2502-4e95-b8c7-21fcec0001a3` | 高清摄像头.pdf | 110 | `23fe1cf9-5ce1-4d61-b687-df3538fb66c0` | True | True |
| `3d8cd0d6-de1a-4036-87d9-b60bc0151e20` | 云台.pdf | 21 | `dd2e9475-3d8a-4d02-9772-e5dd748af3eb` | True | False |
| `46400c15-91bb-4c4f-8859-bf09b90cfd0c` | 空气炸锅.pdf | 1 | `310d75d6-98bd-484e-baf8-22a18b21afae` | True | True |
| `7309217d-0078-4eaa-9872-a230860b48a0` | 高清摄像头.pdf | 113 | `daab2b84-e25f-4fed-bcfd-8fd278b9d6b9` | True | True |
| `7bee0196-1b3f-42e1-8d62-3f1adbb8f9d8` | 空气炸锅.pdf | 17 | `b3b17f5b-ad64-451c-8101-5ab3237e7ff5` | True | True |
| `8bc1c4a7-9fb4-46ba-bc25-d964bcbc388e` | 空气炸锅.pdf | 7 | `72c99279-3f18-4e39-9bbd-1660a2b42999` | True | True |
| `8e666f28-b6f1-4f5f-92a5-57c2029bc5b3` | 高清摄像头.pdf | 107 | `0c633190-24f9-41ab-96f9-a98bdc3844b6` | True | True |
| `a04c9c0e-16c8-411e-a3d9-ecc97b0504bd` | 高清摄像头.pdf | 112 | `51e08a1e-6967-4969-aacd-7dc21b5f4b18` | True | True |
| `acf4a586-4ce1-4e7c-807d-d155d4c5d669` | 高清摄像头.pdf | 114 | `daefc31a-246c-4d45-a8c6-6e9d1b9ff8b6` | True | True |
| `b34e69e7-ee26-44ff-98fd-13725a417a14` | 高清摄像头.pdf | 109 | `49167b8f-06ac-4878-b80a-a9f6a4200893` | True | True |
| `b6966c02-8924-4120-9506-a2590fe7d557` | 云台.pdf | 8 | `c2ce4433-59d5-4ce6-8f50-dc3ba6a63dc8` | True | False |
| `ef2c351e-b957-4125-aaf4-d72db26a5f35` | 云台.pdf | 1 | `f88e35b1-bb50-4c39-8b3b-6a6ecf90f558` | True | True |
| `f501e99c-e90e-49fa-822b-f14474c336ce` | 空气炸锅.pdf | 2 | `87a2444e-2be4-4952-b7cb-b11300d0f3c2` | True | False |
| `fd528915-b687-4d8e-b150-0435feb16aae` | 高清摄像头.pdf | 108 | `eadba531-1712-4003-b063-bf5aedc028c2` | True | True |

短节点的差异来自正文边界变化及“旧 14 个启发式短候选”和“新 15 个
长度为 80 字符以下的短节点”的统计口径差异；没有证据表明存在重复节点。

## Source-index consistency

| Index | Chroma | DocStore | CSV | ID 集合一致 | 正文一致 | CSV 规范化哈希一致 |
|---|---:|---:|---:|---|---|---:|
| old | 240 | 240 | 240 | True | 240/240 | 240/240 |
| clean | 240 | 240 | 240 | True | 240/240 | 240/240 |

## Limits

- 本任务只比较逻辑节点、正文、metadata 和切片结构。
- 未调用 Embedding、Reranker 或 LLM，未重新入库，未执行检索评测。
- Chroma 物理文件哈希不作为逻辑变化证据；SQLite housekeeping 可能改变物理
  文件哈希。
- 所有四份结果文件均写入 `comparison/`；未修改旧/新索引、PDF、旧审计报告
  或新索引 Manifest。
