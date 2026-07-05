# Parallel Image Reader

**KiraAI 并行识图插件** — 两阶段架构，既不污染聊天历史，也不浪费 VLM 调用。

::: tip 解决的问题
KiraAI 原生按顺序处理图片：N 张图 = N 次串行 VLM 调用。多图场景下延迟叠加严重。
本插件将其改造为并发调用，延迟从 `sum(t_i)` 降低为 `max(t_i)`。
:::

::: warning v2.1.0 两阶段架构
此前版本在 `ON_IM_MESSAGE` 阶段就调 VLM——但此时 chat 插件尚未决定消息命运，被
`discard` 的图片白白浪费了 VLM 调用。v2.1.0 改为：阶段1仅标记占位（阻止本体串行 VLM），
阶段2在 `ON_IM_BATCH_MESSAGE`（确定会发给 LLM）时才并行描述。
:::

## 特性

- **两阶段架构** — 阶段1标记占位阻止本体串行 VLM，阶段2并行描述并替换
- **零历史污染** — 图片替换为 `[Image: 描述]` 文字后写入历史，纯文本无占位符
- **零 VLM 浪费** — 仅在确定发给 LLM 时才调 VLM，被 `discard` 的消息零开销
- **并发读图** — `Semaphore` + `asyncio.gather` 并发调用 VLM
- **缓存复用** — 使用 KiraAI 内置 `image_desc_cache`，相同图片秒回
- **VLM 超时保护** — 单次调用超时 60 秒，超时自动降级
- **质量调节** — 可选 JPEG 压缩后送 VLM，减小 payload 降低延迟
- **日志完整** — 每条 VLM 请求/响应都打印到控制台，紫色高亮

## 快速开始

```bash
pip install -r requirements.txt
```

将插件目录复制到 KiraAI `data/plugins/` 下，在 WebUI 中启用即可。
