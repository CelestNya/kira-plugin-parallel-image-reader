# Parallel Image Reader

**KiraAI 并行识图插件** — 拦截 IM 消息中的图片，并发调用 VLM 描述，
大幅降低多图场景的回复延迟。

::: tip 解决的问题
KiraAI 原生按顺序处理图片：N 张图 = N 次串行 VLM 调用。多图场景下延迟叠加严重。
本插件将其改造为并发调用，延迟从 `sum(t_i)` 降低为 `max(t_i)`。
:::

## 特性

- **零阻塞拦截** — `ON_IM_MESSAGE` 时将图片替换为占位符，不阻塞消息管道
- **并发读图** — `ON_LLM_REQUEST` 时 `Semaphore` + `asyncio.gather` 并发调用 VLM
- **缓存复用** — 使用 KiraAI 内置 `image_desc_cache`，相同图片秒回
- **质量调节** — 可选 JPEG 压缩后送 VLM，减小 payload 降低延迟
- **日志完整** — 每条 VLM 请求/响应都打印到控制台，紫色高亮

## 快速开始

```bash
pip install -r requirements.txt
```

将插件目录复制到 KiraAI `data/plugins/` 下，在 WebUI 中启用即可。
