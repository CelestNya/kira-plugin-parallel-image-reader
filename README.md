# Parallel Image Reader — 并行识图插件

**KiraAI 插件** — 两阶段并行识图，既不污染聊天历史，也不浪费 VLM 调用。

## 解决的问题

KiraAI 原生按顺序处理图片：N 张图 = N 次串行 VLM 调用。多图场景下延迟叠加严重。

此前版本（v2.0.x）在 `ON_IM_MESSAGE` 阶段就调 VLM——但此时 chat 插件尚未决定消息命运，被 `discard` 的图片白白浪费了 VLM 调用。

本插件（v2.1.0）采用**两阶段架构**：
- **并发读图**：Semaphore + `asyncio.gather` 并发调用 VLM
- **零历史污染**：图片替换为 `[Image: 描述]` 文字后写入历史，纯文本无占位符
- **零 VLM 浪费**：仅在 `ON_IM_BATCH_MESSAGE`（确定会发给 LLM）时才调 VLM，被 `discard` 的消息零开销
- **阻止本体串行**：`ON_IM_MESSAGE` 阶段给 Image 设 caption 占位标记，让本体 `message_format_to_text` 跳过 VLM
- **缓存复用**：使用 KiraAI 内置 `image_desc_cache`，相同图片秒回
- **VLM 超时保护**：单次调用超时 60 秒，超时自动降级

## 安装

```bash
# 通过 KiraAI 内建插件安装功能安装
# 仓库地址: https://github.com/CelestNya/kira-plugin-parallel-image-reader

# 或者手动复制
git clone https://github.com/CelestNya/kira-plugin-parallel-image-reader.git
cp -r kira-plugin-parallel-image-reader /path/to/kiraai/data/plugins/parallel_image_reader

# 在 data/config/plugins.json 中启用
# {"parallel_image_reader": true}
```

> **依赖：** KiraAI 主程序（本插件是 KiraAI 插件，不能独立运行）

## 配置

在 KiraAI WebUI 插件页面配置：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `max_concurrent` | integer | 3 | 最大并发 VLM 调用数 |
| `quality_enabled` | switch | false | 启用 JPEG 压缩后再送 VLM |
| `quality_value` | integer | 85 | JPEG 压缩质量 (10-100) |

### 质量模式

- **关闭（默认）**：使用 KiraAI 原生 `desc_img` 路径，保持与原行为一致
- **开启**：将图片转为 JPEG（指定 quality）后发送。payload 更小，上传更快，但画质有损

## 工作流程

```
阶段1  IM 消息 → [ON_IM_MESSAGE] (优先级 SYS_HIGH-1, 早于 chat 插件)
                    ├─ 递归遍历 chain（含 Reply/Forward 嵌套，带环检测）
                    ├─ Image/Sticker → hash 查缓存
                    │                 ├─ HIT  → 替换为 Text("[Image: 描述]")
                    │                 └─ MISS → 替换为 Text(占位符)，原图暂存到 message._pir_pending
                    └─ 不调 VLM、不干预 event 策略

       chat 插件决定 discard / buffer / flush
                    ├─ discard → 消息终止，零 VLM 开销 ✅
                    └─ buffer/flush → 进入 batch 处理 ↓

阶段2  批量消息 → [ON_IM_BATCH_MESSAGE] (优先级 SYS_HIGH-1)
                    ├─ 本体 message_format_to_text 已执行（占位 Text 不触发 VLM）
                    ├─ 读取各 message._pir_pending，收集所有暂存图片
                    ├─ Semaphore + gather 并行 VLM（原生模式 / 质量模式）
                    ├─ 替换 message_str 中的占位符 → [Image: 描述]（持久化前，不进历史）
                    └─ 替换 chain 中的占位 Text → Text("[Image: 描述]")

LLM 请求 → [ON_LLM_REQUEST] 注入 system hint 说明 [Image: ...] 格式
```

## 日志

控制台日志以 `[parallel_vlm]`（紫色）显示：
- `[VLM] request | image=1920x1080 | quality=85 | prompt=...`
- `[VLM] response | len=342 | 画面中是一位年轻女性...`
- `[VLM] cache HIT [session] | md5=a1b2c3d4... | ...`
- `[VLM] TIMEOUT | 60s`

## 版本记录

- **v2.1.0** — 两阶段架构：ON_IM_MESSAGE 标记占位 + ON_IM_BATCH_MESSAGE 并行填充，discard 消息零 VLM
- **v2.0.1** — 清理冗余代码、README 更新
- **v2.0.0** — 重构：VLM 移至 on_im_message 内联替换，移除 stash/__IMG__ 机制
- **v1.1.0** — 并行识别、缓存、质量压缩
