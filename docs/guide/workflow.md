# 工作流程

## 两阶段架构

本插件采用两阶段设计，核心思路是：**在消息命运确定之前只做轻量占位，确定会发给 LLM 之后才并行调 VLM**。

```
阶段1  IM 消息 → [ON_IM_MESSAGE] (优先级 SYS_HIGH-1=99, 早于 chat 插件 HIGH=50)
                    │
                    ├─ 递归遍历 chain（含 Reply.chain / Forward.chains，带环检测）
                    ├─ 对每个 Image/Sticker:
                    │   ├─ hash 查 image_desc_cache
                    │   ├─ HIT  → 替换为 Text("[Image: 描述]")（零 VLM）
                    │   └─ MISS → 替换为 Text(占位符)，原图暂存到 message._pir_pending
                    │
                    └─ 不调 VLM、不干预 event 策略

       chat 插件 (HIGH=50) 决定消息命运:
                    ├─ discard → 消息终止，零 VLM 开销 ✅
                    ├─ buffer  → 进入缓冲区等待 debounce
                    └─ flush/trigger → 立即进入 batch 处理 ↓

阶段2  批量消息 → handle_im_batch_message()
                    │
                    ├─ 本体 message_format_to_text() 已执行完毕
                    │  （占位 Text 不触发 VLM ✅）
                    │
                    ├─ [ON_IM_BATCH_MESSAGE] (优先级 SYS_HIGH-1=99)
                    │   ├─ 读取各 message._pir_pending，收集暂存图片
                    │   ├─ Semaphore + gather 并行 VLM
                    │   │   ├─ 原生模式 → desc_img
                    │   │   └─ 质量模式 → JPEG 压缩 → VLM
                    │   ├─ 写入 image_desc_cache
                    │   ├─ 替换 message_str 中的占位符 → [Image: 描述]
                    │   └─ 替换 chain 中的占位 Text → Text("[Image: 描述]")

LLM 请求 → [ON_LLM_REQUEST] (优先级 SYS_HIGH-1=99)
              └─ 注入 system hint 说明 [Image: ...] 格式含义
```

## 为什么需要两阶段

| 版本 | 方案 | 问题 |
|------|------|------|
| v1.x | ON_IM_MESSAGE 放占位符 → ON_LLM_REQUEST 才 VLM | 占位符在 ON_LLM_REQUEST 之前已持久化，泄漏到聊天历史 |
| v2.0.x | ON_IM_MESSAGE 直接 VLM 替换 | chat 插件尚未决定消息命运，discard 的消息白做了 VLM |
| **v2.1.0** | **ON_IM_MESSAGE 占位+暂存 → ON_IM_BATCH_MESSAGE 并行** | **既不污染历史，也不浪费 VLM** |

### 关键机制：Text 占位符 + message._pir_pending 暂存

阶段1把 Image/Sticker 替换为 `Text(占位符)`，占位符格式 `\x00PIR_{md5}\x00`（不可见控制字符）。
本体 `message_format_to_text` 遍历 chain 时只看到 Text 元素，直接输出其 text，**不触发 VLM**。

原图元素暂存到 `message._pir_pending`（自定义属性）。KiraAI 本体在 `flush_session_messages`
中构建 batch 时，`messages=[m.message for m in pending_messages]`——batch 里的 message 就是
`ON_IM_MESSAGE` 阶段的同一个实例，`_pir_pending` 属性得以保留。

阶段2并行 VLM 后，替换 `message_str` 中的占位符为 `[Image: 描述]`。**`message_str` 在持久化
之前被修正**（`handle_im_batch_message` 第456行生成 → `ON_IM_BATCH_MESSAGE` 第460行修正 →
第532行入 prompt → 第570行持久化），占位符不进入历史。

### 为什么不用 Image.caption 钩子

KiraAI 本体 `message_format_to_text` 对 Image 的处理：`caption is None` 时调 VLM，否则跳过。
看似可以设 caption 占位标记阻止 VLM——但 else 分支会**把 caption 当作真实描述写入缓存**
（`image_desc_cache.set(md5, ele.caption)`），导致缓存被占位标记污染。

Text 占位符方案完全绕开了这个陷阱：本体看不到 Image 元素，自然不会调 VLM 也不会写缓存。

## 事件优先级

| 事件 | 优先级 | 说明 |
|------|--------|------|
| `ON_IM_MESSAGE` | `SYS_HIGH - 1` = 99 | 早于 chat 插件(50)，标记 caption 占位 |
| `ON_IM_BATCH_MESSAGE` | `SYS_HIGH - 1` = 99 | 本体 format_to_text 之后，并行 VLM + 替换 |
| `ON_LLM_REQUEST` | `SYS_HIGH - 1` = 99 | 仅注入 system prompt 说明 `[Image: ...]` 格式 |

## 缓存机制

缓存 key 为图片内容的 MD5 哈希，与编码方式无关：

- **阶段1 缓存命中**：直接替换为 `[Image: 描述]`，无需暂存，阶段2 零 VLM
- **阶段2 缓存未命中**：并行 VLM → 写入 `image_desc_cache`
- **阶段2 双重检查**：`_describe_parallel` 内部再次查缓存（防并发刷新）
- 缓存由 KiraAI 统一管理，支持定期清理过期条目
- **占位符不污染缓存**：阶段1用 Text 替换 Image（不是设 caption），本体不会把占位符写入缓存

## 日志输出

控制台以紫色 `[parallel_vlm]` 显示：

```
[ParallelImageReader] stashed 2 pending, 1 cache-hit [qq:gm:123]
[ParallelImageReader] batch (native) [qq:gm:123]: 2 pending, concurrency=3
[VLM] #1/2 desc_img [qq:gm:123] | md5=a1b2c3d4... | prompt=描述这张图片...
[VLM] #1/2 done [qq:gm:123] | len=342 | 画面中是一位年轻女性...
[ParallelImageReader] described 2 images [qq:gm:123]
```
