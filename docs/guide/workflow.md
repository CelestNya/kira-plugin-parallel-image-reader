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
                    │   └─ MISS → 替换为 Text(<!--PIR:md5-->)，原图暂存到 message._pir_pending
                    │
                    ├─ 可选：乐观加载（eager_loading）
                    │   └─ 创建后台 task 并行 VLM（不应 await，handler 立即返回）
                    │
                    └─ 不调 VLM（除非乐观加载）、不干预 event 策略

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
                    │   ├─ 分流乐观/非乐观：
                    │   │   ├─ 有 _pir_optimistic → await 后台 task 复用结果
                    │   │   └─ 无 → 现场并行 VLM
                    │   ├─ 写入 image_desc_cache
                    │   ├─ 替换 message_str 中的占位符 → [Image: 描述]
                    │   └─ 替换 chain 中的占位 Text → Text("[Image: 描述]")
                    │
                    └─ Ctrl+C 取消时：_cleanup_placeholders 清理占位符后重新抛出

LLM 请求 → [ON_LLM_REQUEST] (优先级 SYS_HIGH-1=99)
              └─ 注入 system hint 说明 [Image: ...] 格式含义
```

## 为什么需要两阶段

| 版本 | 方案 | 问题 |
|------|------|------|
| v1.x | ON_IM_MESSAGE 放占位符 → ON_LLM_REQUEST 才 VLM | 占位符在 ON_LLM_REQUEST 之前已持久化，泄漏到聊天历史 |
| v2.0.x | ON_IM_MESSAGE 直接 VLM 替换 | chat 插件尚未决定消息命运，discard 的消息白做了 VLM |
| **v2.1.0** | **ON_IM_MESSAGE 占位+暂存 → ON_IM_BATCH_MESSAGE 并行** | **既不污染历史，也不浪费 VLM** |
| **v2.2.0** | **+ 乐观加载（可选）** | **可提前后台识图，追求更低延迟** |

### 关键机制：Text 占位符 + message._pir_pending 暂存

阶段1把 Image/Sticker 替换为 `Text("<!--PIR:md5-->")`（XML 注释样式）。
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

## 乐观加载（v2.2.0）

`eager_loading=True` 时，阶段1收到图片即启动后台 VLM task（`asyncio.create_task`）：
- **不阻塞 handler 链**：task 启动后立即返回，chat 插件照常决策
- **复用结果**：阶段2 await 该 task，已跑完则零等待，未跑完则等待
- **非乐观路径并行**：乐观 task 和现场 VLM 通过 `asyncio.gather` 两组同时推进
- **异常隔离**：task 崩溃 → `gather(return_exceptions=True)` 捕获 → 降级为 `"(description unavailable)"`
- **被 discard 的消息**：task 仍跑完并写缓存，下次同图命中复用

### 并发控制

实例级 `asyncio.Semaphore` 替代之前的局部 Semaphore，乐观 task 和批量调用共享全局并发度，
防止多消息同时到达时 VLM 并发超限。

### 生命周期管理

- `_optimistic_tasks` 集合持有 task 强引用（防 GC），`add_done_callback` 自动移除
- `terminate()` 取消所有未完成 task，插件卸载无泄漏

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
- `_is_valid_desc` 防御：含 `\x00` 或 `<!--PIR:` 的缓存视为无效，跳过命中

## Ctrl+C 取消保护

`on_im_batch_message` 同时捕获 `asyncio.CancelledError`，先执行 `_cleanup_placeholders` 清除
残留在 message_str 中的占位符，再重新抛出。确保 Ctrl+C 退出时也不会把 `<!--PIR:...-->` 带入历史。

## 日志输出

控制台以紫色 `[parallel_vlm]` 显示：

```
[ParallelImageReader] stashed 2 pending, 1 cache-hit [qq:gm:123]
[ParallelImageReader] batch (native) [qq:gm:123]: 2 pending, concurrency=3
[VLM] #1/2 desc_img [qq:gm:123] | md5=a1b2c3d4... | prompt=描述这张图片...
[VLM] #1/2 done [qq:gm:123] | len=342 | 画面中是一位年轻女性...
[ParallelImageReader] described 2 images [qq:gm:123]
```
