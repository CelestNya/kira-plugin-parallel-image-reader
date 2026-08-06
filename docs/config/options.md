# 配置选项

在 KiraAI WebUI 插件页面中配置。

## load_mode

- **类型**: `enum`（单选）
- **选项**: `lazy` / `eager` / `llm_select`
- **默认**: `lazy`

图片描述加载方式，三态互斥：

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `lazy`（懒加载） | 触发时才并行识图，被忽略的消息零 VLM 调用 | 默认，最省 |
| `eager`（乐观加载） | 收到图片即后台识图，触发时结果已就绪 | 追求低延迟，VLM 调用量增加 |
| `llm_select`（LLM 选择性加载） | 只放标识符 `[Image #id: (未识别)]`，LLM 按需调 `describe_image` 工具 | 最省 VLM，图片按需加载 |

切换模式后，历史中的旧标识符会在 LLM 请求时自动扫描替换（三模式共享缓存，无缝换态）。

## max_concurrent

- **类型**: `integer`
- **默认**: `3`
- **范围**: 1 及以上

最大并发 VLM 调用数。控制同时发起的图片描述请求数量（三模式通用）。

值越小对 API 限流越友好，值越高延迟越低。

## quality_enabled

- **类型**: `switch`
- **默认**: `false`

启用 JPEG 压缩后再送 VLM（三模式通用）。

- **关闭（默认）**：使用 KiraAI 原生 `desc_img` 路径，图片以其原始编码方式发送，画质无损。
- **开启**：将图片转为 JPEG（由 `quality_value` 控制质量）后发送，payload 更小，上传更快。

## quality_value

- **类型**: `integer`
- **默认**: `85`
- **范围**: 10–100

JPEG 压缩质量，仅在 `quality_enabled` 开启时生效。

- **100**: 最高画质，文件较大
- **85**: 画质与文件大小的良好平衡
- **50**: 文件小，有明显压缩痕迹

## forward_max_depth

- **类型**: `integer`
- **默认**: `1`

合并转发（Forward）嵌套展开的最大层数。**默认 1 = 只读第一层**（与 KiraAI 原生行为一致）；调大可展开深层嵌套。上游 forward 读取更新前建议保持默认（向后兼容）。

用途：KiraAI 核心渲染转发时只渲染第 1 层，嵌套层内容会被过滤（插件在阶段1 拍平展开后才不丢）——本配置控制展开到第几层。调小可节省 token 并防御恶意超深嵌套（递归崩溃）。

---

## auto_read_config（自动读取）

折叠分组，`load_mode=llm_select` 时生效：这些场景图片明确是给 bot 看的（无 LLM 决策空间），直接读取，不等 LLM 决定。

### private_single_auto_read

- **类型**: `switch`
- **默认**: `true`

私聊对话中只出现一张图时直接读取（不等待 LLM 调工具）。

### mention_reply_auto_read

- **类型**: `switch`
- **默认**: `true`

被 @ 提及，或引用（Reply）含图消息时，直接读取图片。

---

## id_map_limit

- **类型**: `integer`
- **默认**: `1000`

标识符映射表（short_id → 完整 md5）的最大条数，超限按写入顺序淘汰最早条目（FIFO）。

映射表用于：换态后历史中的标识符反查缓存，以及 `describe_image` 工具在历史回合加载图片描述。
