# 工作流程

## 数据流

```
IM 消息 → [ON_IM_MESSAGE] 遍历 chain
                            ├─ 找到 Image/Sticker → hash 查缓存
                            │                       ├─ HIT  → 直接用已有描述
                            │                       └─ MISS → Semaphore × N 次 VLM
                            │                                 ├─ 原生模式 → desc_img
                            │                                 └─ 质量模式 → JPEG 压缩 → VLM
                            │                                 ↓ 写入缓存
                            └─ chain[i] = Text("[Image: 描述]") → 写入历史已是纯文字

LLM 请求 → [ON_LLM_REQUEST] 无害通过，历史已清洁
```

## 事件优先级

| 事件 | 优先级 | 说明 |
|------|--------|------|
| `ON_IM_MESSAGE` | `SYS_HIGH - 1` = 99 | 在默认 chat 插件之前拦截，修改消息链 |
| `ON_LLM_REQUEST` | `SYS_HIGH - 1` = 99 | 仅注入 system prompt 说明 `[Image: ...]` 格式 |

优先级确保了图片在到达历史之前就已经被替换为文字描述。

## 缓存机制

缓存 key 为图片内容的 MD5 哈希，与编码方式无关：

- **第一次发送**：VLM 调用 → 计算 MD5 → 写入 `image_desc_cache`
- **第二次发送相同图片**：直接返回缓存描述，零 VLM 调用
- 缓存由 KiraAI 统一管理，支持定期清理过期条目

## 日志输出

控制台以紫色 `[parallel_vlm]` 显示：

```
[VLM] #2/3 desc_img [qq:gm:123] | md5=a1b2c3d4... | prompt=描述这张图片...
[VLM] #2/3 cache HIT [qq:gm:123] | md5=e5f6... | 一位金发少女...
[VLM] response | len=342 | 画面中是一位年轻女性，身穿白色上衣...
[VLM] TIMEOUT | 60s
[ParallelImageReader] described 3 images [qq:gm:123]
```
