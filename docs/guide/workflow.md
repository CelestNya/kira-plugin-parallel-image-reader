# 工作流程

## 数据流

```
IM 消息 → [ON_IM_MESSAGE] 提取 Image/Sticker → 替换为占位符 → 放行
                                                          ↓
LLM 请求 → [ON_LLM_REQUEST] 取 stash → 查缓存
                                        ├─ HIT  → 直接取描述
                                        └─ MISS → Semaphore × N 次 VLM
                                                  ├─ 原生模式 → desc_img
                                                  └─ 质量模式 → JPEG 压缩 → VLM
                                        ↓
                    替换占位符为 [Image: 描述内容] → 注入 system prompt
```

## 事件优先级

| 事件 | 优先级 | 说明 |
|------|--------|------|
| `ON_IM_MESSAGE` | `SYS_HIGH - 1` = 99 | 在默认 chat 插件之前拦截，修改消息链 |
| `ON_LLM_REQUEST` | `SYS_HIGH - 1` = 99 | 在 LLM 请求前注入图片描述 |

优先级确保了：
1. 图片先被替换为占位符，默认 chat 插件看到的是纯文本
2. VLM 调用在 LLM 请求前完成，描述已就绪

## 缓存机制

缓存 key 为图片内容的 MD5 哈希，与编码方式无关：

- **第一次发送**：VLM 调用 → 计算 MD5 → 写入 `image_desc_cache`
- **第二次发送相同图片**：直接返回缓存描述，零 VLM 调用
- 缓存由 KiraAI 统一管理，支持定期清理过期条目

## 日志输出

控制台以紫色 `[parallel_vlm]` 显示：

```
[VLM] request | image=1920x1080 | quality=85 | prompt=描述这张图片...
[VLM] response | len=342 | 画面中是一位年轻女性，身穿白色上衣...
[VLM] #2/3 cache HIT | md5=a1b2c3d4... | 一位金发少女...
[VLM] #2/3 desc_img | md5=e5f6... | prompt=这是用户发送的第 2 张...
[VLM] FAILED | APITimeoutError: timeout
[Inject] __IMG__sess_0__0__ -> [Image: 画面中是一位年轻女性...]
```
