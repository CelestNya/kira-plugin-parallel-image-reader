# Parallel Image Reader — 并行识图插件

**KiraAI 插件** — 拦截 IM 消息中的图片，并发调用 VLM 描述，大幅降低多图场景的回复延迟。

## 解决的问题

KiraAI 原生按顺序处理图片：N 张图 = N 次串行 VLM 调用。多图场景下延迟叠加严重。

本插件：
- **零阻塞**拦截：`ON_IM_MESSAGE` 时将图片替换为占位符，不阻塞消息管道
- **并发读图**：`ON_LLM_REQUEST` 时 Semaphore + `asyncio.gather` 并发调用 VLM
- **缓存复用**：使用 KiraAI 内置 `image_desc_cache`，相同图片秒回

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

## 日志

控制台日志以 `[parallel_vlm]`（紫色）显示：
- `[VLM] request | image=1920x1080 | quality=85 | prompt=...`
- `[VLM] response | len=342 | 画面中是一位年轻女性...`
- `[VLM] cache HIT | md5=a1b2c3d4... | ...`
- `[Inject] __IMG__sess__0__ -> [Image: ...]`

## 提示词

单图和并发多图使用不同的 VLM 提示词（不可配置，硬编码在代码中）：
- 单图：KiraAI 内置提示
- 多图：告知 VLM 这是第几张，要求详细描述细节以免因上下文缺失产生过简描述
