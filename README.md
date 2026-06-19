# Parallel Image Reader — 并行识图插件

**KiraAI 插件** — 拦截 IM 消息中的图片，并发调用 VLM 描述后直接替换为文字，历史记录清洁无污染。

## 解决的问题

KiraAI 原生按顺序处理图片：N 张图 = N 次串行 VLM 调用。多图场景下延迟叠加严重。

本插件：
- **并发读图**：Semaphore + `asyncio.gather` 并发调用 VLM
- **零历史污染**：`ON_IM_MESSAGE` 时直接完成描述并替换为 `[Image: 描述]` 文字，写入历史的就是纯文字
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
IM 消息 → [ON_IM_MESSAGE] 遍历 chain
                            ├─ 找到 Image/Sticker → hash 查缓存
                            │                       ├─ HIT  → 直接用已有描述
                            │                       └─ MISS → Semaphore × N 次 VLM
                            │                                 ├─ 原生模式 → desc_img
                            │                                 └─ 质量模式 → JPEG 压缩 → VLM
                            │                                 ↓ 写入缓存
                            └─ chain[i] = Text("[Image: 描述]") → 写入历史已是纯文字

LLM 请求 → 无事可做，历史已清洁
```

## 日志

控制台日志以 `[parallel_vlm]`（紫色）显示：
- `[VLM] request | image=1920x1080 | quality=85 | prompt=...`
- `[VLM] response | len=342 | 画面中是一位年轻女性...`
- `[VLM] cache HIT [session] | md5=a1b2c3d4... | ...`
- `[VLM] TIMEOUT | 60s`

## 版本记录

- **v2.0.1** — 清理冗余代码、README 更新
- **v2.0.0** — 重构：VLM 移至 on_im_message 内联替换，移除 stash/__IMG__ 机制
- **v1.1.0** — 并行识别、缓存、质量压缩
