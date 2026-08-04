# Parallel Image Reader — 并行识图插件

**KiraAI 插件** — 并发识别图片内容，大幅降低多图场景回复延迟。

图片越多效果越明显：N 张图的延迟从「串行累加」降为「最慢的一张」。

## 特性

- **并发识图** — 多张图片同时识别，不等前一张开完再开下一张
- **零历史污染** — 图片描述仅以文字形式写入聊天记录，不会残留任何标记
- **零浪费** — 被忽略的消息不发 VLM，不烧额度
- **缓存复用** — 同一张图再次出现秒回，不用重复识别
- **超时降级** — 单张图超时自动跳过，不影响其他图片和回复
- **质量调节** — 可选压缩后送识图，适合图片大/网络慢的场景
- **三种加载模式**（`load_mode` 切换，运行时换态无缝）：
  - **懒加载**（默认）— 触发时才识图，被忽略的消息零消耗
  - **乐观加载** — 收到即后台识图，更低延迟但消耗更多
  - **LLM 选择性加载** — 只放标识符 `[Image #id: ]`，LLM 按需调 `describe_image` 工具识图，最省 VLM

## 安装

```bash
# 仓库地址
https://github.com/CelestNya/kira-plugin-parallel-image-reader

# 或手动复制
git clone https://github.com/CelestNya/kira-plugin-parallel-image-reader.git
cp -r kira-plugin-parallel-image-reader /path/to/kiraai/data/plugins/parallel_image_reader
```

> 依赖 KiraAI 主程序，不能独立运行。

## 配置

在 KiraAI WebUI 插件页面配置：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `load_mode` | 枚举 | lazy | 加载模式：lazy（懒加载）/ eager（乐观加载）/ llm_select（LLM 选择性加载） |
| `max_concurrent` | 整数 | 3 | 同时识别的图片数，越大越快但消耗越高 |
| `quality_enabled` | 开关 | 关 | 压缩后送识图，减少上传体积 |
| `quality_value` | 整数 | 85 | 压缩质量 (10-100)，开启压缩时生效 |
| `llm_select_config.id_map_limit` | 整数 | 1000 | 标识符映射表上限（llm_select 模式） |

## 版本记录

- **v2.3.0** — 三种加载模式（load_mode 三态），新增 LLM 选择性加载（describe_image 工具）；三模式统一标识符格式 `[Image #id: ...]`；换态时历史标识符自动扫描替换
- **v2.2.0** — 新增乐观加载开关（默认关），收到图片立即后台识图；实例级并发控制；增强 Ctrl+C 取消保护
- **v2.1.0** — 两阶段架构重构：等确定发给 LLM 后才调用识图，被忽略的消息零消耗
- **v2.0.1** — 文档更新、代码清理
- **v2.0.0** — 重构：移除旧版 stash/标记机制
- **v1.1.0** — 并行识别、缓存、质量压缩
