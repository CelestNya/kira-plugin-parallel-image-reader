# 配置选项

在 KiraAI WebUI 插件页面中配置。

## max_concurrent

- **类型**: `integer`
- **默认**: `3`
- **范围**: 1 及以上

最大并发 VLM 调用数。控制同时发起的图片描述请求数量。

值越小对 API 限流越友好，值越高延迟越低。

## quality_enabled

- **类型**: `switch`
- **默认**: `false`

启用 JPEG 压缩后再送 VLM。

- **关闭（默认）**：使用 KiraAI 原生 `desc_img` 路径，
  图片以其原始编码方式发送，画质无损。
- **开启**：将图片转为 JPEG（由 `quality_value` 控制质量）后发送。
  payload 更小，上传更快，适合图片体积较大或网络较慢的场景。

## quality_value

- **类型**: `integer`
- **默认**: `85`
- **范围**: 10–100

JPEG 压缩质量，仅在 `quality_enabled` 开启时生效。

- **100**: 最高画质，文件较大
- **85**: 画质与文件大小的良好平衡
- **50**: 文件小，有明显压缩痕迹
