# 安装

## 前置条件

- KiraAI 主程序（本插件是 KiraAI 插件，不能独立运行）
- Python >= 3.11
- Pillow >= 10.0.0

## 安装步骤

### 方式一：通过 KiraAI 内建插件安装（推荐）

在 KiraAI WebUI 中进入插件管理，输入仓库地址：

```
https://github.com/CelestNya/kira-plugin-parallel-image-reader
```

### 方式二：手动复制

```bash
git clone https://github.com/CelestNya/kira-plugin-parallel-image-reader.git
cp -r kira-plugin-parallel-image-reader /path/to/kiraai/data/plugins/parallel_image_reader
```

### 启用插件

在 `data/config/plugins.json` 中添加：

```json
{
  "parallel_image_reader": true
}
```

或在 KiraAI WebUI 插件页面中点击启用。

## 验证

启动 KiraAI，观察控制台输出：

```
[parallel_vlm] [ParallelImageReader] initialized: max_concurrent=3, quality=off
```
