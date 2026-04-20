# image-maker 本地 booru tagger 配置说明

> 开发/改动代码前，建议先阅读项目根目录的 `AGENTS.md`（模块索引与改动入口说明）。

## 安装说明

1. 安装 Python 依赖：

```bash
pip install numpy onnxruntime
```

2. 下载 WD14 模型文件（任选一个系列，如 ConvNextV2）：
   - `model.onnx`
   - `selected_tags.csv`

3. 将模型放到以下任一位置：
   - `data/models/wd14/model.onnx`
   - `models/wd14/model.onnx`
   - `wd14_tagger_model/model.onnx`

   对应的标签文件放到同目录下的 `selected_tags.csv`。

4. `booru tags` 的词表过滤会读取 `config-autocomplete.json` 中的 `csv_path`，默认是：
   - `data/tags/danbooru.csv`

## config-autocomplete.json 配置项（中文）

- `local_booru_tagger_model_path`
  - 本地 WD14 模型 `model.onnx` 路径，支持相对路径和绝对路径。
  - 为空时按内置候选路径自动查找。

- `local_booru_tagger_tags_path`
  - 本地 WD14 标签定义文件 `selected_tags.csv` 路径。
  - 为空时按内置候选路径自动查找。

- `local_booru_tagger_max_tags`
  - WD14 推理后最多保留的候选 tag 数量。

- `local_booru_tagger_general_threshold`
  - General 类标签阈值（0~1）。

- `local_booru_tagger_character_threshold`
  - Character 类标签阈值（0~1）。

- `local_booru_tagger_meta_threshold`
  - Meta 类标签阈值（0~1）。

- `local_booru_tagger_rating_threshold`
  - Rating 类标签阈值（0~1）。

- `local_booru_tagger_keep_rating_tags`
  - 是否在最终候选中保留 rating 类标签（`true/false`）。
  - `false` 时即使分数达到阈值也会过滤掉 rating 标签。

- `local_booru_tagger_use_autocomplete_filter`
  - 是否使用 `csv_path` 指向的 danbooru 词表做二次过滤（`true/false`）。

## 示例配置

```json
{
  "enable_autocomplete": true,
  "csv_path": "data/tags/danbooru.csv",
  "max_results": 50,
  "min_chars": 2,
  "local_booru_tagger_model_path": "data/models/wd14/model.onnx",
  "local_booru_tagger_tags_path": "data/models/wd14/selected_tags.csv",
  "local_booru_tagger_max_tags": 60,
  "local_booru_tagger_general_threshold": 0.35,
  "local_booru_tagger_character_threshold": 0.35,
  "local_booru_tagger_meta_threshold": 0.75,
  "local_booru_tagger_rating_threshold": 0.75,
  "local_booru_tagger_keep_rating_tags": false,
  "local_booru_tagger_use_autocomplete_filter": true
}
```
