---
name: "fashion-auto-collect"
description: "Automatically collects fashion item images from Lolibrary/WEAR/MAYLA sites based on a theme, then generates an anime girl illustration using the collected items as reference. Supports character image injection and AI-composed photography poses. Invoke when user requests fashion-themed character generation (e.g. 甜美洛丽塔, 学院风, 夏日少女) with specific clothing requirements, or when user wants to dress a specific character in new outfits."
---

# Fashion Auto-Collect & Generate

一站式流水线：从 lolibrary / WEAR / MAYLA 等站点按主题自动搜索采集服饰素材图片（裙子、鞋子、袜子、发饰、包袋），支持注入角色参考图保持角色辨识度，通过 LLM 自动生成摄影构图/姿势描述，最后用素材+角色图作为参考图，调用 AIGC2D API 生成少女插画。

## 入口脚本

`fashion_pipeline.py` — 位于项目根目录，是 pipeline 的唯一切入点。

## 执行方式

### 基础用法

```bash
python fashion_pipeline.py \
    --site hybrid \
    --theme "甜美洛丽塔" \
    --parts dress,shoes,socks,hair_accessory,bag \
    --extra-prompt "粉发棕瞳少女，夏日花园场景" \
    --instructions "高跟凉鞋，日系少女插画风格" \
    --style "自动(主题默认)" \
    --characters 1 \
    --ratio 3:4 \
    --resolution 2K \
    --api aigc2d \
    --output-prefix sweet_lolita_summer
```

### 指定角色参考图

```bash
python fashion_pipeline.py \
    --site hybrid \
    --theme "甜美洛丽塔" \
    --parts dress,shoes,socks,hair_accessory,bag \
    --character-image "d:/nikki-comic.jpg" \
    --character-desc "粉色长发、绿色眼睛的少女，气质优雅可爱" \
    --extra-prompt "夏日花园午后场景" \
    --instructions "日系插画风格，高跟凉鞋" \
    --characters 1 \
    --ratio 3:4 \
    --resolution 2K \
    --api aigc2d \
    --output-prefix nikki_sweet_summer
```

### 跳过构图生成

```bash
python fashion_pipeline.py ... --no-composition
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--site` | hybrid | 采集站点: lolibrary / wear / mayla / hybrid |
| `--theme` | 甜美洛丽塔 | 主题名，用于匹配主题配置文件中的风格/颜色/元素标签 |
| `--brand` | (自动) | 品牌 slug，lolibrary 模式必填 |
| `--parts` | dress,shoes,socks,hair_accessory,bag | 采集部位，逗号分隔 |
| `--max-pages` | 1 | 每部位扫描页数 |
| `--extra-prompt` | "" | 额外生图 Prompt（人物/场景/氛围描述） |
| `--instructions` | "" | 系统约束（构图、镜头、质感要求） |
| `--style` | 主题默认 | 画风预设名（如 puracotte-style），留空用主题默认 |
| `--characters` | 1 | 主角人数: 1 或 2 |
| `--ratio` | 3:4 | 出图宽高比 |
| `--resolution` | 2K | 分辨率: 1K / 2K / 4K |
| `--api` | aigc2d | API 类型: aigc2d / aigc-2d-gpt |
| `--output-prefix` | fashion_pipeline | 输出文件前缀 |
| `--character-image` | "" | **角色参考图路径**（如 d:/nikki-comic.jpg） |
| `--character-desc` | "" | **角色文字描述**（配合角色图使用，如 "粉色长发、绿色眼睛的少女"） |
| `--no-composition` | false | 跳过 LLM 构图生成 |
| `--no-collect` | false | 跳过采集，从已有目录读取素材 |
| `--bundle-dir` | "" | 已有素材目录（配合 --no-collect） |

## 流水线步骤

1. **采集**: 根据 site/theme/brand/parts 调用 `FashionCollectionService.collect_bundle()` 搜索并下载服饰素材（Lolibrary 按 shoes/socks 分类定向搜索）
2. **构建 Prompt**: 根据主题、画风、场景、主角描述拼接生图提示词。若有角色图，在 prompt 前面注入角色设定，instructions 中加入 face consistency 约束
3. **LLM 摄影构图生成**: 从 `prompts/fashion-composition-system.md` 和 `prompts/fashion-composition-user.md` 加载 prompt 模板，调用 `fetch_llm_json()` 请求 LLM 设计专业摄影构图方案（构图类型、镜头角度、角色姿态、光线、景深、需避免元素），注入到 instructions 末尾
4. **生图**: 角色参考图（第一张）+ 服饰素材图作为参考图序列传入 `generate_image_aigc2d()`

## 输出

- 图片保存在 `data/YYYYMMDD/fashion-pipeline/` 目录
- Pipeline 元数据写入 `{output_prefix}_pipeline_meta.json`

## 注意事项

- 首次使用 MAYLA 站点需安装 Playwright: `pip install playwright && python -m playwright install chromium`
- lolibrary 模式优先按 `shoes`/`socks` 分类定向搜索，若无结果回退品牌全量搜索兜底
- 角色图必须是真实存在的本地文件路径
- 素材采集会随机打乱候选项顺序以增加多样性
- LLM 构图方案使用 aigc2d 配置中的 API key，通过 `/v1/chat/completions` 调用
- 构图 prompt 模板位于 `prompts/fashion-composition-system.md`（system）和 `prompts/fashion-composition-user.md`（user），使用 `{{变量}}` 占位符替换，可按需编辑迭代
