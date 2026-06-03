# 服饰采集与少女生图策略基线

本文档用于记录当前服饰采集与少女生图链路的可复现基线，作为后续新增 `theme`、`style`、`hybrid` 混采策略前的留档。

目标：

- 让当前实现可以被稳定复现
- 让后续修改可以明确知道该改哪里
- 让“当前已实现”和“计划新增”严格分层，避免混淆

## 1. 当前能力范围

当前已经打通的流程：

1. 从站点采集 `dress / shoes / socks / hair_accessory / bag`
2. 下载素材图到本地
3. 输出 `collection_bundle.json`
4. 可选导出 `cache/last_state.json` 给 `make-pic.py`
5. 调用 `aigc2d` 接口生成 1 张少女图
6. 将最终图片保存到 `data/<日期>/fashion-generate/`

当前入口脚本：

- 根目录 CLI：`fashion-generate.py`
- 根目录抓取工具：`web-probe.py`
- GUI 入口：`modules/fashion_collection/collector_tab.py`

## 2. 当前实现结构

### 2.1 站点适配层

- `modules/fashion_collection/lolibrary_adapter.py`
  - 负责解析 Lolibrary 搜索页
  - 负责解析 Lolibrary 单品详情页
- `modules/fashion_collection/wear_adapter.py`
  - 负责解析 WEAR 分类页
  - 负责从 WEAR 详情页 `__NEXT_DATA__` 中提取 `coordinateItems`

### 2.2 服务层

- `modules/fashion_collection/collection_service.py`
  - 对外统一提供 `collect_bundle()`
  - 根据 `site_key` 分发到 Lolibrary 或 WEAR
  - 负责按部位选中单品并下载图片

### 2.3 数据结构

- `modules/fashion_collection/models.py`
  - `CatalogItem`: 站点单品
  - `CollectedAsset`: 已下载素材
  - `CollectionBundle`: 一次采集结果

### 2.4 导出桥接

- `modules/fashion_collection/make_pic_bridge.py`
  - 导出 `collection_bundle.json`
  - 导出 `cache/last_state.json`
  - 当前槽位映射：
    - `dress -> 衣服1`
    - `shoes -> 鞋子`
    - `socks -> 袜子`
    - `hair_accessory -> 发饰`
    - `bag -> 手持物`

### 2.5 生图调用

- `fashion-generate.py`
  - 拼装当前采集结果
  - 调用 `modules/others/api_backend.py` 中的 `generate_image_aigc2d()`

## 3. 当前站点策略

### 3.1 Lolibrary

用途定位：

- 更适合抓主服装
- 侧重品牌库、款式库、裙装信息

当前搜索 URL 规则：

- `https://lolibrary.org/search?brands[]={brand_slug}&page={page}`

当前解析逻辑：

1. 从搜索页解析卡片
2. 进入详情页解析标题、品牌、分类、图片、notes、tags
3. 使用关键词推断部位

当前部位关键词：

- `dress`: `jsk`, `op`, `jumperskirt`, `one piece`, `onepiece`, `dress`, `sets`
- `shoes`: `shoes`, `shoe`, `heels`, `boots`, `pumps`
- `socks`: `socks`, `sock`, `otks`, `utks`, `tights`, `leggings`
- `hair_accessory`: `hair accessory`, `head band`, `barrette`, `hair clip`, `hairpin`
- `bag`: `bag`, `handbag`, `shoulder bag`, `tote bag`, `basket bag`

当前选择策略：

- 按搜索结果顺序遍历
- 每个部位只取第一个命中的单品
- 不做颜色打分
- 不做主题打分
- 不做多候选排序

结论：

- 当前 Lolibrary 侧重“先命中一件主服装”
- 还不是“围绕主题挑最优款式”

### 3.2 WEAR

用途定位：

- 更适合抓鞋袜和实穿搭配件
- 也可以抓裙子，但当前优先目的是补齐组合

当前固定分类入口：

- `dress`: `https://wear.jp/women-category/onepiece/dress/`
- `shoes`: `https://wear.jp/women-category/shoes/sandal/`
- `socks`: `https://wear.jp/women-category/leg-wear/socks/`
- `hair_accessory`: `https://wear.jp/women-category/hair-accessory/`
- `bag`: `https://wear.jp/women-category/bag/handbag/`

当前解析逻辑：

1. 打开分类页
2. 读取页面内 `__NEXT_DATA__`
3. 找到 `content_tiles`
4. 进入每个穿搭详情页
5. 读取 `props.pageProps.coordinateItems`
6. 将每个单品转成 `CatalogItem`

当前品牌过滤：

- `brand_slug` 为空时不过滤
- 填写时仅对 `brand` 和 `notes` 做弱匹配

当前选择策略：

- 按分类页顺序遍历穿搭
- 每个部位只取第一个命中的单品
- 不做颜色一致性排序
- 不做搭配整体评分

结论：

- 当前 WEAR 更像“按部位补位”
- 不是“围绕主裙做搭配优化”

## 4. 当前组合策略

当前组合方式由 `collection_service.py` 决定。

### 4.1 Lolibrary 模式

- 输入：`brand_slug + preferred_parts`
- 遍历搜索结果
- 对详情页做部位推断
- 命中后立刻下载
- 每个部位只保留首个命中素材

### 4.2 WEAR 模式

- 输入：`preferred_parts`
- 对每个部位访问固定分类页
- 逐个穿搭展开详情
- 命中该部位就下载
- 每个部位只保留首个命中素材

### 4.3 当前局限

- 组合是“首个命中优先”
- 没有主题驱动
- 没有颜色和元素的统一性校验
- 没有候选池排序
- 没有人工确认步骤

## 5. 当前提示词策略

当前生成入口为 `fashion-generate.py`。

### 5.1 当前输入参数

- `--site`
- `--brand`
- `--pages`
- `--parts`
- `--aspect-ratio`
- `--resolution`
- `--character-count`
- `--prompt`
- `--instructions`
- `--output-subdir`
- `--file-prefix`
- `--export-state`

### 5.2 当前 prompt 拼接方式

当前 `prompt` 结构：

1. 用户输入的 `--prompt`
2. 自动生成与服装匹配的 `场景设定`
3. 自动生成 `主角描述`，支持 `1-2` 人
4. 固定标题 `服饰参考清单：`
5. 每个素材的 `part + item.title`
6. 每个素材的 `prompt_hint`

示例结构：

```text
请生成一位可爱梦幻的少女，全身像，站姿自然，画面干净，突出服装整体搭配感。

服饰参考清单：
- dress: xxx
  要点: 用于参考连衣裙/主服装的版型、花纹、材质与配色。
- shoes: xxx
  要点: 用于参考鞋子的鞋型、鞋跟、装饰、材质与颜色。
- socks: xxx
  要点: 用于参考袜子的长度、花边、图案、透明度与颜色。
```

当前 `instructions` 角色：

- 用作更强约束
- 控制“严格参考素材图”
- 控制整体画风方向

当前问题：

- 主题没有结构化
- style 没有单独字段
- 服装内容和画风要求混在 `prompt` / `instructions` 里

## 6. 当前 AIGC2D 配置来源

配置文件：

- `conf/config-image.json`

当前默认生图后端：

- `current_api = aigc2d`

当前 `aigc2d` 配置项：

- `base_url`
- `api_key`
- `model`
- `default_aspect_ratio`
- `timeout`
- `resolution`

当前 `fashion-generate.py` 调用参数：

- `prompt=final_prompt`
- `image_paths=采集到的素材图路径`
- `aspect_ratio`
- `instructions`
- `resolution`
- `api_type="aigc2d"`
- `save_sub_dir`
- `file_prefix`
- `return_metadata=True`

最终图片保存位置由 `generate_image_aigc2d()` 决定：

- `data/<日期>/<save_sub_dir>/`

## 7. 当前 style 相关配置基线

虽然 `fashion-generate.py` 还没有独立 `style` 参数，但项目已有画风预设体系。

当前 style 配置文件：

- `conf/config-styles.json`

当前 UI 已复用该配置的位置：

- `app.py`
- `make-pic.py`
- `modules/image_generation/single_gen_debug_tab.py`
- `modules/image_generation/sd_workflow_tab.py`
- 其他带“画风预设”下拉的模块

结论：

- 后续 `fashion-generate.py` 新增 `--style` 时，优先复用 `conf/config-styles.json`
- 不建议另起一套 style 存储文件

## 8. 当前产物与复现路径

### 8.1 采集产物

目录：

- `data/fashion-collector/<site>/<timestamp>/`

内容：

- `dress/`
- `shoes/`
- `socks/`
- `collection_bundle.json`
- `<file_prefix>_aigc2d_result.json`

### 8.2 make-pic 状态导出

- `cache/last_state.json`

### 8.3 最终生成图

- `data/<日期>/fashion-generate/`

### 8.4 请求回放

由 `api_backend.py` 生成：

- `*_replay_*.json`
- `*_replay_*.py`

## 9. 当前复现命令

### 9.1 直接采集并生成

```bash
python fashion-generate.py --site wear --pages 1 --parts dress shoes socks --aspect-ratio 2:3 --file-prefix wear_combo_test --export-state
```

### 9.2 使用更明确的提示词

```bash
python fashion-generate.py --site wear --pages 1 --parts dress shoes socks --aspect-ratio 2:3 --file-prefix wear_combo_run2 --export-state --prompt "请生成一位穿着采集服饰组合的少女，全身像，站姿自然，画面明亮，突出连衣裙、鞋子和袜子的整体搭配。" --instructions "请严格参考输入的服饰图片完成一位少女角色的穿搭组合，保持服装、鞋子、袜子的款式与颜色尽量协调，输出精致日系少女插画风格，人物完整，全身清晰。"
```

### 9.3 只做网页抓取验证

```bash
python web-probe.py download "https://wear.jp/yyuk1101a/26674416/" --attr src --contains "imgz.jp" --download-dir cache/web-probe-downloads --limit 2
```

## 10. 后续实施边界

以下内容是计划新增，不属于当前已实现逻辑：

### 10.1 theme

目标：

- 让采集从“首个命中”变成“主题驱动选品”

建议最小方案：

- 新增 `--theme`
- 建立 `主题 -> 关键词` 映射
- 按标题、品牌、分类、tags、notes 做打分

### 10.2 style

目标：

- 让画风要求与服装要求分层

建议最小方案：

- 新增 `--style`
- 从 `conf/config-styles.json` 读取 style 内容
- 将 style 注入 `instructions`

### 10.3 hybrid

目标：

- 主裙走 Lolibrary
- 鞋袜走 WEAR

建议最小方案：

- 新增 `--site hybrid`
- `dress` 来源优先 `lolibrary`
- `shoes/socks` 来源优先 `wear`

## 11. 甜美洛丽塔主题基线

这是后续首个主题实现的默认设计基线。

主题名：

- `甜美洛丽塔`

建议主题标签：

- 风格：`sweet`, `lolita`, `romantic`
- 颜色：`pink`, `white`, `ivory`, `light blue`
- 元素：`lace`, `frill`, `ribbon`, `bow`, `floral`
- 裙装：`jsk`, `op`, `a-line`, `tiered`
- 鞋子：`tea party shoes`, `round toe`, `platform`
- 袜子：`lace socks`, `frill socks`, `over knee`, `knee socks`

建议站点优先级：

- `dress -> Lolibrary`
- `shoes -> WEAR`
- `socks -> WEAR`

建议 style 默认值：

- `梦幻柔光`
- `日系插画`
- `高细节蕾丝`
- `少女时尚感`
- `明亮糖果色`

## 12. 修改时优先查看的文件

如果要改采集策略，优先看：

- `modules/fashion_collection/collection_service.py`
- `modules/fashion_collection/lolibrary_adapter.py`
- `modules/fashion_collection/wear_adapter.py`

如果要改桥接输出，优先看：

- `modules/fashion_collection/make_pic_bridge.py`

如果要改生图 prompt 结构，优先看：

- `fashion-generate.py`
- `modules/others/api_backend.py`

如果要接入 style 配置，优先看：

- `conf/config-styles.json`
- `app.py`
- `make-pic.py`

## 13. 文档使用原则

后续任何采集/主题/style 策略调整，建议同步更新本文件：

1. 先写“目标策略”
2. 再写“实际落地文件”
3. 再写“复现命令”
4. 最后记录“产物路径”

这样可以保证：

- 每次改动都有基线
- 后续问题可以精确回溯
- 不会出现“能跑但不知道具体策略”的状态
