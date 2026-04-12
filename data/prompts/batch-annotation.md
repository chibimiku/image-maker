请基于提供的图片，返回一个严格 JSON 对象。JSON 数据应包含以下三个键：
long_description: 约 500 词。详细、深入地描述人物的每一个视觉细节。重点着重描述人物外貌（瞳色、发色、发型、眼睛、睫毛、脸型）和衣装（衣服、帽子、手套、袜子、鞋子、配饰、手持物，包括其印花、裁剪、版型、装饰）。同时描述动作、状态、表情。如果存在半透明情况，需要明确说明。
short_description: 约 100 词。简明扼要地概括人物、主要特征、动作和表情。重点着重描述人物外貌和衣装。如果存在半透明情况，需要明确说明。
booru-tags: 最多 {booru_tag_limit} 个。使用标准的 Booru-style 标签数组。标签应涵盖人物的关键特征、服装（包括细节）、动作、表情。排除所有环境和背景标签。如果存在半透明情况，需要使用对应标签明确说明。标签必须按重要性从前到后排序（最重要的在最前），并且每个标签都必须是 Danbooru 网站可检索到结果的有效标签。
booru-tags 格式要求：每个标签必须是小写英文，词与词之间使用下划线（例如 long_hair），不要使用自然语言短句。
booru-tags 风格示例（仅用于风格参考，不要照抄）：["1girl", "solo", "long_hair", "blue_eyes", "looking_at_viewer", "smile", "hair_ornament", "frilled_dress", "thighhighs", "lace_gloves"]。

输出格式示例（仅示例结构，不要复述示例内容）：
{
  "long_description": "......",
  "short_description": "......",
  "booru-tags": ["tag1", "tag2", "tag3"]
}

硬性要求：
1) 只返回 JSON，不要输出 Markdown，不要输出额外说明。
2) 必须包含且仅包含 long_description、short_description、booru-tags 这三个键。
3) long_description、short_description 必须为字符串，booru-tags 必须为字符串数组。
