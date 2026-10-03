这一页的定位信息（供你核对，不是让你照抄）：

- PDF 页序号：{pdf_page}
- 印刷页码：{printed_page}
- 已知所属画师章节：{chapter_hint}
- 可选 page_type 枚举：{page_types}

请读这一页，按下面这个 JSON 结构输出（**只输出 JSON**，值拿不到就 `null` 或 `[]`）：

{
  "page_type": "<枚举之一>",
  "printed_page": <这一页页脚印的页码，整数；没印页码就 null>,
  "chapter": "<所属画师章节名，照抄；非画师章节填 '理论' 或 '笔记'>",
  "artist": "<画师名，照抄；没有就 null>",
  "work_title": "<作品名，照抄；没有就 null>",
  "story_theme": "<作品名旁边那句『讲述…的故事』，照抄；没有就 null>",
  "title": "<页面大标题，照抄；没有就 null>",
  "headings": ["<这一页出现的版块标题，照抄>"],
  "summary_zh": "<这一页的核心内容，2~4 句中文>",
  "main_palette": [
    {"order": 1, "name_zh": "<印刷的色名，照抄>", "name_en": "<你翻的英文色名，小写>",
     "role": "primary|secondary|accent|neutral|null", "share_hint": "<这个色在画面里管什么>"}
  ],
  "hue_tone_range": "<『所用色彩的色相和色调范围』版块讲了什么，中文；没有就 null>",
  "hue_balance": "<『色相平衡』版块讲了什么（有彩色/无彩色的比例关系），中文；没有就 null>",
  "tone_balance": "<『色调平衡』版块讲了什么（4 类色调的比例），中文；没有就 null>",
  "color_relations": ["<色彩关系，英文小写，如 complementary / analogous / monochromatic / triadic / warm-cool contrast / split-complementary>"],
  "highlight_points": [{"id": "A", "title": "<照抄>", "body": "<中文概括>"}],
  "note_points": [{"title": "<照抄>", "body": "<中文概括>"}],
  "process_points": ["<绘画过程中的配色要点，按顺序，中文>"],
  "rules": [
    {"concept": "<英文短标识，小写下划线>",
     "explanation": "<中文，这条规则讲的是什么>",
     "applies_to": "palette|lighting|postprocess|mood|composition|atmosphere",
     "trigger": "<什么画面/情绪/题材下适用，中文>",
     "prompt_keywords": ["<英文逗号短语>"],
     "negative_keywords": ["<英文逗号短语，要避免的>"],
     "color_strategy": "<中文，配色动作>",
     "post_process_strategy": "<中文，确定性后处理动作>"}
  ],
  "prompt_fragments": ["<英文，逗号分隔的提示词短语串>"],
  "unreadable": ["<看不清/无法判断的地方>"]
}
