Analyze the supplied image as an observer. Return only one JSON object.
Describe visible facts in neutral English: medium/art style, composition, lighting,
viewpoint, people, hair and eye colour when visible, clothing construction and
colours, accessories, props, environment, poses and visible interactions.

Use the source's actual medium and clothing terminology. Do not convert photographs
to illustrations or replace a fashion category with another category at this stage.
Use age-neutral terms such as person or female character; do not infer age.
Preserve the actual framing, coverage, pose, occlusion and uncertainty. If a face or
other detail is blurred, covered or unrecognizable, say so briefly. Do not imagine
hidden features, remove censorship or invent anatomy. Describe visible clothing
without sexual emphasis. Do not transcribe text inside the image.

Be precise where the image supports precision. An approximately 300-500 word
description is sufficient; use fewer words when evidence is limited. Never invent
facts or repeat material to meet a word count. The short description should be
approximately 100 words or fewer. Later text editing handles generation adaptations.

Include a natural Japanese title (at most 20 characters) and its Chinese translation.
Use standard Japanese characters, without Latin letters, numbers or punctuation.
Provide up to 12 Japanese Pixiv tags grounded in clearly visible content. Provide up
to {booru_tag_limit} known Danbooru tags, lowercase with underscores, ordered by
importance. Local tag candidates are fallible suggestions: keep only those supported
by the image. Omit guesses. Do not claim a censored or obscured face is clear.

Return these fields:
{
  "english_description": "...",
  "short_description": "...",
  "booru-tags": ["..."],
  "japanese_title": "...",
  "chinese_title": "...",
  "pixiv_tags": ["..."]
}
