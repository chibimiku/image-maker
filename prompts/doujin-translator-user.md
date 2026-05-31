Please analyze and translate the provided image.
Important Context: This is a doujinshi (fan comic), and the text is highly likely to be written vertically.

{context_instruction}

CRITICAL IMAGE DIMENSIONS:
The image provided has a resolution of {img_width} pixels in width and {img_height} pixels in height.

Your task is to identify all text regions, extract the original text in {source_lang}, and translate it into {target_lang}.
ALSO, you must generate a summary of the current page's plot, characters involved, and key events.

CRITICAL INSTRUCTION FOR REGION SEPARATION AND COORDINATES:
1. You MUST treat every single speech bubble, text box, or visually separated text block as a completely independent region.
2. The coordinates in "xyxy" and "lines" MUST BE ABSOLUTE PIXEL VALUES based on the {img_width}x{img_height} resolution.
3. DO NOT output fake, sequential, or grid-like coordinates (e.g., [50,50, 300,300]). You must visually locate the actual text and estimate its precise bounding box on this {img_width}x{img_height} canvas. X values must be between 0 and {img_width}, Y values must be between 0 and {img_height}.

Return the result STRICTLY as a JSON object containing TWO main keys: "page_summary" and "regions".

1. "page_summary" MUST contain:
   - "characters": Array of strings (Names of characters appearing or mentioned on this page).
   - "plot": String (A brief summary of the events/dialogue on this page).
   - "key_terms": Array of strings (Important items, places, or specific terms introduced or used).

2. "regions" MUST be an array of objects. Each object MUST contain the following keys:
   - "xyxy": Array of 4 integers [x_min, y_min, x_max, y_max] representing the precise bounding box.
   - "lines": A list of text lines within the region. Each line is represented by a polygon of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
   - "text": Array of strings. The original extracted text.
   - "translation": String. The final translated text for the entire region combined.
   - "src_is_vertical": Boolean. true if the original text is written vertically, false otherwise.
   - "_detected_font_size": Integer. The estimated font size of the text.

Example structure:
{
  "page_summary": {
    "characters": ["Tachibana", "Yamada"],
    "plot": "Tachibana hands Yamada a glass of water and thanks him for his help.",
    "key_terms": ["Water glass"]
  },
  "regions": [
    {
      "xyxy": [331, 331, 370, 558],
      "lines": [
        [[331, 331], [370, 331], [370, 558], [331, 558]]
      ],
      "text": ["水だワン！！"],
      "translation": "是水汪！！",
      "src_is_vertical": true,
      "_detected_font_size": 39
    }
  ]
}
