Compare two images of the same scene. Image 1 is BEFORE repaint and is the sole authority for which logical content elements exist. Image 2 is AFTER repaint, with intentional improvements to line coherence and rendering. Examine both images directly.

Identify only logical content introduced by repaint: new objects, accessories, ornament motifs, garment components, extra characters, body parts, symbols or background props that are visibly absent in Image 1. For each candidate inspect the corresponding region in Image 1. Do not mistake a sharpened existing detail, clarified ambiguous shape, changed brushwork, shading, colour nuance, line smoothness or tiny position drift for a new object. If the before image is too ambiguous to establish absence, put it in uncertain and do not recommend deletion. Do not judge whether the picture ought to contain the element based on world knowledge or the scene's theme.

Return exactly one JSON object:
{
  "additions": [
    {
      "element": "specific added logical element",
      "location": "precise visual location on Image 2, including nearby landmarks",
      "before_evidence": "what is visibly present at the corresponding location in Image 1",
      "after_evidence": "what makes this a distinct added element in Image 2",
      "absent_before": true,
      "confidence": 0.0,
      "removal_instruction": "a local edit that removes only this element and restores the original underlying material/design without changing rendering elsewhere"
    }
  ],
  "uncertain": ["ambiguous differences that must be preserved"],
  "preserve": ["important already-existing objects/details at risk of being wrongly removed"],
  "summary": "short factual comparison"
}
Do not output Markdown or generic stylistic criticism. An empty additions list is valid. Removal is permissible only where absence is clear from the before image. Confidence must be between 0 and 1.
