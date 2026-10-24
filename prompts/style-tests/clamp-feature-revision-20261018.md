Re-examine the attached CLAMP original colour illustrations, especially Cardcaptor Sakura and Chobits, against the supplied current prompts and ACTUAL GENERATED RESULTS. The aim is stronger recognition of the source drawing language, not simply a prettier generic watercolour illustration. This is prompt revision, not weight training.

Use SOURCE images as evidence. Generated results show failures only; they are never new style references. Source indices correspond to the original ten-image dataset. Ignore publication lettering, ornamental frames, volume numerals and margins. Do not transfer Sakura/Chi identity, specific intrinsic colours, hairstyles, ears, costume details, flowers, wings, magic objects or composition. Separate age-dependent anatomy from drawing conventions; the fixed test subject is a 24-year-old adult.

Inspect these hypotheses, accepting, qualifying or rejecting each from visible evidence:
- The current words "romantic shoujo", "luminous pastels", "soft watercolour" and generic "large expressive eyes" are insufficiently discriminative. Identify which actual line and eye construction separates these references from modern generic anime watercolour.
- Examine vertical eye aperture versus width, the space occupied by the iris, upper-lid arc and weight, outer-corner taper and lash grouping, lower-lid omissions, pupil/iris relationship, dark upper iris and reserved highlights. Do not invent measured ratios or force children's proportions onto an adult. State cross-reference invariants separately from series/age-dependent variants.
- Examine cheek/jaw/chin abstraction, placement and economy of the tiny nose and mouth, absence of realistic lip volume and facial planes. Adult identity can survive a change in drawn facial abstraction. "Preserve identity" must not mean freeze the generated eye geometry that is currently wrong for the target rendering.
- Examine intentional ink-like continuity, local pressure accents at eyes and overlaps, and the relationship of bolder graphic anchors to fine internal strokes. Do the references support banning all black/dark line and large dark fills? Avoid turning everything into weak sepia contours.
- Examine long decisive hair divisions, grouped locks, negative gaps, plain highlight shapes, and fine terminal strands. Do not replace readable drawn hair with dense painterly texture or shiny 3D ribbons.
- Distinguish flat, clear chromatic shapes and paper-white skin/light areas from subtle wash variation. Do not equate CLAMP with beige fog, global low saturation, uniform paper noise, diffuse brown lighting or physically modelled thick paint.
- Fabrics and backgrounds must retain their requested content while using the source line/shape simplification. Decorative framing and magical particles are not a substitute for source facial/line grammar.

Compare the earlier winning Gemini direct, GPT first-pass and repainted images with the sources. Give a compact diagnosis of why they still look generic. Contradictions in the old wording may be removed rather than mechanically preserved. Do not add any hypothesis unsupported by the source images. An 80% statement is a visual judgement from this finite set, not a computed success rate; document genuine cross-series differences instead of hiding them.

Return a JSON object:
{
  "source_evidence": [{"feature": "precise drawing feature", "reference_indices": [0], "visible_evidence": "Chinese description", "exceptions_or_variants": "Chinese description"}],
  "generated_result_failures": [{"candidate_id": "exact supplied ID", "visible_difference": "Chinese description", "prompt_cause": "specific wording or conflict", "correction": "specific rendering instruction"}],
  "revised_master_prompt": "450-750 word English master instruction. Start with: You are an advanced generative AI model specialized in creating images in the following art style. Adhere strictly to these stylistic rules: . Put discriminative facial/eye and ink/shape grammar first, colour and materials second. Describe HOW rather than WHO. Preserve adult identity, intrinsic colours, hairstyle, clothing, action and setting, but explicitly permit source-supported changes to stylized eye proportions, lid construction, iris design and facial abstraction. Avoid contradictory locks that freeze those rendering changes.",
  "revision_guidance": "200-350 word English continuing brief for the next two rounds: retained distinctive targets, variable traits, prohibited regressions, and cover-design exclusions. This is guidance to verify, not automatic evidence.",
  "removed_or_rewritten_claims": [{"old_claim": "old text", "reason": "Chinese reason", "replacement": "English replacement"}],
  "confidence_limits": ["Chinese limitations"]
}
