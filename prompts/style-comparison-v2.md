Evaluate ALL labeled CANDIDATE images jointly against rendering invariants visible across ALL STYLE DATASET REFERENCE images. You are applying an interpretable visual rubric, NOT computing neural-network embeddings, Gram matrices, LPIPS, CSD, or LoRA training losses. Never fabricate these algorithm values or describe rubric scores as measured percentages. Do not favor later rounds or any generation channel.

Return JSON {"assessments": [...]} with exactly one object per candidate ID. Every object contains:
id; the 12 numeric scores below; subject_fidelity; four boolean gates reference_content_copied, major_structure_defect, explicit_subject_mismatch, uncertain; reason (Chinese); dimension_evidence (object with exactly the 12 dimension keys below).

Each dimension_evidence value is {"reference": "brief concrete Chinese observation, citing reference indices where possible", "candidate": "brief visible Chinese observation", "difference": "brief matching/differing feature"}. Keep each field concise (about 10–25 Chinese characters). Evidence must describe what is visible, not restate the score. Compare multiple representative references rather than copying one reference's character or hue. No additional totals/ranking/best ID; code computes them.

All numeric scores are 0–10: 0 incompatible, 2 very weak, 4 weak with large differences, 6 moderate with clear differences, 8 strong with minor differences, 10 essentially matching. Intermediate values allowed. Score similarity to the source rendering treatment, NOT generic beauty/quality or maximum detail. A sparse source style should reward appropriate sparsity; a flat cel style should not be penalized for lacking painterly realism.

12 independent rendering dimensions:
- linework: stroke shape, continuity, taper, pressure/line-weight modulation and contour color. Separate this from edge hierarchy.
- edge_hierarchy: hard/soft/lost edges, outer-versus-inner strength, overlap separation; do not reward outlining every object.
- face_geometry: stylized face planes, cheek/jaw/chin/nose/mouth simplification and abstraction; exclude character identity and pose.
- eye_rendering: eyelid/lash construction, iris layers, pupil and catchlight hierarchy; exclude intrinsic iris hue and facial identity.
- hair_grouping: broad lock organization, strand sparsity, overlap/shadow pockets, highlight grouping; exclude hairstyle identity and intrinsic hair color.
- shading: cel/painterly transitions, form-plane simplification, contact/occlusion shadow treatment; separate illumination behavior.
- lighting: glow, rim/bounce light, highlight softness and contrast distribution; exclude literal light-source location or copied scene.
- color_logic: warm/cool and saturation/value relationships, chromatic shadows, accent balance; NEVER measure literal reference hair/eye/costume palette copying as style success.
- texture_finish: brush mark/grain/wash/noise/smoothness and scale, distinguishing selective texture from uniform noise.
- material_response: degree of abstraction and rendering grammar for skin/cloth/metal/transparency, NOT copied material objects or outfit design.
- detail_hierarchy: selective detail density and sharpness at focal versus subordinate regions, large-form versus micro-detail balance; exclude literal composition matching.
- background_rendering: scene-independent background simplification, atmospheric depth, edge/detail/contrast balance relative to the figure; requested different scenery is not a style mismatch.

subject_fidelity independently measures explicit requested character attributes, intrinsic colors, action and scene, not rendering quality.
reference_content_copied=true if reference-specific character identity/costume/props replace the requested subject. major_structure_defect=true only for clear serious anatomy/connectivity failures. explicit_subject_mismatch=true for contradicted explicitly requested attributes or scene. uncertain=true if image evidence is insufficient to reliably assess candidate (including regions needed for the rubric that cannot be observed); never invent invisible face/hair/material details. Give provisional numeric values with the uncertainty gate and explanation rather than silently declaring a confident pass. Gate exclusions are independent of numeric scores. reason explains key differences and every triggered gate in Chinese.
