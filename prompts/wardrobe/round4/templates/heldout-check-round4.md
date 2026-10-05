Cross-check the supplied wardrobe rule package against the attached images. The images in this request are held-out references: they were kept out of every extraction request and must not be used to enlarge the extraction evidence set.

For each attached image, report independently:
1. Which of the supplied core_construction rules the image actually supports or contradicts, with the visible garment evidence (upper garment, neckline, sleeve, waist or sash, skirt silhouette and layers, trim placement, fabric motif).
2. Which rules the image cannot test at all, and why (crop, pose, occlusion, garment type absent from this image).
3. Any visible construction the package does not mention, and whether it looks like a design-specific variation rather than a family rule.
4. The family you would assign from garment construction alone, before comparing with the supplied label, and the family you assign after comparing. State explicitly when the two differ.
5. Any structure you had to guess. Never promote a guess into evidence.

Rules: color, face, age, hairstyle, background, painting medium or fashionability cannot establish a family. A single attached image cannot confirm a rule; three independent design groups are required for a core rule, so state how many independent groups this check can actually add. Do not treat these held-out images as new extraction evidence, and do not rewrite the rule package silently: propose changes as separate suggestions with the image ID that motivated each one. If the check motivates a change, say in `held_out_role` that the held-out set has now been used for revision and can no longer be called an independent hold-out.

Return JSON only: {"case_id":"...","family":"...","images":[{"id":"...","visible_construction":"...","rules_supported":[{"rule_id":"...","verdict":"supported|contradicted|untestable","evidence":"..."}],"rules_unmentioned":["..."],"family_before_comparison":"classical|sweet|gothic|mixed|uncertain","family_after_comparison":"...","mismatch_reason":"...","hidden_structure_speculation":["..."],"evidence":["..."]}],"independent_design_groups_added":0,"suggested_changes":[{"rule_id":"...","suggestion":"...","motivated_by":"..."}],"held_out_role":"independent_check|used_for_revision","unverified":["..."]}. Include exactly the supplied IDs, one record each.
