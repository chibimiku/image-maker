Review the final illustration before publication. Ordered images:
Image 1 = GPT first pass, an anchor for requested action, scene objects and layout;
Image 2 = current final candidate; Image 3 = rendering style reference ONLY;
Image 4 = row-major 3x3 detail sheet of Image 2.
Compare the images visually, not just names or labels. Identity correctness is insufficient.

Trace arms and legs PER CHARACTER from shoulder/pelvis through joints to hands/feet.
Assign every visible foot to its owner; a correct total count across people is insufficient.
Report extra, fused, disconnected or wrongly assigned limbs and provably missing visible
segments. Respect real occlusion and crop; never demand that hidden limbs become visible.
Check hand construction, gloves, shoe/strap attachment and intended support points.
Image 1 may itself have anatomy errors: matching it does not excuse extra/misassigned limbs.
Apply body-proportion ratios only when EXPLICIT BODY-PROPORTION TARGETS are supplied;
otherwise preserve the chosen stylization, including chibi and short limbs.

Check large pose/scene drift against Image 1 and the stated content. Authorized design
variations in the user message override those particular content locks only.
Judge style from Image 3's visible face/eye abstraction, hair grouping, edge hierarchy,
colour layering, gradients, material handling and highlight distribution. Do not mistake
uniform flat anime fills for a reference with layered chromatic shading and selective glow.
Use the selected style text as guidance, but when its rendering shorthand conflicts with
visible reference treatment, report the concrete visible treatment rather than demanding
generic cel shading. Never copy reference identity, clothing, pose or background.
High-key art is allowed. Report over-whitening only where a veil, clipped highlights or
lost midtones erase readable subject forms, intrinsic dark clothing or material separation;
do not compare raw global mean brightness or demand a universally darker image.

Return JSON only with needs_refine, severity (none|minor|major), summary,
structural_issues [{region, observed, repair, confidence}],
line_issues [{region, observed, repair, confidence}],
background_drift [{region, original, candidate, repair, confidence}],
style_gaps [{aspect, candidate, target, repair, confidence}], protected_features [strings].
Confidence is 0 to 1. Include only concrete material defects with confidence >= 0.72.
Also return ownership_uncertain [{region, reason}] for visible limb segments whose owner
or hip connection cannot be established. Do not invent a hidden continuation to dismiss
an unmatched visible segment; ordinary genuine occlusion alone need not be flagged.
Do not approve a changed standing pose merely because hair and outfit match.
