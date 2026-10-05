Inspect the actual attached images. Each image is labelled in the text block immediately before it. Do not infer identity from filenames. Treat each image as independent: judge it only against the conditions given for this case, never against the other image, and never require a garment, colour, accessory or skin tone that this case did not describe.

Evaluate these dimensions separately and report each as pass, fail, uncertain or not_visible:
1. person_count — exactly one adult woman.
2. identity_anchors — age expression (adult/mature where described), skin tone, hair colour and style, eye colour, glasses presence and type, visible body build.
3. clothing_action — for PRESERVE: the described garments, their categories, colours and described construction are still there; a decorative replacement dress fails even if it is attractive. For DESIGN/FILL: a recognisable, coordinated clothing structure rather than a few added ornaments.
4. specified_items — each explicitly described garment, colour and named accessory appears in its described position (for example a brooch on the left lapel, a pin on the left upper chest, a book in the right hand).
5. framing — the actual lower frame edge on the figure against the stated framing requirement. For a close portrait, seeing the waist or lower torso fails even if the feet are absent.
6. scene — the described setting and held objects.

Rules: uncertain is not a pass; do not infer hidden body parts; for a close portrait the full outfit is not verifiable and missing skirt or shoes are not a defect; an age number cannot be verified from a face, so only check that the adult/mature expression is appropriate. Protected failures must not be offset by attractive clothing. If an image cannot be read, say so instead of imagining it.

Return JSON only: {"case_id":"T01","images":[{"label":"A","observed_person_count":1,"observed_clothing":"concrete visible description","observed_lower_frame_edge":"where the image ends on the figure","person_count":"pass|fail|uncertain","identity_anchors":"pass|fail|uncertain","clothing_action":"pass|fail|uncertain","specified_items":"pass|fail|uncertain","framing":"pass|fail|uncertain","scene":"pass|fail|uncertain","not_visible":["..."],"evidence":["specific evidence"],"full_outfit_verifiable":false}],"preferred":"A|B|tie|inconclusive","reason":"concrete comparison; do not trade a protection failure for attractive clothing"}. Include exactly the labels supplied, one record each.
