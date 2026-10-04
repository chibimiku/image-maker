Evaluate candidate outputs against a reference artwork set to select the closest reusable rendering style. Reference pictures define drawing mechanics, not required subjects or literal colours. The candidate images all use the same test subject. Do not favour the latest round, the longest prompt, or copied reference content.

Score each candidate on these dimensions, each from 0 to 10:
- line_and_edge: tapered coloured contours, selective lost edges, long connected curves, hierarchy between silhouette and internal detail;
- shape_language: face abstraction, eye construction, hair mass grouping, planar folds and silhouette rhythm;
- shading_and_finish: balance of crisp planar shadow shapes and selective soft transitions, luminous highlights, surface treatment;
- colour_relationships: separation of luminous lights and saturated dark accents, local colour preservation, hue shifts and warm/cool relationships. Similar literal hues alone do not earn points;
- composition_and_detail: layered flowing shapes, distribution of detail and negative space, coherent background treatment adapted to the requested subject.

Also score subject_fidelity from 0 to 10, checking the actual supplied test prompt. Identify reference content leakage, visible anatomy defects, excess generic gloss, muddy gradients, disconnected lines, excessive white haze, unwanted lettering, and incorrect scene or palette transfer. Treat subject fidelity as a separate requirement; do not confuse image beauty with style closeness. Assign style_score as the arithmetic mean of the five rendering dimensions. Assign selection_score as 0.85 * style_score + 0.15 * subject_fidelity. Serious visible structural defects or reference character leakage make eligible=false regardless of the numeric score. Assess uncertainty honestly: scores are qualitative visual judgments, not calibrated measurements.

Return ONLY JSON:
{"candidates":[{"id":"supplied candidate label","line_and_edge":0,"shape_language":0,"shading_and_finish":0,"colour_relationships":0,"composition_and_detail":0,"style_score":0,"subject_fidelity":0,"selection_score":0,"eligible":true,"strengths":["specific visual evidence"],"weaknesses":["specific visual evidence"],"content_leakage":[],"anatomy_issues":[],"confidence":0.0}],"best_candidate_id":"eligible candidate label or null","comparison_summary":"brief evidence-based explanation"}
