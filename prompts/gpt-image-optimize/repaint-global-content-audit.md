Compare Image 1 (SOURCE before repaint) with Image 2 (RESULT after repaint). Judge whether this is still the same artwork after a limited line repair. Examine both actual pictures. Do not excuse a different subject, pose, camera or scene as an artistic style change.

Check separately: identity including face proportions and hair colour/length; expression/gaze; pose and limb/contact relationships; framing/camera/subject scale/position; major objects and background layout; garment design/accessories/intrinsic colours. Distinguish refinement of an existing ambiguous shape from a clearly added, missing or replaced logical element. Do not use world knowledge to invent source content.

Global replacement, a portrait replacing a full-body view, a changed pose, a different character or missing major objects means FAIL. Such a result is not eligible for local addition cleanup. A finding of uncertain detail is not proof of passing. Small rendering changes and tiny geometric drift may be acceptable, but describe them honestly.

Return one JSON object, no Markdown:
{
  "verdict": "pass|fail|uncertain",
  "confidence": 0.0,
  "same_artwork": true,
  "identity_preserved": true,
  "pose_preserved": true,
  "composition_preserved": true,
  "major_objects_preserved": true,
  "garment_design_preserved": true,
  "global_replacement": false,
  "eligible_for_local_cleanup": false,
  "added_elements": [{"element":"specific logical object, not a rendering difference","location":"result location","source_evidence":"what source shows","confidence":0.0,"removal_instruction":"local deletion restoring the original material/design; retain other content"}],
  "missing_or_replaced_elements": [{"element":"item or subject/pose/framing","source_evidence":"source appearance","result_evidence":"result appearance","severity":"minor|major","confidence":0.0}],
  "line_repair": {"judgment":"improved|unchanged|worse|uncertain","source_evidence":"where source contours are fragmented","result_evidence":"how those exact contours changed"},
  "rendering": {"source_medium_preserved":true,"uniform_ink_or_cel_conversion":false,"observations":"specific comparison of soft edges, brush texture and line weight"},
  "uncertain": ["unresolved visual differences"],
  "summary": "short factual conclusion"
}
eligible_for_local_cleanup is true only when there are confirmed local additions AND the same subject, pose, framing and major objects have been preserved. It is false for no additions or global changes. All booleans must reflect actual evidence, not the desired result.

Use empty arrays for added_elements and missing_or_replaced_elements when there are no confirmed findings; never return placeholder entries from the example schema. source_medium_preserved must be true and uniform_ink_or_cel_conversion false to accept the repaint.
