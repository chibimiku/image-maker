Locate a single explicitly identified target face and visible drawn hair strokes for read-only art-style measurement. Return JSON only. Do not judge similarity or beauty.
Coordinates are [x,y] normalized to 0..1 on the supplied EXIF-transposed image. No guessed central/top crop. If multiple characters are visible and a single target is not unambiguous, return {"ambiguous":true,"reason":"..."} and no geometry. If no readable face return {"unavailable":true,"reason":"..."}.
All annotations are proposals requiring human inspection; never set confirmed=true. Only mark actually visible features, never reconstruct occluded or closed eyes.
Schema:
{
 "target_description":"short location/appearance to identify the chosen face",
 "pose":"frontal|three_quarter|profile|unknown",
 "face_outline":[[x,y],...],
 "eyes":{
  "viewer_left":{"state":"open|closed|occluded|unknown","upper_lid":[[x,y],...],"lower_lid":[[x,y],...],"iris":[[x,y],...],"lashes":[[[x,y],...],...]},
  "viewer_right":{"state":"open|closed|occluded|unknown","upper_lid":[[x,y],...],"lower_lid":[[x,y],...],"iris":[[x,y],...],"lashes":[[[x,y],...],...]}
 },
 "hair_regions":[[[x,y],...],...],
 "hair_strands":[{"points":[[x,y],...],"polarity":"dark|light"},...],
 "nose":[x,y],"mouth":[x,y],
 "limitations":["occlusion/low-resolution/lighting/angle uncertainties"]
}
face_outline is the visible facial outline excluding hair, minimum 3 points. viewer_left/right refer to image viewer's left/right, not anatomical left/right. Eye lids have 7 or more points ordered left to right, shared visible inner and outer corners. Trace the visible eyelid boundary, not eyebrow or iris boundary. Iris is a closed polygon for the visible iris portion. Individual lashes are polylines tracing visible strokes from their root on the lid toward the free tip, not the whole eye outline.
Hair regions must exclude face, clothes, background and outer silhouette. Hair strand polylines must lie inside these regions and follow ONLY individual visible fine strands; provide at least 3 readable paths when available, not broad shading or silhouette edges. Do not invent paths for smooth unlined hair. Dark/light polarity distinguishes drawn dark strands and bright strands. Omit unreadable regions; empty lists are valid and measurement will remain unavailable. Nose/mouth are visible centers only and can be omitted.
