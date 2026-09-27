Review anatomy for EVERY visible character. Image 1 is the current final candidate; Image 2 is a row-major
3x3 detail sheet of that same image. Inspect every visible hand, including gloved hands,
hands holding cloth or props, overlapping hands and fingers near the image boundary.
Image 3 is the continuous LEFT lower-body crop (x=0..65%, y=25..100% of Image 1).
Image 4 is the continuous RIGHT lower-body crop (x=35..100%, y=25..100%). They overlap:
the same limb appearing in multiple crops is still one limb. Use these continuous crops
to trace hips, skirt occlusion, exposed thighs, stocking segments and shoes together.
Report clearly malformed, fused, duplicated or disconnected digits, impossible thumb
attachment, incoherent joints and palm-to-wrist connections. Name each region by image
location and describe a concrete local repair. A hand has five digits anatomically, but
occluded fingers need not be visible: never demand five exposed fingers, reveal hidden
limbs, remove gloves, change the gesture or impose realistic proportions on stylized art.
Before reporting, build a per-character limb inventory: trace shoulder-elbow-wrist-hand
and pelvis-thigh-knee-calf-ankle-foot paths separately for each person. Name owners by
screen location and intrinsic appearance (for example left white-dressed / right dark-dressed).
Do not accept the correct TOTAL number of legs across two people as proof: one person
may have one leg and the other three. Trace every visible foot to its own pelvis. Report
extra limbs, fused limbs, disconnected joints, limbs assigned to the wrong character,
and a missing leg only when the visible pose exposes the region where it must continue.
Mark genuinely occluded or off-frame limbs as such; never invent a hidden limb or force
two arms and two legs to be visible. Apply the same checks to single subjects and chibi.
First enumerate ALL visible thigh, stocking, calf and shoe segments, including segments
at the far right or lower edge, before describing each owner. A thigh without a visible
shoe still counts as an exposed segment requiring an owner. Do not explain a leg as
hidden while an unassigned thigh/calf segment is visible. Conversely, absence of a shoe
does not prove absence of the leg. If a visible segment cannot be reconciled with the
owners' two-leg topology, return ownership_uncertain with its image location and reason;
do not approve by inventing an occlusion or guess which limb to delete.
Keep the existing stylized proportions; do not judge style, identity, colour or composition.
Return JSON only with structural_issues (list of objects containing region, observed,
repair and confidence from 0 to 1), character_limb_inventory (list of objects with owner,
arms and legs descriptions including visible paths and occlusions), summary and needs_refine. Set needs_refine=true for
material anatomy defects. Empty structural_issues means no visible defect. Omit uncertain
findings below 0.72 confidence. Do not approve an image solely because identity matches.
Include ownership_uncertain (list of {region, reason}); use [] when every visible segment
has a coherent owner. Ordinary genuine occlusion alone is not an ownership uncertainty.
