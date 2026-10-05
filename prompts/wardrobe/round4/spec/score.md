# round4 score.md（离线冻结镜像）

来源模板：prompts/wardrobe/round4/templates/score-round4.md
sha256(来源)：d2ca298e20627f423972783077e937ccd651338585e0bd0aabb87f8fbe4052b1
说明：本文件是 spec 目录运行期读取的副本，与来源模板逐字相同。

---

Inspect one actual image against the supplied person, source outfit, authorized action, protected colours/accessories, scene and framing. Treat the source outfit description as clothing content, not a requirement to preserve it when complete transformation is authorized.

First describe visible clothing construction without using the requested family name. Then infer its dominant family as classical, sweet, gothic, mixed or uncertain, using garment structure and trim organization rather than colour, face, age, painting style or background. If the target label is withheld, do not guess it from filenames. Compare to the requested family only after this observation has been saved.

Evaluate complete_transformation, lolita_structure, target_family_evidence, protected_colours, protected_accessories, person_count, identity_attributes, scene, full_body_framing and visible_defects separately as pass, fail, uncertain or not_visible, each with concrete visible evidence. For complete transformation of a hoodie, utility outfit or uniform, adding lace or a bow while leaving the original garment structure unchanged is insufficient. Preservation guard cases use their own clothing-preservation requirement instead.

Do not copy reference people, wardrobe colours or accessories into the conditions. Do not trade protected failures for attractive clothing. Exact numerical age and identical face cannot be established from a fictional text-only person. If the image cannot be read, say so. Overall status must retain all required failing, uncertain or hidden conditions instead of dropping person count, scene or accessories from aggregation. Human review can disagree with automated observations; retain both records.

ROUND 4 AGGREGATION RULES (the overall verdict must not hide any condition):
1. `overall` is derived mechanically from the ten fields above and never from how attractive the outfit looks:
   - `failed` if any of complete_transformation_or_preservation, lolita_structure_when_requested, target_family_evidence_when_requested, protected_colours, protected_accessories, person_count, scene, full_body_framing is `fail`;
   - `unproven` if none of those are `fail` but at least one is `uncertain` or `not_visible`;
   - `pass` only if all required fields are `pass`. A missing field is `unproven`, never `pass`.
2. `person_count` failing (not exactly one person) or `scene` failing (not the plain grey background with no held objects) always forces `overall` to `failed`. Counting people or scene conditions may never be dropped from the summary.
3. Protected colours and accessories are separate requirements: list each requested colour and each requested accessory with its observed state and its left/right position. An accessory that moved sides is a `fail` with the observed side recorded.
4. `visible_defects` is reported independently and never offsets a protected failure; also record `clothing_connection_issues` for implausible body/clothing joins and `trim_ownership_issues` for lace or ornaments that belong to no plausible garment part.
5. `target_family_evidence` is judged against the requested family only after `family_before_comparison` has been written down. A correct family name produced without a construction-based reason is `uncertain`, not `pass`.
6. Age, exact face identity and painting style are not verifiable from a text-only person description: report them as `not_visible` with the reason instead of guessing.

Return JSON only: {"case_id":"...","family_requested":"classical|sweet|gothic|preservation|null","image_label":"...","observed_clothing":"concrete construction description without the family name","family_before_comparison":"classical|sweet|gothic|mixed|uncertain","family_reasoning":"construction-based","family_after_comparison":"...","complete_transformation_or_preservation":"pass|fail|uncertain|not_visible","lolita_structure_when_requested":"pass|fail|uncertain|not_visible","target_family_evidence_when_requested":"pass|fail|uncertain|not_visible","protected_colours":[{"required":"...","observed":"...","state":"pass|fail|uncertain|not_visible"}],"protected_accessories":[{"required":"...","observed_position":"...","state":"pass|fail|uncertain|not_visible"}],"person_count":"pass|fail|uncertain|not_visible","observed_person_count":1,"identity_attributes":"pass|fail|uncertain|not_visible","scene":"pass|fail|uncertain|not_visible","full_body_framing":"pass|fail|uncertain|not_visible","visible_defects":["..."],"clothing_connection_issues":["..."],"trim_ownership_issues":["..."],"not_visible":["..."],"evidence":["..."],"overall":"pass|failed|unproven","overall_rule":"which field forced the overall status"}. If the image cannot be read at all, return the same schema with every field `not_visible`, `overall":"unproven"`, and put the reason in `not_visible`.
