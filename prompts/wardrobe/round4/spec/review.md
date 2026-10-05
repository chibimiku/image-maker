# round4 review.md（离线冻结镜像）

来源模板：prompts/wardrobe/round4/templates/review-round4.md
sha256(来源)：26a868a9ac0d9f641232b15111dc256424c98b2b96a51888ce22562954a9243e
说明：本文件是 spec 目录运行期读取的副本，与来源模板逐字相同。

---

Review the supplied round 4 request drafts for readiness. This is a text review of drafts, not image validation, and not evidence that the planned generation works.

Round 4 is a single-arm early screening: there is no control arm and no paired comparison. Every slot in wave 1 uses the same frozen channel parameters; the only difference between slots is the case (U, H, W, R, GP, GF) and the target family. Do not demand a control arm and do not treat the absence of one as a defect.

Check the following and report each as pass, fail or unclear with a short actual excerpt:
1. Identity and scene text is carried unchanged into every conversion draft: one 32-year-old adult woman, short wavy dark-brown hair, grey-green eyes, warm brown skin, rectangular black glasses, medium adult build; natural stance, both arms relaxed, no held objects, plain grey background, one person only, full body with both feet visible.
2. Each conversion draft contains the complete-transformation authorization for U, H, W and R, states which source garment categories may be replaced, and keeps the explicitly protected colours and accessories with their left/right positions (blue bird pin, copper leaf pin, gold star brooch on the wearer's left).
3. Preservation drafts (GP, GF) route away from full transformation: they must not carry a target-family Lolita design block, and must state the garments that stay unchanged.
4. No draft contains evidence IDs, image labels, validation expectations, scoring fields, target-family answer keys, evaluator instructions, reference-character traits, or instructions derived from the reference study images.
5. Wardrobe text in a conversion draft is labelled with its provenance: an extracted package must be marked as extracted, and a hand-written draft must be marked as a draft. A draft must never be described as an extracted or trained artifact.
6. Each draft records the template files and their SHA-256 values it was assembled from.
7. The batch and slot structure matches the plan: six extraction batches of at most four images each, three held-out check requests of three images each, twelve conversion drafts and two preservation drafts; no held-out image appears in an extraction request body or attachment list.

Return JSON only: {"ready":true,"blocking":[{"slot":"...","evidence":"short actual excerpt","reason":"..."}],"nonblocking":["..."],"unverified":["..."],"checked":{"identity_text":"pass|fail|unclear","conversion_authorization":"...","preservation_routing":"...","no_expectation_leak":"...","provenance_labelling":"...","template_hashes":"...","slot_structure":"...","heldout_isolation":"..."}}. `ready` must be false only for a real defect in the drafts, a missing draft, or an internally inconsistent plan. The absence of online evidence is not a blocker. Do not claim the prompts work from text review alone.
