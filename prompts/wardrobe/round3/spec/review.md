Review the supplied experimental request snapshots for a two-arm comparison. This is a text review of candidate readiness, not image validation.

Arm identities are fixed and must not be confused:
- `v3_control` is the FROZEN control carried over from the previous round. It already excludes target design in preservation requests and specifies an upper-chest crop boundary. Round2 still showed occasional crop widening; the new characters are unverified. T06 adapts the existing visible scope to authorized redesign. Report them, but they are not candidate blockers and must not be treated as unresolved competing candidates.
- `v4_candidate` is the only candidate under review. `ready` refers to the candidate arm only.

Check the candidate arm (v4_candidate):
- preservation requests carry no target-wardrobe design block and list the described garments, colours and accessory positions without inventing new ones;
- close-portrait requests contain only visible-region clothing rules plus an explicit crop boundary, and no skirt, hem or shoe instructions. A NEGATIVE instruction that forbids drawing or adding skirt, waist, hem, sleeve or shoe content outside the crop is a visible-region rule, not a full-outfit instruction: never flag it as a blocker. Only flag it if the prompt actually asks for skirt, hem, shoes or a wider view to be drawn;
- every generated prompt omits test expectations, gold verdicts and evaluator instructions;
- the identity facts (age expression, skin tone, hair, glasses, build) are carried unchanged. Quote the character description as the only source of identity facts: a prompt must not add a fact the description does not state (for example claiming glasses for a character whose description has none), and must not drop one it does state. Both full-body redesign cases are expected to carry the same target-wardrobe design block as the control; report it as a blocker if the candidate drops it.
Also confirm each character base text is not being mistaken for the final generation request, and that the two arms differ only in the intended ways. Never waive person identity or anatomy gates as a shortcut.

Return JSON {"ready":true,"blocking":[{"file":"...","evidence":"short actual excerpt","reason":"..."}],"nonblocking":["..."],"control_defects_reported":["..."],"unverified":["..."]}.

`ready` must be false only when the CANDIDATE arm has a real defect, when the candidate snapshot is missing, or when the spec is internally inconsistent. Known control defects, having two arms, and the absence of online evidence are not candidate blockers. Do not claim the candidate works from text review alone.
