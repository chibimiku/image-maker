CONSERVATIVE IDENTITY CORRECTION — PASS {iteration} OF {max_iterations}. Edit the supplied image, changing ONLY the
listed intrinsic character differences. The supplied image is the sole visual source; do not use or imagine any
other character or style-reference image.
Preserve the exact face geometry, expression, gaze, pose, anatomy, crop, outfit structure except any part explicitly
listed for correction, accessories not listed,
background, lighting, colours not listed, brushwork, line quality and all existing detail. Do not redesign or
beautify the character. Do not add objects.

STABLE IDENTITY ANCHORS EXTRACTED FROM THE ACTUAL GPT FIRST-PASS PROMPT:
{identity_anchors}

These anchors are locks, not a request to repaint the whole character. Preserve every already-correct anchor exactly.
Apply each required correction locally and minimally. Keep the current line continuity and do not weaken, fragment,
redraw or restyle contours outside the corrected feature.

{structural_exception}

REQUIRED CORRECTIONS:
{corrections}

Return one corrected image with the same aspect ratio. No text, border or watermark.
