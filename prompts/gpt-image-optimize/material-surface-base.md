# MATERIAL SURFACE PASS

You are a MATERIAL RENDERING model. The input image is the sole source of truth for content; this pass changes **how the named surfaces respond to light**, and nothing else.

## LOCKED (do not change)

Do not change identity, face, eyes, expression, hairstyle, pose, anatomy, garment silhouette and cut, garment base colour and hue, trims and bows, props, background, composition, framing, crop, line character or aspect ratio. Do not add, remove, move or redesign any object. Do not restyle the picture: no cel conversion, no full repaint, no global contrast, brightness, saturation or colour-grade change, no resolution change.

## WHAT THIS PASS IS FOR

The present render is too flat: the named surfaces read as matte paper or flat fill, with no specular structure and no micro-texture. Your job is to give those surfaces their real material behaviour.

The base colour, pattern and geometry stay exactly as they are. What changes is **value structure inside the material**: where the specular sits, how sharp and how elongated each highlight is, where the reflected band falls, how steep the falloff is, how fine the micro-grain and crease lines are, and how dark the local occlusion inside folds and seams is.

Highlights may only be added **on the named surfaces**, and they must follow the existing form — never a spot, blob or sparkle floating on the picture. Nothing outside the named surfaces may gain gloss, contrast or new texture, and no existing detail may be lost.

## TARGETS

**{targets}**

Treat only those surfaces as **{material}** and apply:
{material_clauses}

The target is defined by **how the surface is already rendered**, not by an object list: whatever object carries that material in this image is the target, and an object that does not carry it stays untouched. If a target is only partly visible, treat the visible part only; never complete or invent the hidden part.

## OUTPUT

One image, same framing, same colours, same level of detail. No text, no border, no watermark.
