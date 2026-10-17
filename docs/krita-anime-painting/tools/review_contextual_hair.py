"""Local turning planes and overlap shadows after inspecting the full head."""
import copy
import json
from author_contextual_hair import OUT,PARTS,FACE,svg,p,bez,normal,poly,node,R,base

SHADES={
 'A':'M-26 -182C-38 -169 -30 -145 -6 -133Q-39 -139 -47 -155C-48 -173 -34 -182 -26 -182Z',
 'D':'M40 -138C89 -118 110 -76 91 -31L77 36Q91 -28 80 -65C76 -103 56 -125 40 -138Z',
 'E':'M-59 -56C-69 -8 -53 42 -54 76C-48 121 -72 161 -107 213Q-87 166 -79 142C-51 98 -76 58 -68 4Z',
 'F':'M60 -43C74 13 62 64 74 106Q82 144 124 200Q89 172 75 148C50 113 62 61 52 18Z',
 'G':'M-24 -127C-71 -106 -83 -72 -74 -35Q-65 7 -24 43C-52 31 -81 -9 -83 -30C-96 -79 -60 -118 -24 -127Z',
 'H':'M18 -134C30 -106 39 -81 59 -62Q83 -39 74 -7C68 -33 33 -47 22 -75Q8 -110 18 -134Z',
 'I':'M7 -131C19 -105 0 -78 6 -54Q8 -32 21 -12C-7 -31 -12 -50 -7 -77Q5 -112 7 -131Z',
 'J':'M91 -76Q105 -56 87 -30Q105 -15 88 1Q105 20 79 35L78 24Q94 13 82 1Q96 -16 83 -29Q98 -51 91 -76Z'}
updates=[]
for idx,n in enumerate(PARTS):
    item=node(n,'fine')
    # Add broad turning planes below the already generated fine marks.
    original=item['svg'];closing='</svg>'
    extra=''
    if n['id'][0] in SHADES:extra+=p(SHADES[n['id'][0]],'url(#shade)')
    # Warm diffuse reflections fade on every side; small selected sharp
    # reflections remain in the earlier material layer.
    for qi,q in enumerate(n['flows']):
        t=.24 if qi==0 else .55;x,y=bez(q,t)
        rx=12 if n['id'][0] in 'BC' else 10;ry=30 if n['id'][0] in 'BC' else 23
        extra+=f'<ellipse cx="{x-3}" cy="{y-3}" rx="{rx}" ry="{ry}" fill="url(#glow)" opacity=".6"/>'
    # Cast shadow from nearer sections, clipped to this section. Only the
    # exposed offset remains after the nearer opaque section covers its source.
    if n['id'][0] in 'BC':
        for upper in PARTS[idx+1:]:
            if upper['id'][0] in 'BC':extra+='<g transform="translate(2 3)">'+p(upper['d'],'#657496',op=.09)+'</g>'
    item['svg']=original.replace(closing,extra+closing)
    updates.append(item)

# Contact shadow is restricted to exposed skin and excludes all hair pieces.
face_shadow=''
for n in PARTS:
    if n['id'][0] in 'GHI':face_shadow+='<g transform="translate(1.2 2)">'+p(n['d'],'#b997aa',op=.13)+'</g>'
review={**base,'stem':'04-context-reviewed','expected_document':str((OUT/'painting/03-context-fine.kra').resolve()),'layers':updates}
(OUT/'04-review.json').write_text(json.dumps(review),encoding='utf-8')
append={**base,'stem':'hairstyle-context-reviewed','expected_document':str((OUT/'painting/04-context-reviewed.kra').resolve()),'layers':[
 {'name':'33 Restrained fringe contact shadow','svg':svg(face_shadow),'svg_masks':[svg(p(FACE,'#fff'))],
  'svg_erase_masks':[svg(p(n['d'],'#fff')) for n in PARTS]}]}
(OUT/'05-contact.json').write_text(json.dumps(append),encoding='utf-8')
print('Prepared context-specific broad planes, soft reflections and clipped overlap shadows')
