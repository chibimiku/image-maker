"""Original hair study on the reviewed head; staged native Krita document."""
import ast
import copy
import json
import math
import random
import re
from pathlib import Path

OUT=Path('data/test-result/20261016/krita-hair-material-study')
OUT.mkdir(exist_ok=True)
tree=ast.parse(Path('docs/krita-anime-painting/tools/author_seated_practice.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'hp','hd','p'}],type_ignores=[]),'head-projection','exec'))
R=random.Random(161044)
DEFS='''<defs>
<linearGradient id="newFront" x2="100%" y2="28%"><stop stop-color="#7099b8"/><stop offset=".28" stop-color="#b9d7e4"/><stop offset=".53" stop-color="#a0c4d9"/><stop offset="1" stop-color="#557d9e"/></linearGradient>
<linearGradient id="newBack" x2="100%" y2="10%"><stop stop-color="#638aa9"/><stop offset=".27" stop-color="#accbdb"/><stop offset=".58" stop-color="#557d9d"/><stop offset=".82" stop-color="#8bb0c8"/><stop offset="1" stop-color="#486e91"/></linearGradient>
<radialGradient id="silkLight"><stop stop-color="#edf9f6" stop-opacity=".7"/><stop offset=".45" stop-color="#d0eaf0" stop-opacity=".35"/><stop offset="1" stop-color="#bfdde8" stop-opacity="0"/></radialGradient>
<linearGradient id="silkGlint" x2="0%" y2="100%"><stop stop-color="#c8e4ed" stop-opacity="0"/><stop offset=".4" stop-color="#eef9f4" stop-opacity=".55"/><stop offset=".65" stop-color="#e0f1ed" stop-opacity=".38"/><stop offset="1" stop-color="#abcbdc" stop-opacity="0"/></linearGradient>
</defs>'''

def path(d,fill='none',stroke='none',w=1,op=1):
    tokens=re.findall(r'[MCQLZ]|-?\d*\.?\d+',d)
    cmds=[];i=0
    counts={'M':2,'L':2,'C':6,'Q':4,'Z':0}
    while i<len(tokens):
        c=tokens[i];k=counts[c];cmds.append([c]+[float(v) for v in tokens[i+1:i+k+1]]);i+=k+1
    return hd(cmds,fill,stroke,w,op)

def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2600" viewBox="350 150 290 314.1666666667">'+DEFS+s+'</svg>'

BACK='M-100 -30C-116 -111 -64 -155 5 -151C87 -153 116 -103 103 -38C100 20 123 87 121 140C120 183 128 213 140 238Q120 231 111 206C118 253 103 279 89 292Q80 265 73 244C64 278 41 308 13 324Q27 286 19 262C-9 298 -38 306 -66 309Q-50 282 -53 254C-79 286 -100 297 -122 292Q-104 261 -105 232Q-125 255 -143 255C-115 207 -126 160 -112 105Q-100 51 -100 -30Z'
FRONT='M-100 -30C-116 -111 -64 -155 5 -151C87 -153 116 -103 103 -38Q105 -8 96 13L90 -14Q78 -20 73 -39C63 -65 49 -91 30 -112C26 -72 -18 -38 -65 -24Q-63 25 -69 58C-68 105 -85 143 -96 164C-99 112 -95 79 -97 48Q-111 74 -120 98C-109 56 -107 9 -100 -30Z'
rear=path(BACK,'url(#newBack)','#5c82a1',.65)
front=path(FRONT,'url(#newFront)','#668cab',.55)
# Long coherent shadow shapes follow the same connected scalp, without an
# independent cap or horizontal ring at the crown.
front+=path('M29 -141C14 -104 -12 -62 -65 -24Q-31 -50 -15 -83C3 -112 8 -135 9 -149Z','#608aa9',op=.35)
front+=path('M47 -139C94 -117 104 -79 96 -34L96 13L90 -14Q83 -22 80 -42C88 -83 73 -113 47 -139Z','#486e91',op=.31)
front+=path('M-81 -101C-97 -54 -95 8 -97 48Q-88 29 -82 0C-88 -50 -79 -75 -65 -105Z','#6b95b3',op=.28)
front+=path('M-69 40C-69 101 -85 143 -96 164Q-83 102 -85 56Z','#507b9c',op=.28)

# Major rear separations are broad value wedges, not outlines on every fiber.
for d in [
 'M-92 -55C-121 40 -99 105 -116 168C-120 205 -116 234 -122 256Q-108 224 -103 192C-88 119 -112 28 -77 -64Z',
 'M-65 55C-76 118 -49 179 -60 227Q-67 258 -87 277C-57 264 -46 235 -45 211C-39 155 -62 110 -52 62Z',
 'M91 -44C110 29 84 91 106 158Q120 205 111 235C125 206 119 182 116 162C95 80 122 17 103 -40Z',
 'M59 53C47 115 82 169 67 221Q57 253 31 283C64 264 80 235 81 210C84 153 59 112 72 57Z']:
    rear+=path(d,'#476d90',op=.32)

gloss=path('M-98 -74C-87 -118 -46 -143 -10 -144C-42 -127 -66 -98 -76 -62Z','url(#silkLight)')
gloss+=path('M-65 -112C-48 -130 -17 -139 6 -137Q-12 -112 -32 -81Q-50 -67 -65 -61Z','url(#silkGlint)')
gloss+=path('M46 -130C71 -120 83 -98 88 -75L81 -64C75 -92 61 -111 46 -130Z','url(#silkGlint)',op=.7)
gloss+=path('M-97 9C-92 44 -101 89 -95 128Q-85 91 -84 59Q-89 23 -85 3Z','url(#silkLight)',op=.65)

def cubic(q,t):return tuple((1-t)**3*q[0][j]+3*(1-t)**2*t*q[1][j]+3*(1-t)*t*t*q[2][j]+t**3*q[3][j] for j in (0,1))
flows=[((-12,-143),(-49,-139),(-88,-103),(-87,-27),25),
       ((11,-145),(-4,-107),(-30,-57),(-65,-24),31),
       ((32,-142),(80,-124),(96,-68),(93,-20),23),
       ((-91,-40),(-111,28),(-79,87),(-96,158),24)]
fibers=''
for a,b,c,d,count in flows:
    for j in range(count):
        off=R.uniform(-5.5,5.5)
        q=[(v[0]+off,v[1]+off*.22) for v in (a,b,c,d)]
        t0=R.uniform(.04,.45);t1=R.uniform(.64,.97)
        pts=[cubic(q,t0+(t1-t0)*k/6) for k in range(7)]
        dd='M'+'L'.join(f'{x:.3f} {y:.3f}' for x,y in pts)
        fibers+=path(dd,stroke=R.choice(['#d6eaf0','#517b9b','#dfeeea']),w=R.uniform(.12,.24),op=R.uniform(.16,.32))
back_fibers=''
for side in (-1,1):
    for j in range(42):
        xx=side*R.uniform(82,119);y0=R.uniform(-20,110);y1=y0+R.uniform(45,120)
        back_fibers+=path(f'M{xx} {y0}C{xx+side*8} {y0+36} {xx-side*5} {y1-25} {xx+side*3} {y1}',stroke=R.choice(['#c3dce5','#3c6587','#9abbd0']),w=R.uniform(.13,.27),op=.25)

scene=json.loads(Path('data/test-result/20261016/krita-seated-practice-v4/head-practice/head-closeup.json').read_text(encoding='utf-8'))
layers=scene['layers'];by={n['name']:n for n in layers}
for name,body in [('06 Long rear hair grouped by flow',rear),('17 Rooted bangs and shoulder locks with fine fibers',front)]:
    by[name].clear();by[name].update(name=name,svg=svg(body),visible=False)
for n in layers:
    if n['name'].startswith('19b') or n['name'] in ['11b Soft shadows under the bangs','24 Refined hair broad shadows and broken sheen']:
        n['visible']=False
    if n['name']=='23 Refined cheek planes and native soft color':n['svg_erase_masks'][0]=svg(path(FRONT,'#fff'))
    if n['name']=='25 Ear lobe attachment and occluded earrings':n['svg_erase_masks']=[svg(path(FRONT,'#fff'))]
# Draw the construction over the already reviewed facial anatomy, then hide it
# only after checking crown volume and the locations of brows and ears.
guides=path('M-98 -30C-109 -109 -62 -151 5 -148C80 -150 109 -101 100 -36',stroke='#9a7c89',w=.65)
guides+=path('M-96 -30C-80 -106 -29 -145 27 -143M27 -143C12 -87 -21 -49 -65 -24M27 -143C75 -122 95 -75 93 -20M-95 -25C-105 40 -87 95 -96 158',stroke='#b58d94',w=.7)
guides+=path(FRONT,stroke='#8cabc0',w=.4)+path(BACK,stroke='#b5c7d1',w=.35)
layers.append({'name':'26 New hair skull and flow sketch','svg':svg(guides),'checkpoint':'01-hair-sketch.png'})
scene.update(name='Azure head — connected hair material study',stem='01-hair-construction',blank_checkpoint='00-blank.kra',layers=layers,details=[{'file':'head-detail.png','rect':[5,5,280,285]}])
(OUT/'01-construction.json').write_text(json.dumps(scene,ensure_ascii=False),encoding='utf-8')

mass_updates=[{'name':name,'visible':True} for name in ['06 Long rear hair grouped by flow','17 Rooted bangs and shoulder locks with fine fibers']]
mass_updates.append({'name':'26 New hair skull and flow sketch','visible':False})
mass={**scene,'stem':'02-hair-masses','expected_document':str((OUT/'painting/01-hair-construction.kra').resolve()),'layers':mass_updates}
(OUT/'02-masses.json').write_text(json.dumps(mass),encoding='utf-8')
detail=[{'name':'27 Grouped silk reflections','svg':svg(gloss),'svg_masks':[svg(path(FRONT,'#fff'))]},
        {'name':'28 Selective fibers following connected masses','svg':svg(fibers),'svg_masks':[svg(path(FRONT,'#fff'))]}]
# Rear texture is inserted into its existing dedicated layer so it remains
# behind the face and front fringe rather than being appended on top.
back_update={'name':'19b Fine fibers for 06 Long rear hair grouped by flow','svg':svg(back_fibers),'visible':True,'svg_masks':[svg(path(BACK,'#fff'))],
 'svg_erase_masks':[by['11 Face and ear base aligned to head camera']['svg'],svg(path(FRONT,'#fff'))]}
material={**scene,'stem':'03-hair-material','expected_document':str((OUT/'painting/02-hair-masses.kra').resolve()),'layers':[back_update]}
(OUT/'03-rear-texture.json').write_text(json.dumps(material),encoding='utf-8')
final={**scene,'stem':'azure-hair-material','expected_document':str((OUT/'painting/03-hair-material.kra').resolve()),'layers':detail}
(OUT/'04-highlights.json').write_text(json.dumps(final),encoding='utf-8')
(OUT/'elements.json').write_text(json.dumps({'scope':'new connected hairstyle on previously reviewed head, not new full-body illustration','parting':[27,-143],'camera':{'yaw':13,'pitch':-2,'roll':-5},'light':'large upper-left source','sequence':['skull and directional sketch','connected front and back masses','shared shadows','limited grouped reflections','selective fine fibers','visual correction'],'checks':['no independent scalp cap','highlights change width and dissolve along flow','face and ears occlude rear hair','eyes stay readable','detail subordinate to form']},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared construction, masses, rear texture, then grouped reflection recipes')
