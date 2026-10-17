"""Continue the planned queue with frontal facial planes and tapered lids."""
import copy
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

OUT=Path('data/test-result/20261017/krita-planned-head-batch')
scene=json.loads((OUT/'final-scene-snapshot.json').read_text(encoding='utf-8'))
face=copy.deepcopy(next(n for n in scene['layers'] if n['name']=='Face placement for hairstyle evaluation'))
root=ET.fromstring(face['svg']);R=random.Random(171220)
for par in root.iter():
    for n in list(par):
        if n.tag.split('}')[-1]=='path' and n.get('stroke') in {'#77748a','#a8a2b0','#bf98a5'}:par.remove(n)
for n in root.iter():
    if n.get('fill')=='#f9e7da':n.set('fill','url(#frontalSkin)')
    if n.get('fill')=='#f1d9ce':n.set('fill','url(#frontalNeck)')
defs=next(n for n in root if n.tag.split('}')[-1]=='defs')
newdefs=ET.fromstring('''<defs>
<linearGradient id="frontalSkin" gradientUnits="userSpaceOnUse" x1="-63" y1="-80" x2="95" y2="120"><stop stop-color="#fff3e5"/><stop offset=".6" stop-color="#f8e1d7"/><stop offset="1" stop-color="#e6bdc3"/></linearGradient>
<linearGradient id="frontalNeck" gradientUnits="userSpaceOnUse" x1="-30" y1="80" x2="38" y2="145"><stop stop-color="#e6bec1"/><stop offset=".52" stop-color="#f4dacd"/><stop offset="1" stop-color="#e6c3bf"/></linearGradient>
<radialGradient id="noseSoft"><stop stop-color="#d59cab" stop-opacity=".35"/><stop offset="1" stop-color="#e7b6bc" stop-opacity="0"/></radialGradient>
<radialGradient id="noseLight"><stop stop-color="#fff9e9" stop-opacity=".8"/><stop offset="1" stop-color="#fff4e7" stop-opacity="0"/></radialGradient>
<radialGradient id="cheek"><stop stop-color="#e4a1b1" stop-opacity=".27"/><stop offset="1" stop-color="#f1c4c2" stop-opacity="0"/></radialGradient>
<radialGradient id="coolPlane"><stop stop-color="#989bbd" stop-opacity=".15"/><stop offset="1" stop-color="#bcc2d2" stop-opacity="0"/></radialGradient>
</defs>''')
for n in newdefs:defs.append(n)
def p(d,fill='none',stroke='none',w=1,op=1):return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'
def e(x,y,rx,ry,fill,op=1):return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{op}"/>'
features=''
for side in [-1,1]:
    cx=side*37
    features+=p(f'M{cx-23} 0Q{cx} -17 {cx+23} -1L{cx+23} .3Q{cx} -14.6 {cx-23} 1.1Z','#68677e')
    for j in range(3):
        x=cx+side*(21-j*2);y=-1.5-j*1.25
        features+=p(f'M{x} {y}Q{x+side*3.5} {y-1.6} {x+side*(6-j*.8)} {y-5.5-j*.8}Q{x+side*2} {y+1} {x-side} {y+1}Z','#737087',op=.8-j*.12)
    features+=p(f'M{cx-19} 4Q{cx} 17 {cx+20} 3',stroke='#c0a2b0',w=.4,op=.8)
    features+=p(f'M{cx-20} -10Q{cx} -23 {cx+19} -10',stroke='#d4b5bd',w=.35,op=.65)
    features+=p(f'M{cx-20} -24Q{cx-1} -28.3 {cx+20} -24Q{cx} -25.5 {cx-20} -24Z','#9d9cac',op=.55)
    for j in range(24):
        x=cx-18+j*1.5;y=-25-1.8*(1-((x-cx)/20)**2);dx=side*R.uniform(.8,1.5)
        features+=p(f'M{x} {y+.9}Q{x+dx*.4} {y} {x+dx} {y-.6}',stroke='#9299a9',w=.11,op=R.uniform(.28,.47))
features+=e(1,36,5,3.5,'url(#noseSoft)')+e(-1,31,2.2,5,'url(#noseLight)',.65)
features+=p('M-2 34Q1 38 4 35Q1 37 -2 34Z','#c89aa7',op=.38)
features+=e(.7,34.1,1.3,.65,'#fff5e5',.75)
features+=p('M-8 62Q1 60.8 10 62Q1 63.7 -8 62Z','#ba8e9f',op=.34)
features+=p('M-6 62Q1 61.5 8 62',stroke='#a47c91',w=.33,op=.72)
features+=p('M-2 65Q1 65.5 4 65',stroke='#fff4e7',w=.5,op=.5)
for n in ET.fromstring('<g>'+features+'</g>'):root.append(n)
face['svg']=ET.tostring(root,encoding='unicode')
all_defs=ET.tostring(defs,encoding='unicode')
def svg(body):return '<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="3000" viewBox="-240 -230 480 720">'+all_defs+body+'</svg>'
FACE='M-71 -65C-77 -17 -59 48 -28 74Q-6 92 7 95C53 72 74 24 72 -56C65 -111 -53 -120 -71 -65Z'
soft=e(-46,29,25,14,'url(#cheek)')+e(47,28,23,15,'url(#cheek)')+e(63,40,24,44,'url(#coolPlane)')
erase=[]
for cx in [-37,37]:erase.append(svg(p(f'M{cx-25} 0Q{cx} -19 {cx+25} -1Q{cx} 22 {cx-25} 0Z','#fff')))
parts=json.loads((OUT/'batch-plan.json').read_text(encoding='utf-8'))['structure_changes']
original_parts=json.loads(Path('data/test-result/20261017/krita-contextual-hair/elements.json').read_text(encoding='utf-8'))['parts']
for n in original_parts:
    if n['id'][0] in 'GHIJ':erase.append(svg(p(parts.get(n['id'],n['d']),'#fff')))
base={k:scene[k] for k in ['name','width','height','scale','details']}
one={**base,'stem':'05-frontal-face-planes','expected_document':str((OUT/'painting/04-frontal-gem-eyes.kra').resolve()),'layers':[face]}
(OUT/'05-face-integration.json').write_text(json.dumps(one),encoding='utf-8')
two={**base,'stem':'planned-head-reviewed','expected_document':str((OUT/'painting/05-frontal-face-planes.kra').resolve()),'layers':[
 {'name':'35 Frontal warm cheeks and cool side plane','svg':svg(soft),'svg_masks':[svg(p(FACE,'#fff'))],'svg_erase_masks':erase}]}
(OUT/'06-soft-planes.json').write_text(json.dumps(two),encoding='utf-8')
for update in one['layers']+two['layers']:
    prior=next((n for n in scene['layers'] if n['name']==update['name']),None)
    if prior is None:scene['layers'].append(copy.deepcopy(update))
    else:prior.update(copy.deepcopy(update))
scene['stem']='planned-head-snapshot'
(OUT/'final-scene-snapshot.json').write_text(json.dumps(scene),encoding='utf-8')
print('Prepared frontal facial planes, tapered eyelids, fine brows and restrained nose/lip forms')
