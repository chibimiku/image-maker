"""Execute the first planned batch against a frozen contextual hairstyle."""
import copy
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

SRC=Path('data/test-result/20261017/krita-contextual-hair')
OUT=Path('data/test-result/20261017/krita-planned-head-batch')
OUT.mkdir(exist_ok=True)
base=json.loads((SRC/'01-outline.json').read_text(encoding='utf-8'))
layers=copy.deepcopy(base['layers'])
for filename in ['02-values.json','03-fine.json','04-review.json','05-contact.json']:
    for n in json.loads((SRC/filename).read_text(encoding='utf-8'))['layers']:
        dest=next((a for a in layers if a['name']==n['name']),None)
        if dest is None:layers.append(copy.deepcopy(n))
        else:dest.update(copy.deepcopy(n))
parts=json.loads((SRC/'elements.json').read_text(encoding='utf-8'))['parts']
original={n['id']:n['d'] for n in parts}
changed={
 'A crown gathering':'M-48 -132C-61 -150 -55 -166 -38 -174C-23 -188 -7 -188 6 -184C29 -190 50 -177 54 -158Q55 -145 45 -133L23 -128Q-4 -136 -22 -127Z',
 'B2 outer long left':original['B2 outer long left'].replace('Q-176 455 -162 444C-124 410 -122 376 -147 349','Q-177 460 -164 446L-169 462Q-147 444 -140 424C-124 404 -125 374 -147 349')}
updates=[]
for n in layers:
    updated=copy.deepcopy(n);has_change=False
    for key in ['svg','svg_masks','svg_erase_masks']:
        seq=[updated[key]] if key=='svg' and key in updated else updated.get(key,[])
        target=[]
        for s in seq:
            for name,d in changed.items():
                if original[name] in s:s=s.replace(original[name],d);has_change=True
            target.append(s)
        if key=='svg' and target:updated[key]=target[0]
        elif key in updated:updated[key]=target
    if has_change:updates.append(updated)

def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="3000" viewBox="-240 -230 480 720">'+s+'</svg>'
def p(d,fill='none',stroke='none',w=1,opacity=1):return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{opacity}" stroke-linecap="round"/>'
def ellipse(x,y,rx,ry,fill,opacity=1):return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{opacity}"/>'
def scene(stem,previous,ll):return {**base,'name':'Planned head practice — shape, detail, integration','stem':stem,'expected_document':str(previous.resolve()),'layers':ll,'details':[]}
one=scene('01-structure-reviewed',SRC/'painting/hairstyle-context-reviewed.kra',updates)
(OUT/'01-structure.json').write_text(json.dumps(one),encoding='utf-8')

# The old face contains the iris and eyelid in one bitmap. Remove only the
# three iris ellipses per eye; preserve the whites, lids and all facial marks.
face=copy.deepcopy(next(n for n in layers if n['name']=='Face placement for hairstyle evaluation'))
r=ET.fromstring(face['svg'])
removed=0
for parent in r.iter():
    for child in list(parent):
        if child.tag.split('}')[-1]=='ellipse' and child.get('rx') in {'10','4','2.4'}:
            parent.remove(child);removed+=1
assert removed==6,removed
face['svg']=ET.tostring(r,encoding='unicode')
two=scene('02-eye-boundaries',OUT/'painting/01-structure-reviewed.kra',[face])
(OUT/'02-eye-boundaries.json').write_text(json.dumps(two),encoding='utf-8')

R=random.Random(171201)
DEFS='''<defs><linearGradient id="gem" x1="0%" y1="0%" x2="0%" y2="100%"><stop stop-color="#435a79"/><stop offset=".44" stop-color="#7ea5bf"/><stop offset=".8" stop-color="#b4d5d9"/><stop offset="1" stop-color="#d0e4d8"/></linearGradient><radialGradient id="irisLight"><stop stop-color="#e0f1e6" stop-opacity=".5"/><stop offset="1" stop-color="#9cbac9" stop-opacity="0"/></radialGradient></defs>'''
irises=[]
hair=[]
for n in parts:
    if n['id'][0] in 'GHIJ':hair.append(svg(p(changed.get(n['id'],n['d']),'#fff')))
for side in [-1,1]:
    cx=side*37;body=DEFS+ellipse(cx,0,10,12,'url(#gem)')
    body+=f'<ellipse cx="{cx}" cy="0" rx="9.7" ry="11.7" fill="none" stroke="#4c6887" stroke-width=".65" opacity=".7"/>'
    for j in range(78):
        a=2*math.pi*j/78+R.uniform(-.015,.015);rr=R.uniform(.52,.71);end=R.uniform(.85,.98)
        x0=cx+9.5*rr*math.cos(a);y0=11.5*rr*math.sin(a)
        x1=cx+9.5*end*math.cos(a+.018);y1=11.5*end*math.sin(a+.018)
        body+=p(f'M{x0} {y0}Q{(x0+x1)/2-.12} {(y0+y1)/2} {x1} {y1}',stroke=R.choice(['#d4e6e1','#6c91ae','#9cc1cc','#496c8d']),w=R.uniform(.08,.19),opacity=R.uniform(.25,.65))
    body+=ellipse(cx+1,4,7,7,'url(#irisLight)')
    body+=ellipse(cx,-1,4,7,'#344e6c')
    body+=f'<ellipse cx="{cx}" cy="-1" rx="4.7" ry="7.5" fill="none" stroke="#8daebf" stroke-width=".3" opacity=".7"/>'
    for j in range(7):
        a=math.pi*(.15+j*.11);x=cx+7.5*math.cos(a);y=9.4*math.sin(a)
        body+=p(f'M{x} {y-1}L{x+.55} {y}L{x-.05} {y+1}L{x-.5} {y}Z','#e5f0df',opacity=.65)
    body+=ellipse(cx-3,-5.2,2.3,2.9,'#eff7ee',.88)+ellipse(cx+.9,-7,.9,1.3,'#eff7ee',.6)
    opening=f'M{cx-22} 0Q{cx} -15.4 {cx+22} -.8Q{cx} 18 {cx-22} 0Z'
    irises.append({'name':f'34 Frontal gemstone iris {side}','svg':svg(body),'svg_masks':[svg(p(opening,'#fff'))], 'svg_erase_masks':hair})
three=scene('03-frontal-gem-eyes',OUT/'painting/02-eye-boundaries.kra',irises)
(OUT/'03-iris-integration.json').write_text(json.dumps(three),encoding='utf-8')

merged=copy.deepcopy(layers)
for stage in [one,two,three]:
    for n in stage['layers']:
        old=next((x for x in merged if x['name']==n['name']),None)
        if old is None:merged.append(copy.deepcopy(n))
        else:old.update(copy.deepcopy(n))
snapshot={**base,'name':'Planned head batch snapshot','stem':'planned-head-snapshot','layers':merged}
snapshot.pop('expected_document',None)
(OUT/'final-scene-snapshot.json').write_text(json.dumps(snapshot),encoding='utf-8')
(OUT/'batch-plan.json').write_text(json.dumps({'targets':['less regular crown contour and attachment','unequal small fork in one rear tip','clip irises to actual eye openings','test detailed frontal irises without importing a different camera angle'],'previous_portrait_preserved':True,'structure_changes':changed,'limits':['provisional crown structure','frontal study face not full anatomical integration','image quality requires visual inspection']},indent=2),encoding='utf-8')
print('Prepared first planned batch: structure, eye boundary repair, frontal iris integration and replay snapshot')
