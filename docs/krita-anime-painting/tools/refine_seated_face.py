"""Three separately reviewed face/material refinements of the seated study."""
import ast
import copy
import json
import math
import random
from pathlib import Path

ROOT=Path('data/test-result/20261016/krita-seated-practice-v4')
OUT=ROOT/'face-refinement'
OUT.mkdir(exist_ok=True)
# Load geometry helpers without running the original drawing recipe.
tree=ast.parse(Path('docs/krita-anime-painting/tools/author_seated_practice.py').read_text(encoding='utf-8'))
names={'p','e','svg','hp','hd','smooth'}
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFS' for t in n.targets)]
exec(compile(ast.Module(body=selected,type_ignores=[]),'technique-helpers','exec'))
S=json.loads((ROOT/'scene.json').read_text(encoding='utf-8'))
byname={x['name']:x for x in S['layers']}
base={k:copy.deepcopy(S[k]) for k in ['name','width','height','scale','details']}
expected=(ROOT/'checked-stages/azure-quiet-page.kra').resolve()
def write(file,layers,stem,previous):
    v={**base,'layers':layers,'stem':stem,'expected_document':str(previous)}
    (OUT/file).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

# Step 1: explicit planes and low-opacity native soft brush transitions.
body='''<defs>
<radialGradient id="warm"><stop stop-color="#d994a1" stop-opacity=".32"/><stop offset="1" stop-color="#edb9bb" stop-opacity="0"/></radialGradient>
<radialGradient id="cool"><stop stop-color="#9c9fb8" stop-opacity=".25"/><stop offset="1" stop-color="#c9b1bd" stop-opacity="0"/></radialGradient>
<radialGradient id="cream"><stop stop-color="#fff9e9" stop-opacity=".75"/><stop offset="1" stop-color="#fff4e6" stop-opacity="0"/></radialGradient>
</defs>'''
for x,y,rx,ry,fill in [(-52,36,32,20,'warm'),(55,32,25,21,'warm'),(73,35,23,53,'cool'),(-24,9,35,47,'cream'),(0,33,7,13,'cream')]:
    xx,yy=hp(x,y);body+=e(xx,yy,rx,ry,'url(#'+fill+')')
body+=hd([['M',-74,-48],['Q',-54,-58,-34,-82],['Q',-54,-39,-74,-33],['Z']],'#afa6b8',opacity=.13)
body+=hd([['M',25,-86],['Q',53,-58,79,-56],['L',80,-47],['Q',46,-45,25,-86],['Z']],'#aaa0b3',opacity=.12)
strokes=[]
for x,y,size,col,op in [(-52,35,32,'#e8abb2',.12),(53,33,27,'#e8abb2',.1),(-26,10,39,'#fff6e6',.11),(5,39,10,'#fff5e4',.18)]:
    xx,yy=hp(x,y);strokes.append({'brush':'b) Airbrush Soft','size':size,'color':col,'opacity':op,'line':[xx,yy,xx+.3,yy+.3]})
skin={'name':'23 Refined cheek planes and native soft color','svg':svg(body),
      'svg_masks':[byname['11 Face and ear base aligned to head camera']['svg']],
      'svg_erase_masks':[byname['17 Rooted bangs and shoulder locks with fine fibers']['svg']],
      'strokes':strokes,'checkpoint':'01-skin-refinement.png'}
write('01-skin.json',[skin],'face-skin-pass',expected)

# Step 2 revises the actual iris layers, so no duplicate pupil or highlights.
eyes=[];R=random.Random(616)
for side in [-1,1]:
    cx=side*46;x,y=hp(cx-2,0)
    aperture=hd([['M',cx-28,0],['C',cx-15,-13,cx+12,-15,cx+28,-2],['Q',cx+9,17,cx-10,11],['Q',cx-21,8,cx-28,0],['Z']],'#ffffff')
    body='''<defs><linearGradient id="iris2" x1="0%" y1="0%" x2="0%" y2="100%"><stop stop-color="#394e6c"/><stop offset=".32" stop-color="#68849f"/><stop offset=".68" stop-color="#a8c5d2"/><stop offset="1" stop-color="#d0e2dc"/></linearGradient><linearGradient id="lidshade" x1="0%" y1="0%" x2="0%" y2="100%"><stop stop-color="#354a68" stop-opacity=".7"/><stop offset=".6" stop-color="#657c97" stop-opacity="0"/></linearGradient></defs>'''
    body+=e(x,y,13.9,17,'url(#iris2)',stroke='#71879c',w=.65)
    for j in range(38):
        a=j*math.tau/38;start=R.uniform(6,8);end=R.uniform(11,15)
        pts=[(x+math.cos(a+math.sin(t*math.pi)*.07)*(start+(end-start)*t)*.85,y+math.sin(a)*(start+(end-start)*t)) for t in [0,.25,.5,.75,1]]
        body+=p(smooth(pts),stroke=R.choice(['#54728c','#a2c4d2','#d5e3d9']),w=R.uniform(.25,.5),opacity=.45)
    body+=e(x,y-1,4.2,7,'#425975')
    body+=p(f'M{x-10} {y+7}Q{x-1} {y+16} {x+9} {y+8}',stroke='#d7e7df',w=1.2,opacity=.65)
    body+=e(x-4.5,y-7,3.2,2.5,'#fffaf0',.95)+e(x-2,y-8,1,1,'#ffffff')+e(x+6,y+9,1.1,1.3,'#f5faf0',.8)
    body+=p(f'M{x-16} {y-16}H{x+16}V{y+2}Q{x} {y-6} {x-16} {y+2}Z','url(#lidshade)')
    item=copy.deepcopy(byname[f'14 Eye {side} grey blue iris']);item['svg']=svg(body)
    item['svg_masks']=[svg(aperture),svg(e(x,y,14,17,'#fff'))]
    eyes.append(item)
write('02-eyes.json',eyes,'face-eye-pass',(OUT/'face-skin-pass.kra').resolve())

# Step 3: broad value shapes, not another increase in strand count.
hair=p('M489 183C465 210 463 240 413 286Q445 261 461 221Z','#557c9f',opacity=.24)
hair+=p('M498 185Q534 228 569 250L564 255Q522 240 498 185Z','#4f769a',opacity=.25)
hair+=p('M399 277C386 391 414 477 366 582Q402 477 394 391Z','#47769b',opacity=.22)
hair+=p('M579 285C599 397 562 481 598 568Q575 472 586 397Z','#4f779d',opacity=.24)
hair+=p('M428 221Q445 204 465 202M512 209Q529 222 541 224',stroke='#f7fcfa',w=5,opacity=.35)
hair+=p('M390 370Q395 401 395 425M589 373Q588 402 584 421',stroke='#f2fbfa',w=3,opacity=.25)
layer={'name':'24 Refined hair broad shadows and broken sheen','svg':svg(hair),
       'svg_masks':[byname['17 Rooted bangs and shoulder locks with fine fibers']['svg']],
       'checkpoint':'03-material-refinement.png'}
write('03-hair.json',[layer],'azure-face-refined',(OUT/'face-eye-pass.kra').resolve())
print('Prepared three independently reviewed refinement steps')
