"""Gem-like anime iris study and separately saved portrait revision.

Geometry stays in the established head projection. No reference pixels or
previous raster artwork are pasted into the drawing.
"""
import ast
import copy
import json
import math
import random
from pathlib import Path

ROOT=Path('data/test-result/20261016/krita-seated-practice-v4')
OUT=ROOT/'gem-eyes'
OUT.mkdir(exist_ok=True)
tree=ast.parse(Path('docs/krita-anime-painting/tools/author_seated_practice.py').read_text(encoding='utf-8'))
names={'p','e','svg','hp','hd','smooth'}
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFS' for t in n.targets)]
exec(compile(ast.Module(body=selected,type_ignores=[]),'head-technique-helpers','exec'))
S=json.loads((ROOT/'scene.json').read_text(encoding='utf-8'))
old={n['name']:n for n in S['layers']}
C={'schema':'krita-gem-eyes/1','palette':['#283e5b','#577f9f','#88bccd','#c4e5dc'],
   'iris_radius':[13.9,17],'pupil_radius':[3.7,6.5],
   'head_projection':'same hp() as the seated portrait; iris is projected too',
   'light_direction':'upper viewer left for both eyes','highlight_gain':.8,
   'facet_opacity':.26,'sparkle_count':6,
   'workflow':['construction','iris depth','facet/texture','reflection','face context check'],
   'checks':['same light direction','pupils remain legible','no highlights outside aperture',
             'no loss of upper eyelid shadow','gem effect visible at face viewing size'],
   'sources':['https://www.clipstudio.net/how-to-draw/archives/163257',
              'https://tips.clip-studio.com/en-us/articles/11782',
              'https://www.clip-studio.com/site/gd_en/csp/startupguide/csp_startup/CSPaint_02/Chapter02_5_1_3.htm']}
(OUT/'elements.json').write_text(json.dumps(C,ensure_ascii=False,indent=2),encoding='utf-8')

GDEFS='''<defs>
<linearGradient id="gemBase" x1="0%" y1="0%" x2="0%" y2="100%"><stop stop-color="#243951"/><stop offset=".25" stop-color="#496d91"/><stop offset=".55" stop-color="#77a8c0"/><stop offset=".82" stop-color="#a7d9dc"/><stop offset="1" stop-color="#dbebd9"/></linearGradient>
<radialGradient id="gemGlow" cx="50%" cy="80%" r="65%"><stop stop-color="#bdeee5" stop-opacity=".55"/><stop offset="1" stop-color="#b0dbdd" stop-opacity="0"/></radialGradient>
<linearGradient id="occlusion" x1="0%" y1="0%" x2="0%" y2="100%"><stop stop-color="#20364f" stop-opacity=".85"/><stop offset=".5" stop-color="#354c6c" stop-opacity=".15"/><stop offset="1" stop-color="#354c6c" stop-opacity="0"/></linearGradient>
<linearGradient id="wet" x1="0%" y1="0%" x2="100%" y2="100%"><stop stop-color="#fffdf1"/><stop offset="1" stop-color="#b8dce4"/></linearGradient>
</defs>'''
def esvg(body):return svg(GDEFS+body)
def eye(side):
    cx=side*46-2;R=random.Random(720+side)
    def draw(cmds,fill='none',stroke='none',w=1,opacity=1):
        return hd([[c[0]]+[v+(cx if j%2==0 else 0) for j,v in enumerate(c[1:])] for c in cmds],fill,stroke,w,opacity)
    def ellipse(x,y,rx,ry,fill,stroke='none',w=.4,opacity=1):
        k=.55228475
        return draw([['M',x+rx,y],['C',x+rx,y+k*ry,x+k*rx,y+ry,x,y+ry],['C',x-k*rx,y+ry,x-rx,y+k*ry,x-rx,y],['C',x-rx,y-k*ry,x-k*rx,y-ry,x,y-ry],['C',x+k*rx,y-ry,x+rx,y-k*ry,x+rx,y],['Z']],fill,stroke,w,opacity)
    def polygon(points,fill,op):
        return draw([['M',*points[0]]]+[['L',*v] for v in points[1:]]+[['Z']],fill,opacity=op)
    base=ellipse(0,0,13.9,17,'url(#gemBase)','#4e6c89',.75)
    base+=ellipse(0,1,12.4,15,'url(#gemGlow)')
    base+=ellipse(0,-1,3.7,6.5,'#2e435f','#5d829d',.45)
    texture=''
    # Uneven translucent facets imply refracted depth without turning the iris
    # into an opaque faceted stone. The darker pupil stays uncovered.
    for j in range(20):
        a=j*math.tau/20+.03;b=a+R.uniform(.18,.32)
        r0=R.uniform(.43,.59);r1=R.uniform(.89,.99)
        pts=[(math.cos(a)*13.4*r0,math.sin(a)*16.4*r0),
             (math.cos(a-.03)*13.4*r1,math.sin(a-.03)*16.4*r1),
             (math.cos(b)*13.4*r1,math.sin(b)*16.4*r1),
             (math.cos(b+.05)*13.4*(r0+.1),math.sin(b+.05)*16.4*(r0+.1))]
        col=R.choice(['#d8f2e6','#85b8d0','#5486a8','#c4e6e1']) if math.sin(a)>0 else R.choice(['#325875','#7babc5','#a1cbd7'])
        texture+=polygon(pts,col,C['facet_opacity']*R.uniform(.5,1.2))
    for j in range(62):
        a=j*math.tau/62;start=R.uniform(.42,.68);end=R.uniform(.79,.98)
        pts=[(math.cos(a+math.sin(t*math.pi)*.05)*13.1*(start+(end-start)*t),math.sin(a)*16.1*(start+(end-start)*t)) for t in [0,.25,.5,.75,1]]
        cmds=[['M',*pts[0]]]+[['Q',*pts[i],*pts[min(i+1,4)]] for i in [1,3]]
        texture+=draw(cmds,stroke=R.choice(['#bddfdc','#517e9e','#89b5c9']),w=R.uniform(.13,.3),opacity=R.uniform(.25,.6))
    texture+=draw([['M',-8,4],['Q',-6,11,0,12],['Q',7,12,10,6]],stroke='#cdebe0',w=.65,opacity=.65)
    texture+=draw([['M',-5,1],['Q',-6,8,0,9]],stroke='#426c8b',w=.6,opacity=.5)
    occlusion=ellipse(0,0,13.9,17,'url(#occlusion)')
    reflection=draw([['M',-9,-8],['Q',-7,-11,-4,-10],['L',-3,-5],['Q',-5,-2,-8,-4],['Z']],'url(#wet)',opacity=.91*C['highlight_gain'])
    reflection+=draw([['M',-3,-9],['L',0,-9],['L',1,-6],['L',-2,-5],['Z']],'#d3e7ed',opacity=.5*C['highlight_gain'])
    reflection+=draw([['M',-12,-3],['Q',-5,-6,5,-3]],stroke='#bddde9',w=.65,opacity=.24)
    for u,v,r in [(6,8,1.25),(8,6,.6),(-4,11,.65),(-9,5,.42),(3,12,.4),(10,1,.35)]:
        reflection+=polygon([(u,v-r),(u+r*.65,v),(u,v+r),(u-r*.65,v)],'#fff9dd',C['highlight_gain']*.83)
    reflection+=ellipse(6,8,.32,.42,'#ffffff')
    aperture=hd([['M',side*46-28,0],['C',side*46-15,-13,side*46+12,-15,side*46+28,-2],['Q',side*46+9,17,side*46-10,11],['Q',side*46-21,8,side*46-28,0],['Z']],'#fff')
    return {'side':side,'center':hp(cx,0),'base':base,'texture':texture,'occlusion':occlusion,
            'reflection':reflection,'mask':aperture,'iris_mask':ellipse(0,0,14,17,'#fff')}

E=[eye(-1),eye(1)]
def large(s,body):
    item=E[0 if s<0 else 1];x,y=item['center'];xx=285 if s<0 else 715
    return f'<g transform="translate({xx} 257) scale(4.7) translate({-x} {-y})">{body}</g>'
def board(body):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1440" viewBox="0 0 1000 600">'+GDEFS+body+'</svg>'
def text(x,y,t,size=14):return f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Segoe UI" font-size="{size}" fill="#708298">{t}</text>'
paper=p('M0 0H1000V600H0Z','#f7f5ef')+text(500,46,'AQUAMARINE  /  IRIS MATERIAL STUDY',19)+text(500,74,'Same head direction · upper-left light · legible pupils',11)
L=[{'name':'01 Warm study paper','svg':board(paper)}]
context='';guides=''
for ee in E:
    side=ee['side'];cx=side*46
    sk=hd([['M',cx-35,-30],['Q',cx,-43,cx+35,-30],['L',cx+37,23],['Q',cx,33,cx-35,23],['Z']],'#f6e2d8')
    sk+=ee['mask'].replace('#fff','#fbfaf4')
    sk+=hd([['M',cx-28,0],['C',cx-15,-13,cx+12,-15,cx+28,-2]],stroke='#62687b',w=1.8)
    sk+=hd([['M',cx-20,8],['Q',cx+4,20,cx+24,5]],stroke='#bc9aab',w=.7)
    sk+=hd([['M',cx-27,-23],['Q',cx,-31,cx+25,-24]],stroke='#9e9caa',w=.8)
    context+=large(side,sk)
    guides+=large(side,ee['iris_mask'].replace('fill="#fff"','fill="none"').replace('stroke="none"','stroke="#99aebd"'))
L+=[{'name':'02 Eye construction and lid context','svg':board(context+guides),'checkpoint':'01-construction.png'}]
for label,key in [('03 Iris depth','base'),('04 Small facets and iris fibers','texture'),('05 Upper eyelid occlusion','occlusion'),('06 Restrained reflections','reflection')]:
    for ee in E:
        body=large(ee['side'],ee[key]);masks=[board(large(ee['side'],ee['mask'])),board(large(ee['side'],ee['iris_mask']))]
        L.append({'name':label+str(ee['side']),'svg':board(body),'svg_masks':masks})
    L[-1]['checkpoint']={'base':'02-depth.png','texture':'03-facets.png','occlusion':'04-shadow.png','reflection':'05-reflections.png'}[key]
# Six small swatches reveal the material process without changing the iris design.
cards=''
for i,(label,keys) in enumerate([('DEPTH',['base']),('FACETS',['base','texture']),('SHADOW',['base','texture','occlusion']),('REFLECTION',['base','texture','occlusion','reflection'])]):
    xx=175+i*216;ee=E[0];x,y=ee['center']
    cards+=f'<g transform="translate({xx} 465) scale(2.05) translate({-x} {-y})">'+''.join(ee[k] for k in keys)+'</g>'+text(xx,529,label,11)
cards+=text(500,577,'Surface sparkle kept small so the gaze remains readable',11)
L.append({'name':'07 Material process swatches','svg':board(cards),'checkpoint':'06-study-complete.png'})
study={'name':'Aquamarine eye material study','width':2400,'height':1440,'scale':2.4,'stem':'gem-eye-study',
       'blank_checkpoint':'00-study-blank.kra','layers':L,'details':[]}
# Two execution sections force an intermediate review before painting reflections.
a=copy.deepcopy(study);a['layers']=L[:-4];a['stem']='gem-eye-underpainting'
# Ends after upper lid shadow, before the final two reflection layers + cards.
a['layers']=L[:-3]
(OUT/'study-underpainting.json').write_text(json.dumps(a),encoding='utf-8')
clean=copy.deepcopy(study)
clean.update(layers=[{'name':'02 Eye construction and lid context','svg':board(context),
                     'checkpoint':'04b-construction-lines-hidden.png'}],stem='gem-eye-underpainting',
             expected_document=str((OUT/'study/gem-eye-underpainting.kra').resolve()))
(OUT/'study-context-review.json').write_text(json.dumps(clean),encoding='utf-8')
b=copy.deepcopy(study);b['layers']=L[-3:];b['expected_document']=str((OUT/'study/gem-eye-underpainting.kra').resolve())
(OUT/'study-reflections.json').write_text(json.dumps(b),encoding='utf-8')

portrait=[]
for ee in E:
    n=copy.deepcopy(old[f"14 Eye {ee['side']} grey blue iris"])
    n['svg']=esvg(''.join(ee[k] for k in ['base','texture','occlusion','reflection']))
    n['svg_masks']=[svg(ee['mask']),svg(ee['iris_mask'])]
    portrait.append(n)
apply={k:copy.deepcopy(S[k]) for k in ['name','width','height','scale','details']}
apply.update(layers=portrait,stem='azure-gem-eyes',expected_document=str((ROOT/'face-refinement/azure-face-refined.kra').resolve()))
(OUT/'apply-to-portrait.json').write_text(json.dumps(apply),encoding='utf-8')
print('Prepared projected iris underpainting, reflection study and portrait revision')
