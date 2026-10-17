"""Visual-review corrections: soften broad planes, keep selected sharp edges."""
import copy
import json
from pathlib import Path
from author_hair_material import OUT,DEFS,R,path,svg,hp,FRONT,BACK

base=json.loads((OUT/'01-construction.json').read_text(encoding='utf-8'))
empty=svg('')
update=[{'name':'17 Rooted bangs and shoulder locks with fine fibers',
         'svg':svg(path(FRONT,'url(#newFront)','#668cab',.55))},
        {'name':'05 Projected chair with cushion and four legs','visible':False}]
# Use the actually reviewed tapered eyelid revision, rather than an earlier
# serialized close-up recipe that still contains the rod-like lashes.
eyes=json.loads(Path('data/test-result/20261016/krita-seated-practice-v4/head-practice/08-eyelids-closeup.json').read_text(encoding='utf-8'))
update.append(next(n for n in eyes['layers'] if n['name'].startswith('15 Almond')))
# Reflection and form shade share a blurred layer. Occlusion creases will be
# restored above it with sharper, narrow paths.
tone=''
for d,fill,op in [
 ('M24 -148C17 -95 -11 -56 -65 -24Q-37 -56 -14 -101Q-1 -131 -5 -149Z','#507da0',.38),
 ('M55 -141C103 -113 105 -67 96 13L88 -14C89 -75 75 -110 55 -141Z','#42698c',.37),
 ('M-96 -83C-111 -20 -90 49 -96 119L-90 155C-77 103 -79 77 -82 51C-93 -17 -82 -57 -72 -94Z','#5c87a7',.32),
 ('M-82 -111C-57 -139 -20 -148 5 -143Q-14 -116 -30 -100Q-55 -78 -77 -63Z','url(#silkLight)',1),
 ('M41 -132C61 -125 78 -101 83 -77L74 -63Q64 -106 41 -132Z','url(#silkLight)',.7),
 ('M-94 4C-94 42 -87 85 -96 130L-84 104Q-76 58 -84 -2Z','url(#silkLight)',.65)]:
    tone+=path(d,fill,op=op)
update.append({'name':'27 Grouped silk reflections','svg':svg(tone),'svg_masks':[svg(path(FRONT,'#fff'))],
               'filters':[{'name':'gaussian blur','settings':{'horizRadius':16,'vertRadius':16,'lockAspect':True}}]})
# Fine lines cover different sublocks with longer quiet intervals and varied
# lengths. Their contrast stays below the iris and the outer hair silhouette.
fibers=''
for j in range(170):
    u=R.random();ox=-63+u*88;oy=-139+18*(1-u)
    ex=-91+u*30;ey=-19+u*13
    if j%4==0:ex=-95+u*13;ey=R.uniform(45,142)
    d=f'M{ox} {oy}C{ox-30} {oy+18} {ex-9} {ey-29} {ex} {ey}'
    fibers+=path(d,stroke=R.choice(['#e5f3f1','#5c87a5','#7ca4bf']),w=R.uniform(.1,.21),op=R.uniform(.12,.27))
for j in range(58):
    u=R.random();sx=20+u*30;sy=-145+u*12;ex=78+u*17;ey=-47+u*47
    fibers+=path(f'M{sx} {sy}C{sx+28} {sy+13} {ex+4} {ey-41} {ex} {ey}',stroke=R.choice(['#e0efed','#4d799c','#88adc6']),w=R.uniform(.1,.2),op=R.uniform(.14,.29))
update.append({'name':'28 Selective fibers following connected masses','svg':svg(fibers),'svg_masks':[svg(path(FRONT,'#fff'))]})
recipe={**base,'stem':'azure-hair-reviewed','expected_document':str((OUT/'painting/azure-hair-material.kra').resolve()),'layers':update}
(OUT/'05-review-soft-planes.json').write_text(json.dumps(recipe),encoding='utf-8')

edges=''
# An asymmetric fork at the parting, a few interlocking tapering planes, and
# only short bright accents. None makes a continuous band across the crown.
for d,fill,op in [
 ('M23 -147C7 -104 -19 -65 -60 -29C-33 -58 -3 -113 2 -148Z','#719cb8',.3),
 ('M14 -142C-4 -95 -21 -75 -39 -61Q-15 -94 -5 -120Z','#cce3e9',.3),
 ('M-18 -141C-50 -122 -80 -88 -84 -48Q-75 -94 -45 -118Z','#729db8',.25),
 ('M-47 -129C-70 -112 -93 -69 -93 -37Q-84 -88 -61 -111Z','#d5e9ed',.32),
 ('M31 -144C46 -135 64 -116 76 -87Q57 -124 37 -130Z','#c4e0e7',.38),
 ('M50 -139C79 -118 88 -84 90 -49C94 -85 81 -125 50 -139Z','#4e7b9e',.28),
 ('M-85 22C-88 76 -83 112 -96 164Q-79 124 -80 89Z','#a9cedf',.5)]:edges+=path(d,fill,op=op)
for d in ['M-64 -115Q-77 -103 -80 -90','M-48 -127Q-56 -122 -61 -117','M-36 -130Q-40 -128 -44 -125',
          'M-7 -126Q-13 -110 -20 -100','M-24 -93Q-30 -82 -35 -76','M54 -121Q64 -111 69 -102',
          'M-87 37Q-87 53 -85 62','M-85 74Q-84 86 -86 99']:
    edges+=path(d,stroke='#e9f6f0',w=.55,op=.48)
strokes=[]
for d in [[['M',-96,-78],['C',-111,-37,-105,15,-112,44]],
          [['M',-83,-113],['C',-109,-98,-122,-60,-116,-31]],
          [['M',61,-133],['C',91,-123,112,-86,107,-55]]]:
    pp=[]
    for cmd in d:
        out=[cmd[0]]
        for i in range(1,len(cmd),2):
            x,y=hp(cmd[i],cmd[i+1]);out.extend([x-350,y-150])
        pp.append(out)
    strokes.append({'brush':'b) Basic-5 Size Opacity','color':'#779cb5','size':.35,'opacity':.45,'path':pp})
layer={'name':'29 Selective sharp glints and silhouette flyaways','svg':svg(edges),'svg_masks':[svg(path(FRONT,'#fff'))],'strokes':strokes}
final={**base,'stem':'azure-hair-final','expected_document':str((OUT/'painting/azure-hair-reviewed.kra').resolve()),'layers':[layer]}
(OUT/'06-review-sharp-edges.json').write_text(json.dumps(final),encoding='utf-8')
print('Prepared softer planes, actual reviewed eyelids, selective sharp glints and three native brush flyaways')
