"""Apply three specific lessons: neck form, cast/form shadow, ear placement."""
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

OUT=Path('data/test-result/20261017/krita-tutorial-volume')
OUT.mkdir(exist_ok=True)
scene=json.loads(Path('data/test-result/20261017/krita-planned-head-batch/final-scene-snapshot.json').read_text(encoding='utf-8'))
by={n['name']:n for n in scene['layers']}
def p(d,fill='none',stroke='none',w=1,op=1):return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'
def e(x,y,rx,ry,fill,op=1):return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{op}"/>'
DEFS='''<defs>
<linearGradient id="shoulderSkin" gradientUnits="userSpaceOnUse" x1="-85" y1="110" x2="101" y2="198"><stop stop-color="#ffefdf"/><stop offset=".48" stop-color="#f5dcd0"/><stop offset="1" stop-color="#dfb7ba"/></linearGradient>
<linearGradient id="neckForm" x1="0%" y1="0%" x2="100%" y2="15%"><stop stop-color="#ad89a0" stop-opacity="0"/><stop offset=".55" stop-color="#b78da0" stop-opacity=".16"/><stop offset="1" stop-color="#9a829e" stop-opacity=".4"/></linearGradient>
<linearGradient id="jawCast" x1="0%" y1="0%" x2="15%" y2="100%"><stop stop-color="#9b7896" stop-opacity=".45"/><stop offset=".48" stop-color="#ba8d9f" stop-opacity=".3"/><stop offset="1" stop-color="#d9a9b1" stop-opacity="0"/></linearGradient>
<radialGradient id="pit"><stop stop-color="#b08c9b" stop-opacity=".36"/><stop offset="1" stop-color="#d4afb3" stop-opacity="0"/></radialGradient>
<radialGradient id="skinLight"><stop stop-color="#fff6e4" stop-opacity=".68"/><stop offset="1" stop-color="#fff1df" stop-opacity="0"/></radialGradient>
<radialGradient id="earBowl"><stop stop-color="#b6879e" stop-opacity=".48"/><stop offset="1" stop-color="#deb1b7" stop-opacity=".1"/></radialGradient>
<linearGradient id="cheekTurn" x1="0%" y1="0%" x2="100%" y2="10%"><stop stop-color="#b193ad" stop-opacity="0"/><stop offset=".55" stop-color="#a28ba9" stop-opacity=".1"/><stop offset="1" stop-color="#a087a5" stop-opacity=".27"/></linearGradient>
</defs>'''
def svg(body):return '<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="3000" viewBox="-240 -230 480 720">'+DEFS+body+'</svg>'
NECK='M-29 71L-28 105Q-31 117 -56 129C-84 137 -112 153 -124 179Q-115 203 -93 216L95 216Q118 204 125 181C114 155 89 140 57 130Q29 119 28 105L29 71Z'
SHIRT='M-101 166Q-78 151 -56 151Q-24 176 3 178Q34 176 57 151Q81 153 104 168L111 216L-110 216Z'
EAR='M71 -13C86 -23 94 -7 89 8Q86 19 83 28C81 39 74 42 69 35Q67 29 71 20L70 6Q65 -5 71 -13Z'
oldface=copy.deepcopy(by['Face placement for hairstyle evaluation'])
r=ET.fromstring(oldface['svg'])
paths=[n for n in list(r) if n.tag.split('}')[-1]=='path']
assert paths[0].get('d').startswith('M-29 75')
assert paths[1].get('d').startswith('M-32 126')
necknode,shirtnode=paths[:2]
necknode.set('d',NECK);necknode.set('fill','url(#shoulderSkin)');necknode.set('stroke','#bd9da9');necknode.set('stroke-width','.55')
shirtnode.set('d',SHIRT);shirtnode.set('fill','#e0e6eb');shirtnode.set('stroke','#a8b3c2');shirtnode.set('stroke-width','.6')
defs=next(n for n in r if n.tag.split('}')[-1]=='defs')
for n in ET.fromstring(DEFS):defs.append(n)
oldface['svg']=ET.tostring(r,encoding='unicode')

# Expose only a lower portion of the ear; avoid inventing a complete braid
# mechanism behind it. The former gathered shape had covered the whole ear.
j=copy.deepcopy(by['J right gathered section'])
old_j='M82 -83C101 -69 95 -42 87 -30Q105 -15 88 1Q105 20 79 35L73 19Q87 8 75 -4Q91 -17 78 -31Q91 -48 78 -61Z'
new_j='M82 -83C101 -69 95 -42 87 -30Q103 -15 89 -3Q96 7 83 16L75 7Q88 -4 77 -16Q91 -30 78 -43Q90 -55 78 -61Z'
def substitute(item,a,b):
    item=copy.deepcopy(item)
    for k in ['svg','svg_masks','svg_erase_masks']:
        if k in item:item[k]=item[k].replace(a,b) if k=='svg' else [x.replace(a,b) for x in item[k]]
    return item
j=substitute(j,old_j,new_j)
initial=[oldface,j]
# Every layer that actually uses the old front silhouette must use the new
# one too, especially the irises and contact shadows.
for n in scene['layers']:
    if n['name'] not in {oldface['name'],j['name']} and old_j in json.dumps(n):initial.append(substitute(n,old_j,new_j))
for n in initial:by[n['name']]=n

guides=p('M-23 83Q0 94 24 82M-26 105Q0 117 26 105M-28 110Q-5 122 2 137M28 110Q9 127 2 137M-58 134Q-29 133 0 142Q32 134 58 136',stroke='#b58d98',w=.6)
guides+=p('M-72 -25L95 -25M-69 35L94 35',stroke='#c1afb8',w=.45,op=.6)+p(EAR,stroke='#a78091',w=.7)
base={k:scene[k] for k in ['width','height','scale','details']}
one={**base,'name':'Tutorial-led volume — neck, ear and light','stem':'01-volume-construction','blank_checkpoint':'00-blank.kra','layers':copy.deepcopy(scene['layers'])}
for n in initial:
    dest=next(x for x in one['layers'] if x['name']==n['name']);dest.update(copy.deepcopy(n))
one['layers'].append({'name':'36 Tutorial neck cylinder and ear landmarks','svg':svg(guides)})
(OUT/'01-construction.json').write_text(json.dumps(one),encoding='utf-8')

neckshade=p('M17 79Q31 94 26 108Q32 121 59 131L78 145Q43 136 29 133C12 120 15 97 17 79Z','url(#neckForm)')
neckshade+=p('M-29 76Q-7 100 8 97Q24 87 30 76L28 96Q11 119 -29 103Z','url(#jawCast)')
neckshade+=p('M-26 104Q-9 122 2 140Q-7 121 -19 112Z','#b98f9e',op=.15)
neckshade+=p('M27 105Q14 128 2 140Q21 124 31 113Z','#a98ba0',op=.15)
neckshade+=e(2,140,10,4.5,'url(#pit)')+e(-20,119,18,21,'url(#skinLight)',.6)
neckshade+=p('M-61 137Q-34 133 -7 143Q-35 138 -61 137Z','#b995a0',op=.22)
neckshade+=p('M10 143Q36 133 62 138Q37 136 10 143Z','#ad8a9c',op=.23)
neckshade+=p('M-57 135Q-31 132 -11 140',stroke='#fff1de',w=1.25,op=.48)
neckshade+=p('M16 140Q38 132 60 136',stroke='#f9e6d8',w=1,op=.42)
allhair=[]
parts=json.loads(Path('data/test-result/20261017/krita-contextual-hair/elements.json').read_text(encoding='utf-8'))['parts']
changes=json.loads(Path('data/test-result/20261017/krita-planned-head-batch/batch-plan.json').read_text(encoding='utf-8'))['structure_changes']
for n in parts:allhair.append(svg(p(new_j if n['id'].startswith('J ') else changes.get(n['id'],n['d']),'#fff')))
new_ear=p(EAR,'url(#shoulderSkin)','#b18b9e',.55)
new_ear+=e(79,12,5.5,10,'url(#earBowl)')
new_ear+=p('M73 -9C84 -14 88 -4 84 7Q78 16 77 30',stroke='#c299ab',w=.8)
new_ear+=p('M74 -8C84 -13 86 -3 83 4',stroke='#fff0df',w=1,op=.85)
new_ear+=p('M75 28Q83 18 77 10Q73 5 77 -1M77 10Q84 6 83 1',stroke='#b18c9f',w=.6,op=.75)
new_ear+=p('M70 12Q77 9 76 16Q74 21 70 20Z','#ebc6c3')
two={**base,'name':one['name'],'stem':'02-neck-and-ear-volume','expected_document':str((OUT/'painting/01-volume-construction.kra').resolve()),'layers':[
 {'name':'37 Neck cylinder form and jaw cast shadow','visibility':{'36 Tutorial neck cylinder and ear landmarks':False},'svg':svg(neckshade),'svg_masks':[svg(p(NECK,'#fff'))], 'svg_erase_masks':[svg(p('M-71 -65C-77 -17 -59 48 -28 74Q-6 92 7 95C53 72 74 24 72 -56C65 -111 -53 -120 -71 -65Z','#fff')),svg(p(SHIRT,'#fff'))]+allhair},
 {'name':'38 Right ear helix bowl and lobe','svg':svg(new_ear),'svg_erase_masks':[svg(p(new_j,'#fff'))]}]}
(OUT/'02-neck-ear.json').write_text(json.dumps(two),encoding='utf-8')

FACE='M-71 -65C-77 -17 -59 48 -28 74Q-6 92 7 95C53 72 74 24 72 -56C65 -111 -53 -120 -71 -65Z'
facelight=p('M54 -35C74 -9 70 23 57 48Q37 80 7 95L25 75Q47 45 46 20Q58 -9 54 -35Z','url(#cheekTurn)')
facelight+=p('M2 26Q8 33 6 39Q3 38 1 35Z','#b990a4',op=.13)
facelight+=e(-8,14,17,35,'url(#skinLight)',.25)
face_erase=[allhair[i] for i,n in enumerate(parts) if n['id'][0] in 'GHIJ']
for cx in [-37,37]:face_erase.append(svg(p(f'M{cx-26} 0Q{cx} -21 {cx+26} -1Q{cx} 24 {cx-26} 0Z','#fff')))
three={**base,'name':one['name'],'stem':'tutorial-volume-reviewed','expected_document':str((OUT/'painting/02-neck-and-ear-volume.kra').resolve()),'layers':[
 {'name':'39 Unified cheek turn and small nose cast','svg':svg(facelight),'svg_masks':[svg(p(FACE,'#fff'))],'svg_erase_masks':face_erase}]}
(OUT/'03-lighting.json').write_text(json.dumps(three),encoding='utf-8')
(OUT/'lesson-map.json').write_text(json.dumps({'camera':'near frontal, no head tilt change','light':'broad source above and to viewer left','applications':[{'source':'https://www.proko.com/course-lesson/how-to-draw-neck-muscles-form/','read':'public lesson notes','applied':'neck cylinder connects jaw underside to clavicles, broad neck-to-shoulder contour, restrained SCM gestures'},{'source':'https://www.clipstudio.net/how-to-draw/archives/158015','read':'public tutorial text','applied':'separate soft turning shade from jaw cast shadow, consistent upper-left light, avoid bright rim around everything'},{'source':'https://www.clipstudio.net/how-to-draw/archives/160418','read':'public tutorial text','applied':'ear near brow-to-nose region, simplified helix and Y antihelix, visible lobe behind gathered hair'}],'checks':['jaw casts on neck, not over face','neck shadow wraps into side plane','no black muscle cords in relaxed anime neck','ear partly occluded by gathered hair','eyes remain untouched by cheek shading','shirt remains a placeholder, not final costume'],'limits':['near-frontal view does not verify side-view neck tilt','exact braid construction remains provisional','no video viewing or traced tutorial pixels']},indent=2),encoding='utf-8')
print('Prepared tutorial-led construction, neck/ear volume, then coherent facial light')
