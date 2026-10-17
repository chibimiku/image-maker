"""Visual corrections and limited native-brush finish for hair study III."""
import copy
import json
import xml.etree.ElementTree as ET
from author_hair_quality_pass import OUT,OLD,BASE,by,svg,path,hp,FRONT,BACK,mask,write

orn=ET.fromstring(by['18 Embroidered hem ribbon and small jewelry']['svg'])
for x in orn.iter():
    if x.tag.split('}')[-1] in {'path','ellipse','circle','polygon','rect'}:
        x.set('opacity','1')
        if x.get('fill','none')!='none':x.set('fill','#fff')
        if x.get('stroke','none')!='none':x.set('stroke','#fff')
opaque_ornament=ET.tostring(orn,encoding='unicode')
current=json.loads((OUT/'03-material.json').read_text(encoding='utf-8'))
updates=copy.deepcopy(current['layers'])
for n in updates:n['svg_erase_masks']=[opaque_ornament]
# Bright accents stay subordinate to iris glints at thumbnail size.
updates[-1]['opacity']=.6

rear=''
for d,fill,op in [
 ('M-102 -30C-113 32 -96 79 -103 122C-111 160 -111 204 -109 234Q-99 201 -96 169C-85 112 -99 59 -91 -22Z','#b0d0df',.22),
 ('M-116 102C-120 153 -111 183 -119 230L-126 261Q-109 238 -106 212C-106 172 -117 149 -108 113Z','#3d6a90',.2),
 ('M99 -20C109 31 98 73 106 120Q117 164 112 202C122 160 117 144 115 124C106 78 121 35 111 -17Z','#b0d1df',.22),
 ('M91 50C87 93 103 133 105 170Q112 207 104 244C117 213 114 184 112 164C107 110 96 94 103 52Z','#345e87',.17)]:
    rear+=path(d,fill,op=op)
for side in [-1,1]:
    for j in range(22):
        u=j/21;x=side*(96+u*15)
        rear+=path(f'M{x} {5+u*33}C{x+side*7} {68+u*26} {x-side*5} {119+u*21} {x+side*2} {182+u*34}',stroke='#b2d1df' if j%3 else '#4f799c',w=.15,op=.23)
updates.append({'name':'19b Fine fibers for 06 Long rear hair grouped by flow','svg':svg(rear),'visible':True,
 'svg_masks':[svg(path(BACK,'#fff'))],'svg_erase_masks':[by[n]['svg'] for n in ['11 Face and ear base aligned to head camera','08 Neck shoulders and cast shadow','09 Seated dress silhouette and tension folds','10 Puff sleeves hands and a quiet book']]+[mask,opaque_ornament]})
# Keep the neck's silhouette and cast-shadow position, but let the shadow
# dissolve towards its lower edge rather than looking like a flat stripe.
neck=copy.deepcopy(by['08 Neck shoulders and cast shadow']);r=ET.fromstring(neck['svg'])
defs=next(n for n in r if n.tag.split('}')[-1]=='defs')
gradient=ET.fromstring('<linearGradient id="neckPenumbra" x1="0%" y1="0%" x2="10%" y2="100%"><stop stop-color="#bd929f" stop-opacity=".6"/><stop offset=".55" stop-color="#d7a7af" stop-opacity=".4"/><stop offset="1" stop-color="#dfb4b5" stop-opacity="0"/></linearGradient>')
defs.append(gradient)
shapes=[n for n in r.iter() if n.tag.split('}')[-1]=='path'];shapes[-1].set('fill','url(#neckPenumbra)');shapes[-1].set('opacity','1')
neck['svg']=ET.tostring(r,encoding='unicode');updates.append(neck)
write('04-occlusion-and-rear.json','04-hair-reviewed',OUT/'painting/03-hair-material.kra',updates)

strokes=[]
for brush,color,size,opacity,commands in [
 ('f) Bristles-4 Glaze','#c5dfe7',1.65,.17,[['M',-43,-127],['C',-63,-116,-82,-95,-87,-69]]),
 ('f) Bristles-4 Glaze','#6c97b6',1.2,.18,[['M',11,-135],['C',1,-113,-12,-83,-26,-64]]),
 ('f) Bristles-4 Glaze','#c4dfe6',1.4,.16,[['M',44,-134],['C',64,-121,83,-90,87,-59]]),
 ('f) Bristles-4 Glaze','#d2e5ea',1.5,.14,[['M',-85,18],['C',-85,41,-79,65,-82,91]]),
 ('b) Airbrush Soft','#e4efdf',8,.09,[['M',-58,-109],['C',-63,-105,-69,-95,-72,-84]])]:
    pp=[]
    for cmd in commands:
        a=[cmd[0]]
        for i in range(1,len(cmd),2):x,y=hp(cmd[i],cmd[i+1]);a.extend([x-350,y-150])
        pp.append(a)
    strokes.append({'brush':brush,'color':color,'size':size,'opacity':opacity,'path':pp,'taper':True,'samples':32})
write('05-native-finish.json','azure-hair-quality-reviewed',OUT/'painting/04-hair-reviewed.kra',[
 {'name':'31 Native glaze and restrained soft window reflection','svg':svg(''),'strokes':strokes}])
print('Prepared rear depth, opaque ornament occlusion, softer neck cast shade and two native brushes')
