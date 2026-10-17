"""Successive head-part studies using the saved projected portrait geometry."""
import ast
import copy
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path('data/test-result/20261016/krita-seated-practice-v4')
OUT=ROOT/'head-practice'
OUT.mkdir(exist_ok=True)
tree=ast.parse(Path('docs/krita-anime-painting/tools/author_seated_practice.py').read_text(encoding='utf-8'))
names={'p','e','svg','hp','hd','smooth','bez','strand'}
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFS' for t in n.targets)]
exec(compile(ast.Module(body=selected,type_ignores=[]),'projected-head-helpers','exec'))
S=json.loads((ROOT/'scene.json').read_text(encoding='utf-8'));OLD={n['name']:n for n in S['layers']}
R=random.Random(101622)
BASE={k:copy.deepcopy(S[k]) for k in ['name','width','height','scale']}
DETAILS=[{'file':'head.png','rect':[355,155,275,285]},
         {'file':'brows.png','rect':[399,265,180,36]},
         {'file':'nose.png','rect':[473,329,42,36]},
         {'file':'mouth.png','rect':[470,359,48,33]},
         {'file':'ear-context.png','rect':[548,294,61,99]}]
PREVIOUS=(ROOT/'gem-eyes/portrait/azure-gem-eyes.kra').resolve()
def recipe(stage,layers):
    global PREVIOUS
    stem='head-'+stage
    sc={**BASE,'layers':layers,'stem':stem,'expected_document':str(PREVIOUS),'details':DETAILS}
    (OUT/(stage+'.json')).write_text(json.dumps(sc,ensure_ascii=False,indent=2),encoding='utf-8')
    PREVIOUS=(OUT/stage/(stem+'.kra')).resolve()
def item(name,body):return {'name':name,'svg':svg(body)}
def ellipse_head(cx,cy,rx,ry,fill,opacity=1):
    k=.55228475
    return hd([['M',cx+rx,cy],['C',cx+rx,cy+k*ry,cx+k*rx,cy+ry,cx,cy+ry],['C',cx-k*rx,cy+ry,cx-rx,cy+k*ry,cx-rx,cy],['C',cx-rx,cy-k*ry,cx-k*rx,cy-ry,cx,cy-ry],['C',cx+k*rx,cy-ry,cx+rx,cy-k*ry,cx+rx,cy],['Z']],fill,opacity=opacity)

brows=''
for side in [-1,1]:
    cx=side*46
    brows+=hd([['M',cx-27,-23],['C',cx-12,-29,cx+7,-30,cx+25,-24],['Q',cx+4,-26,cx-11,-25],['Q',cx-22,-23,cx-27,-23],['Z']],'#8997a6',opacity=.44)
    for j in range(37):
        u=-23+(j+R.uniform(-.4,.4))*44/36;yy=-24-3.8*(1-(u/26)**2)
        length=R.uniform(1.2,3.2)*(1-.35*abs(u)/26);direction=side
        brows+=hd([['M',cx+u,yy+R.uniform(.8,1.6)],['Q',cx+u+direction*.8,yy+.5,cx+u+direction*length,yy-.2]],stroke=R.choice(['#8f99a5','#a2a7ae','#b1adb0']),w=R.uniform(.12,.22),opacity=R.uniform(.25,.45))
recipe('01-brows',[item('12 Fine low arched eyebrows',brows)])

# Nose and mouth share a historical layer. Each revision preserves the other
# feature until it has been separately evaluated and painted.
old_features=OLD['16 Short nose mark small parted mouth and blush']['svg']
root=ET.fromstring(old_features)
shapes=[x for x in root.iter() if x.tag.split('}')[-1] in ['path','ellipse']]
def xml(n):
    q=copy.deepcopy(n);q.tag=q.tag.split('}')[-1];return ET.tostring(q,encoding='unicode')
mouth_old=''.join(xml(n) for n in shapes[2:])
x,y=hp(0,39,67)
NDEFS='''<defs><radialGradient id="noseWarm"><stop stop-color="#d7a1ab" stop-opacity=".36"/><stop offset="1" stop-color="#edc1c0" stop-opacity="0"/></radialGradient><radialGradient id="noseLight"><stop stop-color="#fff8e8" stop-opacity=".8"/><stop offset="1" stop-color="#fff4e7" stop-opacity="0"/></radialGradient><radialGradient id="lipTint"><stop stop-color="#d5a0ac" stop-opacity=".26"/><stop offset="1" stop-color="#eac0c3" stop-opacity="0"/></radialGradient></defs>'''
nose=NDEFS+e(x+1,y+1.7,6,4.5,'url(#noseWarm)')
nose+=e(x-1.5,y-4,3.5,7,'url(#noseLight)',.7)
nose+=p(f'M{x-1} {y-7}Q{x} {y-1} {x-3} {y+2}Q{x} {y+6} {x+5} {y+1}Q{x+3} {y+4} {x-1} {y+4}Q{x-5} {y+2} {x-1} {y-7}Z','#d4a2aa',opacity=.24)
nose+=p(f'M{x-2} {y+3}Q{x} {y+4} {x+2} {y+2.8}',stroke='#bc949a',w=.5,opacity=.75)
nose+=e(x+.9,y+3.2,.45,.27,'#ba919a',.4)+p(f'M{x-1.5} {y-2}Q{x+.5} {y-4} {x+2} {y-2}Q{x+.5} {y-.3} {x-1.5} {y-2}Z','#fff8eb',opacity=.75)
recipe('02-nose',[item('16 Short nose mark small parted mouth and blush',nose+mouth_old)])

cx,cy=hp(0,65)
mouth=ellipse_head(0,67,9,2.8,'url(#lipTint)')
mouth+=hd([['M',-10.5,65],['Q',-3,63.5,0,64.1],['Q',4,63.7,10.5,64.6],['Q',4,66,0,65.7],['Q',-4,66.2,-10.5,65],['Z']],'#bd8f9e',opacity=.32)
mouth+=hd([['M',-8,65],['Q',0,64.2,8,64.7]],stroke='#997285',w=.38,opacity=.65)
mouth+=hd([['M',-3,68],['Q',1,68.5,4.5,68]],stroke='#fff5e7',w=.5,opacity=.3)
blush=''
for xx in [-57,60]:
    bx,by=hp(xx,38);blush+=e(bx,by,24,13,'url(#blush)',.55)
recipe('03-mouth',[item('16 Short nose mark small parted mouth and blush',nose+mouth+blush)])

# A cartilage shape in normalized ear space is reused for the isolated study
# and the small exposed parts of the portrait. The small portrait ears remain
# behind the original front locks; the view does not invent hidden anatomy.
EDEFS='''<defs><linearGradient id="earSkin" x1="0%" y1="0%" x2="100%" y2="70%"><stop stop-color="#fff0e3"/><stop offset=".6" stop-color="#efd1c8"/><stop offset="1" stop-color="#d6aab2"/></linearGradient><radialGradient id="concha"><stop stop-color="#bb8d9f" stop-opacity=".6"/><stop offset="1" stop-color="#ddb0b4" stop-opacity=".08"/></radialGradient></defs>'''
EAR=[('M22 12C68 -16 106 18 96 68C91 96 78 111 74 143C70 175 54 193 30 185C14 180 9 166 16 146L22 120C4 99 -1 75 3 50C6 30 12 19 22 12Z','url(#earSkin)','#bc939e',1.15,1),
 ('M35 88Q61 74 65 97Q63 116 47 130Q29 117 35 88Z','url(#concha)','none',1,1),
 ('M23 26C58 4 88 20 84 62Q80 94 65 113L58 143Q52 169 35 172','none','#d0a2ac',2.3,.8),
 ('M22 23C57 2 89 22 82 61','none','#fff2e5',2.9,.8),
 ('M38 148C52 116 30 103 34 80Q38 49 62 45M35 81Q18 54 25 37','none','#b98e9f',1.8,.8),
 ('M41 143Q52 119 39 104Q33 88 41 70','none','#ffeadd',2.6,.75),
 ('M18 96Q42 91 39 111Q28 123 21 116Z','#efd0c6','#c49aa6',1,.9),
 ('M40 132Q51 135 46 151','none','#bd92a0',1.4,.65)]
def ear_body():return EDEFS+''.join(p(*v) for v in EAR)
face_root=ET.fromstring(OLD['11 Face and ear base aligned to head camera']['svg'])
face_shape=next(n for n in face_root.iter() if n.tag.split('}')[-1]=='path')
ear_portrait=xml(face_shape)+EDEFS
for side in [-1,1]:
    # Ear space -> head space -> established head projection.
    for d,fill,stroke,w,op in EAR:
        import re
        tokens=re.findall(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)',d)
        dd='';j=0
        while j<len(tokens):
            if tokens[j].isalpha():dd+=tokens[j];j+=1
            else:
                u,v=float(tokens[j]),float(tokens[j+1]);px,py=hp(side*(83+u*.20),-8+v*.27)
                dd+=f'{px:.3f} {py:.3f} ';j+=2
        ear_portrait+=p(dd,fill,stroke,w*.22,op*(.85 if side<0 else 1))
recipe('04-ears',[item('11 Face and ear base aligned to head camera',ear_portrait)])

study_layers=[{'name':'01 White study paper','svg':'<svg xmlns="http://www.w3.org/2000/svg" width="2100" height="1400" viewBox="0 0 900 600">'+p('M0 0H900V600H0Z','#f8f5ee')+'<text x="450" y="44" text-anchor="middle" font-family="Segoe UI" font-size="20" fill="#7a7c8d">EAR CARTILAGE / SIMPLIFY FROM STRUCTURE</text></svg>'}]
study_ear='';columns=[]
for i in range(3):
    body=''
    for k,(d,fill,stroke,w,op) in enumerate(EAR):
        if i==0:
            body+=p(d,'none','#aaa0a8',.8,.7)
        elif i==1:body+=p(d,fill,stroke,w,op)
        elif k in [0,1,2,4,6]:body+=p(d,fill,stroke,w*.8,op*.8)
    col=f'<g transform="translate({115+i*270} 143) scale(1.6)">{body}</g>'
    col+=f'<text x="{190+i*270}" y="510" text-anchor="middle" font-family="Segoe UI" font-size="14" fill="#848392">'+['CONSTRUCTION','PLANES AND FOLDS','ANIME SIMPLIFICATION'][i]+'</text>'
    columns.append(col);study_ear+=col
study_svg='<svg xmlns="http://www.w3.org/2000/svg" width="2100" height="1400" viewBox="0 0 900 600">'+EDEFS+study_ear+'</svg>'
study_layers.append({'name':'02 Helix antihelix concha and tragus study','svg':study_svg})
ear_study={'name':'Ear structure study','width':2100,'height':1400,'scale':2100/900,'stem':'ear-structure','blank_checkpoint':'blank.kra','layers':study_layers}
(OUT/'ear-study.json').write_text(json.dumps(ear_study),encoding='utf-8')
under=copy.deepcopy(ear_study);under['stem']='ear-construction'
under['layers']=[study_layers[0],{'name':'02 Ear construction before shading','svg':study_svg.replace(study_ear,columns[0])}]
(OUT/'ear-construction.json').write_text(json.dumps(under),encoding='utf-8')
finish=copy.deepcopy(ear_study);finish['expected_document']=str((OUT/'ear-study/ear-construction.kra').resolve())
finish['layers']=[{'name':'03 Ear planes and simplified contour','svg':study_svg.replace(study_ear,''.join(columns[1:]))}]
(OUT/'ear-shading.json').write_text(json.dumps(finish),encoding='utf-8')

# Cheek and jaw planes: retain the anime silhouette, vary warm light/cool shadow.
FDEFS='''<defs><radialGradient id="cheekCool"><stop stop-color="#a0a5be" stop-opacity=".22"/><stop offset="1" stop-color="#d9b6bf" stop-opacity="0"/></radialGradient><radialGradient id="cheekWarm"><stop stop-color="#edb1b6" stop-opacity=".3"/><stop offset="1" stop-color="#eec6c5" stop-opacity="0"/></radialGradient><radialGradient id="faceCream"><stop stop-color="#fff9e9" stop-opacity=".63"/><stop offset="1" stop-color="#fff4e6" stop-opacity="0"/></radialGradient></defs>'''
planes=FDEFS
for u,v,rx,ry,fill in [(69,35,24,52,'cheekCool'),(-50,38,28,19,'cheekWarm'),(53,36,22,17,'cheekWarm'),(-25,6,29,38,'faceCream'),(-10,78,28,12,'faceCream')]:
    x,y=hp(u,v);planes+=e(x,y,rx,ry,'url(#'+fill+')')
planes+=hd([['M',-49,77],['Q',-20,99,0,100],['Q',25,96,50,73]],stroke='#d4aeb8',w=.55,opacity=.4)
layer=item('23 Refined cheek planes and native soft color',planes)
layer['svg_masks']=[svg(ear_portrait)];layer['svg_erase_masks']=[OLD['17 Rooted bangs and shoulder locks with fine fibers']['svg']]
layer['svg_erase_masks'] += [OLD['13 Eye -1 warm white']['svg'],OLD['13 Eye 1 warm white']['svg'],OLD['15 Almond eyelids and restrained lashes']['svg']]
recipe('05-face-planes',[layer])

# Generic lock construction retains fine detail with lower contrast. Base and
# fibers remain separate, and only their own approved silhouette can show them.
HDEFS='''<defs><radialGradient id="scalp" cx="35%" cy="23%" r="75%"><stop stop-color="#dcebf0"/><stop offset=".48" stop-color="#b6cfde"/><stop offset="1" stop-color="#7494b1"/></radialGradient><linearGradient id="softHair" x1="0%" y1="0%" x2="100%" y2="35%"><stop stop-color="#7496b4"/><stop offset=".3" stop-color="#bed7e4"/><stop offset=".52" stop-color="#dbeaf0"/><stop offset=".76" stop-color="#b4cedf"/><stop offset="1" stop-color="#7a9bb7"/></linearGradient></defs>'''
cap=p('M380 298C372 237 398 193 443 180Q486 158 532 181C572 199 591 246 585 307Q570 370 525 391Q475 414 422 376Q385 347 380 298Z','url(#scalp)','#83a1ba',.7)
for j in range(28):
    x=392+j*1.3
    cap+=p(f'M{x} 248Q{404+j*.8} {186+j*.2} {478+j*.5} {177+j*.25}',stroke='#edf5f6' if j%4==0 else '#8faec4',w=.25,opacity=.35)
rear=copy.deepcopy(OLD['06 Long rear hair grouped by flow']);rear['svg']=rear['svg'].replace('</svg>',HDEFS+cap+'</svg>')
front='';fibers=''
specs=[([(469,183),(427,197),(397,230),(397,287)],18,56),
       ([(511,190),(563,221),(583,253),(580,290)],13,48),
       ([(490,178),(464,214),(453,259),(414,284)],28,62),
       ([(493,178),(519,211),(540,236),(565,253)],26,60),
       ([(395,250),(378,387),(416,472),(366,586)],24,65),
       ([(581,251),(649,330),(586,453),(619,554)],18,68),
       ([(401,305),(389,417),(404,479),(386,530)],12,38),
       ([(605,353),(592,433),(605,477),(628,514)],9.5,38)]
for q,w,n in specs:
    left=[strand(q,i/40,-1,w) for i in range(41)];right=[strand(q,i/40,1,w) for i in range(40,-1,-1)]
    d=smooth(left)+'L%.3f %.3f'%right[0]+smooth(right)[smooth(right).find('C'):]+'Z'
    front+=p(d,'url(#softHair)','#7b9bb4',.55)
    for j in range(n):
        off=-.97+1.94*(j+R.uniform(-.2,.2))/(n-1);lo=R.uniform(.05,.12);hi=R.uniform(.84,.995)
        pts=[strand(q,lo+(hi-lo)*i/32,off,w) for i in range(33)]
        fibers+=p(smooth(pts),stroke=R.choice(['#91b1c7','#acc8d8','#cee1e9','#e4f0f3']),w=R.uniform(.14,.32),opacity=R.uniform(.25,.5))
front_layer=item('17 Rooted bangs and shoulder locks with fine fibers',HDEFS+front)
fine=item('19b Fine fibers for 17 Rooted bangs and shoulder locks with fine fibers',fibers)
fine['svg_masks']=[svg(front)]
# front gradient is irrelevant for alpha masks; include its definitions anyway.
fine['svg_masks']=[svg(HDEFS+front)]
fine['svg_erase_masks']=[OLD['18 Embroidered hem ribbon and small jewelry']['svg']]
rear_fine=copy.deepcopy(OLD['19b Fine fibers for 06 Long rear hair grouped by flow'])
occlusion=''
for source_name in ['07 Stockings and blue ribbon shoes','08 Neck shoulders and cast shadow','09 Seated dress silhouette and tension folds','10 Puff sleeves hands and a quiet book','11 Face and ear base aligned to head camera','17 Rooted bangs and shoulder locks with fine fibers','18 Embroidered hem ribbon and small jewelry']:
    source_svg=svg(ear_portrait) if source_name.startswith('11 ') else front_layer['svg'] if source_name.startswith('17 ') else OLD[source_name]['svg']
    for n in ET.fromstring(source_svg).iter():
        if n.tag.split('}')[-1] not in ['path','ellipse'] or n.get('fill','none')=='none':continue
        n=copy.deepcopy(n);n.tag=n.tag.split('}')[-1];n.set('fill','#fff');n.set('stroke','none');n.set('opacity','1')
        occlusion+=ET.tostring(n,encoding='unicode')
rear_fine['svg_erase_masks']=[svg(occlusion)]
recipe('06-hair',[rear,front_layer,fine,rear_fine,{'name':'24 Refined hair broad shadows and broken sheen','visible':False},
                  {'name':'19 Native Gpen hair finishing','visible':False},
                  {**layer,'svg_erase_masks':[svg(HDEFS+front),OLD['13 Eye -1 warm white']['svg'],OLD['13 Eye 1 warm white']['svg'],OLD['15 Almond eyelids and restrained lashes']['svg']]}])

# The enlarged head revealed repeating eyebrow hooks and a high earring
# attachment. Keep body jewelry intact and move only the two earring shapes.
jroot=ET.fromstring(OLD['18 Embroidered hem ribbon and small jewelry']['svg'])
shapes=[n for n in jroot.iter() if n.tag.split('}')[-1] in ['path','ellipse']]
for n in shapes[-4:]:
    for parent in jroot.iter():
        if n in list(parent):parent.remove(n);break
jewels={'name':'18 Embroidered hem ribbon and small jewelry','svg':ET.tostring(jroot,encoding='unicode')}
recipe('07-head-review',[item('12 Fine low arched eyebrows',brows),jewels])
earrings=''
for side in [-1,1]:
    x,y=hp(side*89,38)
    earrings+=e(x,y,1.15,1.15,'#c4ad79',.9)
    earrings+=p(f'M{x} {y+1}Q{x+1.3} {y+4} {x} {y+7}',stroke='#b49e70',w=.7)
    earrings+=e(x,y+13,3.2,5.2,'#a1c9d8',stroke='#b6a279',w=.6)
    earrings+=p(f'M{x-1} {y+10}Q{x-2} {y+12} {x-1.4} {y+14}',stroke='#e8f3ed',w=.5,opacity=.75)
new_earring=item('25 Ear lobe attachment and occluded earrings',earrings)
new_earring['svg_erase_masks']=[svg(HDEFS+front)]
earring_scene={**BASE,'layers':[new_earring],'stem':'head-07-head-review','expected_document':str(PREVIOUS),'details':DETAILS}
(OUT/'07-earrings.json').write_text(json.dumps(earring_scene,ensure_ascii=False,indent=2),encoding='utf-8')

lids=''
for side in [-1,1]:
    cx=side*46
    lids+=hd([['M',cx-28,0],['C',cx-15,-13,cx+12,-15,cx+28,-2],
              ['L',cx+28,-.6],['C',cx+12,-11.5,cx-14,-10.8,cx-28,1.2],['Z']],'#65677c')
    for j in range(3):
        rx=cx+side*(26-j*3.1);ry=-2.5-j*1.6
        tx=cx+side*(34-j*2.2);ty=-9-j*2.3
        lids+=hd([['M',rx,ry],['Q',rx+side*4,ry-1.3,tx,ty],
                  ['Q',rx+side*3,ry+1,rx-side*1.7,ry+1],['Z']],'#67677b',opacity=.95-j*.1)
    lids+=hd([['M',cx-20,8],['Q',cx+4,20,cx+24,5]],stroke='#b89eac',w=.5,opacity=.8)
    lids+=hd([['M',cx-25,-10],['Q',cx,-22,cx+23,-11]],stroke='#cfb1ba',w=.48,opacity=.68)
    inner=cx-side*27
    lids+=hd([['M',inner,0],['Q',inner+side*3,2,inner+side*2,3.5],['Z']],'#d6a9b1',opacity=.4)
    for j in range(2):
        rx=cx+side*(21-j*3);ry=8+j
        lids+=hd([['M',rx,ry],['Q',rx+side*1.2,ry+1.6,rx+side*2.7,ry+2]],stroke='#a594a7',w=.25,opacity=.5)
lids_layer=item('15 Almond eyelids and restrained lashes',lids)
planes_review={**layer,'svg_erase_masks':[svg(HDEFS+front),OLD['13 Eye -1 warm white']['svg'],OLD['13 Eye 1 warm white']['svg'],svg(lids)]}
recipe('08-eyelids',[lids_layer,planes_review])

# Apply the same reviewed shapes directly to the enlarged head document.
large=json.loads((OUT/'08-eyelids.json').read_text(encoding='utf-8'))
large.update(width=2400,height=2600,scale=2400/290,stem='azure-head-study',
             expected_document=str((OUT/'final-head-reviewed/azure-head-study.kra').resolve()),
             details=[{'file':'head-detail.png','rect':[5,5,280,285]},
                      {'file':'eyes-detail.png','rect':[50,135,170,58]}])
for n in large['layers']:
    for key in ['svg','svg_masks','svg_erase_masks']:
        values=[n[key]] if key=='svg' else n.get(key,[])
        values=[v.replace('width="2400" height="3360" viewBox="0 0 1000 1400"',
                          'width="2400" height="2600" viewBox="350 150 290 314.1666666667"') for v in values]
        if key=='svg':n[key]=values[0]
        elif key in n:n[key]=values
(OUT/'08-eyelids-closeup.json').write_text(json.dumps(large),encoding='utf-8')

assessment={
 'brows':{'before':'Uniform monoline, hard ends; lacks taper and hair direction.','change':'Tapered ash-blue base plus low contrast directional hair marks.'},
 'nose':{'before':'C-shaped mark and bright dot are isolated symbols.','change':'Warm soft underside, smaller coherent highlight and a short nostril hint.'},
 'mouth':{'before':'Two detached strokes with little lip volume.','change':'Small shallow opening, faint lip tint, subordinate lower highlight.'},
 'ears':{'before':'Most cartilage is hidden by hair; current image cannot validate ear anatomy.','change':'Uncovered structural study, then project simplified cartilage into the existing ear positions.'},
 'face':{'before':'Large flat cheek planes, hard jaw contour and neck shadow.','change':'Subtle warm/cool cheek planes and lower cheek light; preserve outline.'},
 'hair':{'before':'Bulbous parting, repetitive high contrast fibers and brush-like shine patches.','change':'Round scalp silhouette, coherent roots, lower contrast fibers and broad value grouping.'}}
(OUT/'pre-drawing-assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared six successive head revisions and an uncovered ear structure study')
