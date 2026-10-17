"""Artwork recipe snapshot, for reproducing the 2026-10-16 study only.

Run from repository root. The reusable renderer is tools/krita_studio.py.
Reuses our own first study's authored geometry, not the user's reference pixels.
"""
import json
import math
import random
import re
from pathlib import Path

base = Path('data/test-result/20261016/krita-blue-girl')
out = Path('data/test-result/20261016/krita-anime-study')
out.mkdir(parents=True, exist_ok=True)
r = random.Random(206)
def old(n):
    return (base / ('layer-%02d.svg' % n)).read_text(encoding='utf-8')
defs = '<defs>' + old(1).split('<defs>')[1].split('</defs>')[0] + '</defs>'
def svg(body):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="2700" viewBox="0 0 900 1350">' + defs + '<g stroke-linecap="round" stroke-linejoin="round">' + body + '</g></svg>'
def p(d, fill='none', stroke='none', width=1):
    return '<path d="%s" fill="%s" stroke="%s" stroke-width="%s"/>' % (d,fill,stroke,width)
def e(x,y,rx,ry,fill,stroke='none',width=1):
    return '<ellipse cx="%s" cy="%s" rx="%s" ry="%s" fill="%s" stroke="%s" stroke-width="%s"/>' % (x,y,rx,ry,fill,stroke,width)
def s(brush,size,color,path=None,line=None,opacity=1,taper=False):
    d={'brush':brush,'size':size,'color':color,'opacity':opacity}
    if path:d['path']=path
    if line:d['line']=line
    if taper:d.update(taper=True,samples=72)
    return d
G='d) Ink-3 Gpen'
B='b) Basic-5 Size'
A='b) Airbrush Soft'
WC='j) WaterC Basic Round-Grain'
layers=[]
layers.append({'name':'01 Paper and window','svg':old(1).replace('stroke-width="3"','stroke-width="1.5"')})
wash=[]
for i in range(16):
    x=r.choice([r.uniform(100,285),r.uniform(650,815)]); y=r.uniform(200,1150)
    wash.append(s(WC,r.uniform(130,210),'#c0d3e7',line=[x,y,x+50,y+100],opacity=.35))
layers.append({'name':'02 Watercolor atmosphere','strokes':wash,'opacity':.5})
layers.append({'name':'03 Botanical silhouettes','svg':old(2),'opacity':.65})
hb=old(4).replace('fill="#edf9ff"','fill="#cbe6f6" fill-opacity=".48"').replace('stroke-width="2.5"','stroke-width="1.2"').replace('stroke-width="1.4"','stroke-width=".8"')
hairsh=[]
for side in [0,1]:
    for j in range(8):
        x=347+j*6 if not side else 544+j*4
        hairsh.append(s(A,22,'#527da8',path=[['M',x,430],['C',x-10 if not side else x+10,660,x-85 if not side else x+55,820,x-65 if not side else x+80,1010]],opacity=.16))
hairlines=[]
for j in range(24):
    x=344+j*1.7
    hairlines.append(s(G,.65 if j%3 else 1,'#648fb8',path=[['M',x,455+j*3],['C',x-6,640,x-90,833,x-83,1012+j*3]],opacity=.38))
for j in range(23):
    x=534+j*1.45
    hairlines.append(s(G,.6 if j%3 else .95,'#6385af',path=[['M',x,447+j*5],['C',x+5,630,x+85,831,x+82,1013+j*2]],opacity=.36))
hairgloss=[]
for x,y,xx,yy in [(327,649,293,799),(349,665,329,812),(568,655,593,792),(590,744,617,877),(298,899,283,1032),(572,937,605,1053)]:
    hairgloss.append(s(A,13,'#edfaff',path=[['M',x,y],['Q',(x+xx)/2-8,(y+yy)/2,xx,yy]],opacity=.65))
layers.append({'name':'04 Hair behind chair - clipped materials','children':[{'name':'Hair silhouette','svg':hb},{'name':'Hair soft shadow','strokes':hairsh,'inherit_alpha':True,'blend':'multiply'},{'name':'Hair internal strand groups','strokes':hairlines,'inherit_alpha':True},{'name':'Hair reflected skylight','strokes':hairgloss,'inherit_alpha':True,'blend':'screen'}]})
layers.append({'name':'05 Chair and carved apron','svg':old(3)})
layers.append({'name':'06 Chair armrests','svg':old(15)})
leg=old(5).replace('stroke-width="2"','stroke-width="1.1"')
leg=re.sub('<path d="M372 862.*?/>','',leg)
leg=leg.replace('M373 762C342 807 351 867 372 927','M373 762C331 806 350 851 372 889C383 905 372 917 372 927')
leg=leg.replace('M467 773C454 838 456 887 470 949','M467 773C443 814 457 859 481 886C493 905 465 926 470 949')
legshade=[s(A,34,'#a7a8c9',path=[['M',374,871],['C',406,933,421,1062,393,1169]],opacity=.2),s(A,33,'#a1aac5',path=[['M',503,867],['C',522,954,526,1069,499,1193]],opacity=.2),s(A,30,'#fff5ed',path=[['M',383,795],['Q',376,836,394,878]],opacity=.32),s(A,26,'#fff5ed',path=[['M',501,797],['Q',482,842,495,876]],opacity=.32)]
for x in [377,501]:legshade.append(s(A,34,'#b9aaca',line=[x,886,x+13,896],opacity=.23))
layers.append({'name':'07 Legs - knee light and stocking volume','children':[{'name':'Leg silhouettes','svg':leg},{'name':'Stocking soft form shadows','strokes':legshade,'inherit_alpha':True},{'name':'Stocking satin glints','strokes':[s(A,10,'#ffffff',path=[['M',391,942],['Q',408,1053,387,1146]],opacity=.5),s(A,9,'#ffffff',path=[['M',501,945],['Q',520,1063,498,1165]],opacity=.5)],'inherit_alpha':True,'blend':'screen'}]})
body=old(6).replace('stroke-width="2"','stroke-width="1.1"').replace('stroke-width="1.8"','stroke-width="1.1"')
bodysh=svg(p('M372 488Q394 510 401 552Q409 585 393 616L382 627L388 599Q401 566 375 540Z','#809abe')+p('M525 491Q500 521 491 554Q480 588 503 624L515 620Q495 575 522 541Z','#a2b4d3')+p('M439 537Q446 580 434 630L446 636L450 538Z','#bdcde5'))
bodysoft=[s(A,24,'#e5aebc',path=[['M',372,459],['Q',386,475,412,481]],opacity=.21),s(A,22,'#dd9eaf',path=[['M',493,455],['Q',513,455,528,472]],opacity=.18),s(A,25,'#ffffff',path=[['M',421,552],['Q',430,582,419,606]],opacity=.6),s(A,20,'#718fb8',path=[['M',456,548],['Q',460,582,454,625]],opacity=.22)]
layers.append({'name':'08 Bodice - hard shadows plus airbrush','children':[{'name':'Bodice and shoulders flats','svg':body},{'name':'Bodice fold planes','svg':bodysh,'inherit_alpha':True,'blend':'multiply','opacity':.55},{'name':'Bodice reflected light','strokes':bodysoft,'inherit_alpha':True}]})
skirt=old(7).replace('stroke-width="2"','stroke-width="1.1"').replace('stroke-width="4"','stroke-width="1.8"')
folds=[]
for x,y,xx,yy in [(389,674,354,787),(417,682,393,803),(444,689,433,820),(474,686,490,814),(506,675,549,786)]:
    folds.append(s(A,13,'#637ca8',path=[['M',x,y],['Q',(x+xx)/2+5,(y+yy)/2,xx,yy]],opacity=.27))
    folds.append(s(G,1,'#6e88b1',path=[['M',x+3,y+9],['Q',(x+xx)/2+7,(y+yy)/2,xx+4,yy-7]],opacity=.55,taper=True))
emb=''
for i in range(18):
    x=330+i*13;y=773+25*math.sin(i/17*math.pi)
    emb+=p('M%.1f %.1fq-5 -12 0 -18q6 8 0 18m0 -7q-9 -2 -10 -9q9 -1 10 9m0 0q9 -2 10 -9q-9 -1 -10 9'%(x,y),stroke='#f0f7fc',width=.65)
layers.append({'name':'09 Skirt - folds embroidery and lace','children':[{'name':'Skirt flats','svg':skirt},{'name':'Fold depth - airbrush and Gpen','strokes':folds,'inherit_alpha':True},{'name':'Lace embroidery','svg':svg(emb),'inherit_alpha':True}]})
arm=old(8).replace('stroke-width="1.8"','stroke-width="1"').replace('stroke-width="1.35"','stroke-width=".75"')
armsoft=[s(A,14,'#d79daa',path=[['M',337,555],['Q',326,612,345,635],['Q',364,658,406,680]],opacity=.25),s(A,14,'#d5a3b6',path=[['M',560,555],['Q',581,613,558,637],['Q',537,655,503,681]],opacity=.25),s(A,9,'#ffffff',path=[['M',341,563],['Q',333,606,350,626]],opacity=.52),s(A,10,'#ffffff',path=[['M',551,558],['Q',570,600,554,622]],opacity=.42)]
layers.append({'name':'10 Arms and folded hands','children':[{'name':'Arms flats','svg':arm},{'name':'Warm arm form shading','strokes':armsoft,'inherit_alpha':True}]})
face=p('M378 304C376 272 403 240 450 240C497 240 524 270 523 307L519 352C512 386 477 413 451 419C425 415 392 390 383 356Z','url(#skin)','#92859b',.95)
face+=p('M384 309Q406 297 439 304L450 283L466 308Q492 296 518 313L517 323Q487 310 464 321L448 306L435 317Q403 308 384 323Z','#cfadbc')
face+=p('M450 340L446 358L450 360',stroke='#b68c9c',width=.85)+p('M441 381Q450 378 459 382',stroke='#b27f91',width=1)+p('M446 387Q451 389 455 387',stroke='#f5d2d4',width=1)
for cx in [411,489]:
    face+=p('M%d 331Q%d 317 %d 327Q%d 345 %d 338Q%d 335 %d 331Z'%(cx-22,cx-2,cx+22,cx+8,cx-18,cx-22,cx-22),'#fffcfa','#9a8b9f',.7)
    face+=e(cx,331,10.8,13.1,'url(#iris)','#61769a',.7)+e(cx,330,3.8,9,'#465584')+e(cx-3.4,324,3.2,4,'#ffffff')+e(cx+4.5,336,1.7,2,'#eefcff')
    face+=p('M%d 331Q%d 313 %d 327'%(cx-23,cx-1,cx+22),stroke='#5b526e',width=2.2)
    face+=p('M%d 339Q%d 345 %d 339'%(cx-15,cx,cx+16),stroke='#b4a2b4',width=.65)
    face+=p('M%d 305Q%d 301 %d 306'%(cx-18,cx,cx+17),stroke='#9194ae',width=1)
    for j in range(3):
        xx=cx-21+j*3 if cx<450 else cx+16+j*3
        face+=p('M%d %dl%d -4'%(xx,326-j%2,-3 if cx<450 else 3),stroke='#5b526e',width=.75)
blush=[s(A,26,'#ec9faf',line=[398,355,407,358],opacity=.35),s(A,24,'#ec9faf',line=[492,356,504,354],opacity=.35),s(A,14,'#e5b1b5',line=[447,358,451,360],opacity=.3),s(A,18,'#fffdf7',line=[429,341,431,353],opacity=.55)]
layers.append({'name':'11 Face - warm skin and refined eyes','children':[{'name':'Face and eyes flats','svg':svg(face)},{'name':'Cheek color - soft airbrush','strokes':blush,'inherit_alpha':True},{'name':'Cheek controlled blur','svg':svg(e(403,354,10,3,'#f5bec6')+e(498,354,9,3,'#f5bec6')),'filters':[{'name':'gaussian blur','settings':{'horizRadius':8,'vertRadius':8,'lockAspect':True}}],'inherit_alpha':True,'opacity':.38}]})
fh=old(10).replace('stroke-width="2"','stroke-width="1"').replace('stroke-width="1.8"','stroke-width="1"').replace('fill="#fff"','fill="#edfaff"').replace('opacity=".72"','opacity=".38"')
fhs=[]
for j in range(10):
    fhs.append(s(G,.7,'#8aaac9',path=[['M',444+j*1.3,221+j*.8],['Q',429+j*.8,264+j*1.7,401+j*2,299+j*1.5]],opacity=.55))
    fhs.append(s(G,.7,'#88a6c6',path=[['M',467+j*2.6,224+j],['Q',477+j*2.4,250+j*1.5,510+j*.7,278+j*1.2]],opacity=.52))
for j in range(7):
    fhs.append(s(G,.65,'#7298bc',path=[['M',380+j*1.5,349],['C',378+j*2,407,389+j,470,351+j*2.5,517]],opacity=.53))
    fhs.append(s(G,.65,'#6d93ba',path=[['M',524+j*.8,349],['C',516+j,417,525+j*1.7,464,558+j*1.8,515]],opacity=.53))
highlights=[]
for j in range(8):
    highlights.append(s(B,2.2,'#f8fdff',path=[['M',390+j*4,246-j*1.7],['Q',403+j*3,230-j,421+j*2.8,225+j*.4]],opacity=.62,taper=True))
    highlights.append(s(B,2,'#f4fcff',path=[['M',479+j*3.7,228+j],['Q',497+j*2,234+j*1.6,505+j*1.5,246+j*2]],opacity=.6,taper=True))
fhs += [s(A,14,'#c6b9db',path=[['M',369,293],['Q',367,325,380,345]],opacity=.26),s(A,17,'#e1d1e5',path=[['M',539,277],['Q',548,310,528,340]],opacity=.24)]
layers.append({'name':'12 Front hair - strand hierarchy and light ribbons','children':[{'name':'Bangs base','svg':fh},{'name':'Fine hair engraving - Gpen','strokes':fhs,'inherit_alpha':True},{'name':'Tapered broken highlight band','strokes':highlights,'inherit_alpha':True,'blend':'screen'},{'name':'Soft reflected fringe light','strokes':[s(A,22,'#ffffff',path=[['M',389,255],['Q',430,217,451,224]],opacity=.27)],'inherit_alpha':True}]})
layers.append({'name':'13 Jewelry','svg':old(11).replace('stroke-width="1.6"','stroke-width="1"')})
layers.append({'name':'14 Sakura branches','svg':old(12).replace('fill="#fbecf2"','fill="#f8e4ed"').replace('stroke-width=".9"','stroke-width=".65"')})
flowerwash=[]
for x,y in [(39,204),(85,220),(183,181),(236,141),(844,151),(772,191),(683,239),(830,256),(71,1001),(143,908),(804,982),(761,940)]:
    flowerwash.append(s(WC,20,'#cf8caa',line=[x-7,y-5,x+5,y+3],opacity=.5))
layers.append({'name':'15 Watercolor flower centers','strokes':flowerwash,'opacity':.65})
layers.append({'name':'16 Petals and title','svg':old(13).replace('A SEAT AMONG BLOSSOMS','AZURE / A PAINTING STUDY'),'checkpoint':'01-before-native-finishing.png'})
stars=[s('z) Stamp Sparkles',18,'#ffffff',line=[x,y,x+.5,y+.2],opacity=.7) for x,y in [(286,425),(618,408),(256,806),(645,826),(314,1045),(586,1108)]]
layers.append({'name':'17 Native sparkle accents','strokes':stars,'checkpoint':'02-native-shaded.png'})
glow=svg(e(350,266,55,18,'#f5faff')+e(590,585,23,85,'#e9e9ff')+e(300,800,25,90,'#e2f4ff'))
layers.append({'name':'18 Optional bloom','svg':glow,'filters':[{'name':'gaussian blur','settings':{'horizRadius':34,'vertRadius':34,'lockAspect':True}}],'blend':'screen','opacity':.15})
scene={'name':'Azure II - native brush and cel shading study','width':1800,'height':2700,'scale':2,'stem':'azure-study-v2','layers':layers,'variants':[{'file':'03-with-bloom.png','visibility':{'18 Optional bloom':True}},{'file':'04-clean-render.png','visibility':{'18 Optional bloom':False}}]}
(out/'azure-study-v2.scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'layers':len(layers),'native_strokes':sum(len(c.get('strokes',[])) for l in layers for c in [l]+l.get('children',[]))}))
