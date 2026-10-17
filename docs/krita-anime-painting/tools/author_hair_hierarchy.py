"""Controlled bundle study: large masses, medium overlap, fine material."""
import json
import math
import random
import sys
from pathlib import Path

OUT=Path('data/test-result/20261017/krita-hair-hierarchy')
OUT.mkdir(exist_ok=True)
R=random.Random(171099)
REVIEW='--review' in sys.argv
def p(d,fill='none',stroke='none',w=1,op=1):return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'
def label(x,y,s,size=16):return f'<text x="{x}" y="{y}" font-family="Segoe UI" font-size="{size}" fill="#687b90" text-anchor="middle">{s}</text>'
def group(i,s):return f'<g transform="translate({[200,600,1000][i]} 155)">{s}</g>'
def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1600" viewBox="0 0 1200 800">'+s+'</svg>'
def qpoint(q,t):return tuple((1-t)**3*q[0][j]+3*(1-t)**2*t*q[1][j]+3*(1-t)*t*t*q[2][j]+t**3*q[3][j] for j in (0,1))
def normal(q,t):
    x=3*(1-t)**2*(q[1][0]-q[0][0])+6*(1-t)*t*(q[2][0]-q[1][0])+3*t*t*(q[3][0]-q[2][0])
    y=3*(1-t)**2*(q[1][1]-q[0][1])+6*(1-t)*t*(q[2][1]-q[1][1])+3*t*t*(q[3][1]-q[2][1])
    z=math.hypot(x,y);return y/z,-x/z
def width(lock,t):return lock['width']*(.36+.9*math.sin(math.pi*t)**.85)*(1-t)**.48
def position(lock,t,u):
    q=lock['q'];x,y=qpoint(q,t);nx,ny=normal(q,t);ww=width(lock,t)
    return x+nx*ww*u,y+ny*ww*u
def poly(pts):return 'M'+'L'.join(f'{x:.3f} {y:.3f}' for x,y in pts)+'Z'
def silhouette(lock):
    pts=[position(lock,j/90,-1) for j in range(91)]+[position(lock,j/90,1) for j in range(90,-1,-1)]
    return poly(pts)
LOCKS=[
 {'id':'rear-left','q':[[-26,7],[-111,166],[85,316],[-79,530]],'width':27,'depth':0},
 {'id':'rear-right','q':[[19,10],[90,159],[-48,335],[80,468]],'width':31,'depth':.08},
 {'id':'middle-left','q':[[-23,0],[-76,173],[82,292],[-49,506]],'width':25,'depth':.19},
 {'id':'middle-right','q':[[20,1],[59,164],[-20,284],[62,486]],'width':23,'depth':.24},
 {'id':'center','q':[[-3,0],[-56,165],[93,304],[-21,552]],'width':29,'depth':.37},
 {'id':'front-left','q':[[-14,3],[-65,189],[60,276],[-44,451]],'width':13,'depth':.42},
 {'id':'front-right','q':[[20,7],[38,159],[3,296],[34,413]],'width':11,'depth':.46}]

def rgb(v):return '#'+''.join(f'{max(0,min(255,round(x))):02x}' for x in v)
def col(value):
    a=(105,128,165);b=(226,236,243)
    return rgb([a[j]*(1-value)+b[j]*value for j in range(3)])
def tone(lock,t,u):
    # An illustrative curved surface field, not measured physical rendering.
    crown=max(0,1-(u+.18)**2)**.58
    flow=.55+.22*math.sin(t*math.pi*3-.6)+.07*math.sin(t*math.pi*7)
    spec=.16*math.exp(-((t-.22)/.11)**2)*math.exp(-((u+.35)/.32)**2)
    return max(.04,min(.95,.16+.41*crown+.22*flow+spec+lock['depth']*.1))
def rendered(lock,index):
    defs='';body=''
    # Narrow strips have a varying transverse gradient. A single straight
    # bounding-box gradient cannot follow a long S curve's turning surface.
    for j in range(145):
        t0=max(0,(j-.035)/145);t1=min(.99998,(j+1.035)/145);tm=(t0+t1)/2
        a=position(lock,tm,-1);b=position(lock,tm,1);gid=f'g{index}_{j}'
        defs+=f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}">'
        for k in range(13):
            u=-1+k/6;defs+=f'<stop offset="{k/12}" stop-color="{col(tone(lock,tm,u))}"/>'
        defs+='</linearGradient>'
        pts=[position(lock,t0,-1),position(lock,t1,-1),position(lock,t1,1),position(lock,t0,1)]
        body+=p(poly(pts),f'url(#{gid})')
    if REVIEW:
        li=LOCKS.index(lock)
        for k,upper in enumerate(LOCKS[li+1:li+3]):
            body+='<g transform="translate(3 5)">'+p(silhouette(upper),'#465b80',op=.16 if k==0 else .07)+'</g>'
    return '<defs>'+defs+'</defs>'+body

paper=p('M0 0H1200V800H0Z','#f4f1ea')+label(600,37,'HAIR HIERARCHY / VOLUME BEFORE FIBERS',23)
paper+=label(600,64,'Same seven original clumps: flat shapes → turning planes → selective texture',13)
for i,s in enumerate(['01 FLAT MASS CONTROL','02 MEDIUM CLUMPS + DEPTH','03 FINE MATERIAL + EDGE VARIATION']):paper+=label([200,600,1000][i],735,s,14)
paper+=label(600,773,'Isolated material exercise; not a finished character or a physical hair simulation',12)
flat=''
for l in LOCKS:flat+=p(silhouette(l),'#aec6dc','#8097b0',.65)
sketch=''
for l in LOCKS:
    q=l['q'];sketch+=p('M'+str(q[0][0])+' '+str(q[0][1])+'C'+' '.join(str(v) for point in q[1:] for v in point),stroke='#b58a94',w=.7)
one={'name':'Hair hierarchy — isolated depth and texture study','width':2400,'height':1600,'scale':2,'stem':'01-bundle-construction','blank_checkpoint':'00-blank.kra','layers':[
 {'name':'01 Neutral study paper','svg':svg(paper)},
 {'name':'02 Flat control with seven overlapping clumps','svg':svg(group(0,flat))},
 {'name':'03 Direction sketch for depth study','svg':svg(group(1,sketch)),'checkpoint':'01-direction-sketch.png'}],'details':[]}
if not REVIEW:(OUT/'01-construction.json').write_text(json.dumps(one),encoding='utf-8')
middle=[]
for j,l in enumerate(LOCKS):middle.append({'name':f'04 Medium clump {j} {l["id"]}','svg':svg(group(1,rendered(l,j))), 'svg_masks':[svg(group(1,p(silhouette(l),'#fff')))],'visibility':{'03 Direction sketch for depth study':False}})
two={**one,'stem':'02-bundle-volume','layers':middle,'expected_document':str((OUT/'painting/01-bundle-construction.kra').resolve())}
if not REVIEW:(OUT/'02-volume.json').write_text(json.dumps(two),encoding='utf-8')

final=[]
for j,l in enumerate(LOCKS):
    shade='';tex='';high=''
    # One medium sublock becomes several uneven smaller divisions. Wide
    # troughs and thin glints are subordinate to the clump's turning value.
    for k,u in enumerate([-.72,-.32,.17,.57]):
        t0=.03+R.random()*.13;t1=.72+R.random()*.26
        pts=[]
        for h in range(81):
            t=t0+(t1-t0)*h/80;uu=u+.035*math.sin(t*11+k)
            pts.append(position(l,t,uu))
        d='M'+'L'.join(f'{x:.2f} {y:.2f}' for x,y in pts)
        shade+=p(d,stroke='#677c9b',w=.65+R.random()*.6,op=.2)
    for k in range(95):
        u=R.uniform(-.96,.96);t0=R.uniform(.015,.5);t1=min(.997,t0+R.uniform(.22,.7));phase=R.random()*6
        if REVIEW and (tone(l,(t0+t1)/2,u)<.58 or abs(u)>.72) and k%4:continue
        pts=[]
        for h in range(35):
            t=t0+(t1-t0)*h/34;uu=u+.012*math.sin(t*16+phase)
            pts.append(position(l,t,uu))
        d='M'+'L'.join(f'{x:.2f} {y:.2f}' for x,y in pts)
        ink=R.choice(['#dce8ef','#899eb9','#f0f2ee','#667f9e']);ww=R.uniform(.12,.29);alpha=R.uniform(.1,.29)
        if REVIEW:
            # A tapered narrow shape avoids a constant-width engraved line.
            left=[];right=[]
            for h,(x,y) in enumerate(pts):
                t=t0+(t1-t0)*h/34;nx,ny=normal(l['q'],t);half=ww*.6*math.sin(math.pi*h/34)**.6
                left.append((x+nx*half,y+ny*half));right.append((x-nx*half,y-ny*half))
            tex+=p(poly(left+right[::-1]),ink,op=alpha)
        else:tex+=p(d,stroke=ink,w=ww,op=alpha)
    for k in range(15):
        u=R.uniform(-.7,.15);t0=R.uniform(.12,.28);t1=t0+R.uniform(.018,.065)
        pp=[position(l,t0+(t1-t0)*h/10,u) for h in range(11)]
        high+=p('M'+'L'.join(f'{x:.2f} {y:.2f}' for x,y in pp),stroke='#f3f5ee',w=R.uniform(.18,.6),op=R.uniform(.24,.5))
    final.append({'name':f'05 Detailed clump {j} {l["id"]}','svg':svg(group(2,rendered(l,j+20)+shade+tex+high)), 'svg_masks':[svg(group(2,p(silhouette(l),'#fff')))]})
three={**one,'stem':'hair-hierarchy-material','layers':final,'expected_document':str((OUT/'painting/02-bundle-volume.kra').resolve())}
if REVIEW:
    reviewed={**one,'stem':'hair-hierarchy-reviewed','expected_document':str((OUT/'painting/hair-hierarchy-material.kra').resolve()),'layers':middle+final}
    (OUT/'04-reviewed.json').write_text(json.dumps(reviewed),encoding='utf-8')
else:(OUT/'03-material.json').write_text(json.dumps(three),encoding='utf-8')
(OUT/'elements.json').write_text(json.dumps({'purpose':'compare flat section shapes against medium clumps and selective material detail','locks':LOCKS,'light':'upper-left broad source; cool shadows, restrained ivory reflections','stages':['construction','turning values in overlapping medium clumps','small divisions','partial thin fibers','limited highlights','visual rejection and correction'],'limits':['procedural illustrative shading, not freehand or physically measured','isolated bundle has no scalp attachment and does not establish a hairstyle','moving to full head requires existing silhouette and occlusion checks']},indent=2),encoding='utf-8')
print('Prepared original seven-clump controlled hierarchy exercise')
