"""Independent expression topic; never opens or revises the hair document."""
import json
import math
import sys
from pathlib import Path

OUT=Path('data/test-result/20261017/krita-expression-topic')
OUT.mkdir(parents=True,exist_ok=True)
PARAMS=[
 {'id':'neutral','label':'01 NEUTRAL / CONTROL','brow':[0,0],'upper':-13,'lower':16,'mouth':'neutral'},
 {'id':'mouth_only','label':'02 ONLY THE MOUTH CHANGES','brow':[0,0],'upper':-13,'lower':16,'mouth':'smile'},
 {'id':'gentle_smile','label':'03 COORDINATED GENTLE SMILE','brow':[-2,-2],'upper':-12,'lower':9,'mouth':'smile','cheek':True},
 {'id':'surprise','label':'04 SURPRISE','brow':[-12,-12],'upper':-19,'lower':22,'mouth':'open'},
 {'id':'sadness','label':'05 CONCERN / SADNESS','brow':[5,-8],'upper':-10,'lower':15,'mouth':'sad'},
 {'id':'focus','label':'06 DETERMINED / FOCUSED','brow':[-1,8],'upper':-10,'lower':11,'mouth':'firm'}]
REVIEW='--review' in sys.argv
if REVIEW:PARAMS[2]['lower']=12
DEFS='''<defs><linearGradient id="skin" x2="80%" y2="100%"><stop stop-color="#fff5e9"/><stop offset="1" stop-color="#efd4ce"/></linearGradient><linearGradient id="iris" x2="0%" y2="100%"><stop stop-color="#425c79"/><stop offset=".5" stop-color="#82b0c5"/><stop offset="1" stop-color="#c1e4de"/></linearGradient><radialGradient id="cheek"><stop stop-color="#e7a2b0" stop-opacity=".4"/><stop offset="1" stop-color="#e7a2b0" stop-opacity="0"/></radialGradient></defs>'''

def p(d,fill='none',stroke='none',w=1,op=1):return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'
def el(x,y,rx,ry,fill,op=1):return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{op}"/>'
def label(x,y,t,size=14):return f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Segoe UI" font-size="{size}" fill="#617487">{t}</text>'
def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1700" viewBox="0 0 1200 850">'+DEFS+s+'</svg>'
def group(i,s):
    x=[200,600,1000][i%3];y=239 if i<3 else 626
    # Same shallow 3/4 head angle for all cells; asymmetry is held constant.
    return f'<g transform="translate({x} {y}) rotate(-5) scale(.96 1)">{s}</g>'

paper=p('M0 0L1200 0L1200 850L0 850Z','#f7f4ed')
paper+=label(600,30,'EXPRESSION TOPIC / BROWS + EYELIDS + MOUTH + CHEEKS',21)
paper+=label(600,54,'Independent study: fixed head angle and design; no changes to the hair painting',13)
for i,v in enumerate(PARAMS):paper+=label([200,600,1000][i%3],405 if i<3 else 792,v['label'],13)
base=[{'name':'01 Independent expression paper','svg':svg(paper)}]
guides=''
for i in range(6):
    s=el(0,-27,80,109,'none')
    s+=p('M-75 -70C-90 -20 -71 51 -30 82Q0 108 11 101C63 76 86 22 78 -60C71 -121 -51 -135 -75 -70Z',stroke='#aebfc8',w=.8)
    s+=p('M0 -129L0 102M-78 -1Q0 8 79 -1M-64 -27Q0 -21 65 -27M-25 66L28 66',stroke='#d4b4b3',w=.7)
    guides+=group(i,s)
base.append({'name':'02 Head envelope and fixed feature coordinates','svg':svg(guides),'checkpoint':'01-expression-construction.png'})
one={'name':'Independent anime expression study','width':2400,'height':1700,'scale':2,'stem':'expression-construction','blank_checkpoint':'00-blank.kra','layers':base,'details':[]}
if not REVIEW:(OUT/'01-construction.json').write_text(json.dumps(one),encoding='utf-8')
faces='';eyes='';brows='';mouths='';cheeks='';whites='';iris_layers=[]
for i,v in enumerate(PARAMS):
    face=p('M-33 67L-35 120Q0 134 38 120L32 68Z','url(#skin)','#bfa2aa',.7)
    face+=p('M-75 -70C-90 -20 -71 51 -30 82Q0 108 11 101C63 76 86 22 78 -60C71 -121 -51 -135 -75 -70Z','url(#skin)','#b89aa6',.8)
    face+=p('M-82 -19C-104 -94 -46 -154 16 -144C88 -139 104 -85 87 -14L74 -51Q51 -75 31 -106C16 -63 -20 -44 -66 -30L-76 58Q-89 23 -82 -19Z','#9fc3d8','#7299b6',.8)
    face+=p('M-4 39Q1 42 5 39',stroke='#c398a1',w=.8)
    faces+=group(i,face)
    eye='';brow=''
    for side in [-1,1]:
        cx=side*41;scale=.94 if side==1 else 1
        u=v['upper'];lo=v['lower']
        d=f'M{cx-25} 0C{cx-12} {u} {cx+11} {u-1} {cx+25} -1Q{cx+10} {lo} {cx-3} {lo*.78}Q{cx-17} {lo*.6} {cx-25} 0Z'
        # Clip each iris to its newly drawn opening, rather than painting a
        # raised lid over a still fully exposed round eye.
        whites+=group(i,p(d,'#fffaf3'))
        iris=el(cx+1,0,12.2,17,'url(#iris)')+el(cx+1,-1,4.6,9.4,'#34516e')
        iris+=el(cx-3,-6,3.3,4.3,'#e4f5ef',.85)+el(cx+6,7,1.5,2,'#eff8e6',.8)
        iris_layers.append({'name':f'04b Iris {i} {side}', 'svg':svg(group(i,iris)),
                            'svg_masks':[svg(group(i,p(d,'#fff')))]})
        eye+=p(f'M{cx-25} 0C{cx-12} {u} {cx+11} {u-1} {cx+25} -1',stroke='#64657d',w=1.7)
        eye+=p(f'M{cx-21} {lo*.31}Q{cx+2} {lo+1} {cx+22} 3',stroke='#b297a6',w=.7)
        outer=cx+side*23
        eye+=p(f'M{outer} -2Q{outer+side*5} -5 {outer+side*7} -9',stroke='#66677e',w=.9)
        outer_y=-26+v['brow'][0];inner_y=-26+v['brow'][1]
        # Left/right ordering is explicit so inward brow movement remains
        # inward on both sides, instead of mirroring the emotional intent.
        if side<0:a,b=outer_y,inner_y
        else:a,b=inner_y,outer_y
        brow+=p(f'M{cx-22} {a}Q{cx} {min(a,b)-5} {cx+22} {b}',stroke='#8495a6',w=1.9)
    eyes+=group(i,eye);brows+=group(i,brow)
    typ=v['mouth']
    if typ=='smile':
        if REVIEW:m=p('M-16 62Q0 71 16 61',stroke='#a27489',w=.9)+p('M-6 69Q1 71 7 68',stroke='#d7a5ad',w=1.1,op=.4)
        else:m=p('M-16 61Q0 73 16 60Q12 71 0 72Q-11 71 -16 61Z','#b77f94',op=.6)+p('M-16 61Q0 73 16 60',stroke='#a27489',w=.8)
    elif typ=='open':m=el(1,68,8,11,'#b67f91')+el(2,73,5,3.3,'#e0a9b4')
    elif typ=='sad':m=p('M-12 69Q1 59 14 69',stroke='#b17e91',w=1)
    elif typ=='firm':m=p('M-13 66Q0 65 14 64',stroke='#a67a8e',w=1)
    else:m=p('M-12 65Q0 64 12 65',stroke='#b17f92',w=.85)
    mouths+=group(i,m)
    if v.get('cheek'):
        cheeks+=group(i,el(-53,24,22,13,'url(#cheek)')+el(54,22,20,12,'url(#cheek)'))
layers=[{'name':'03 Shared head and simplified hair','svg':svg(faces),'visibility':{'02 Head envelope and fixed feature coordinates':False}},
        {'name':'04 Expression warm eye whites','svg':svg(whites)},*iris_layers,
        {'name':'04c Expression eyelid contours','svg':svg(eyes)},
        {'name':'05 Expression eyebrows','svg':svg(brows)},
        {'name':'06 Expression mouths','svg':svg(mouths)},
        {'name':'07 Cheek support for coordinated smile','svg':svg(cheeks),'checkpoint':'02-expression-first-pass.png'}]
two={**one,'stem':'expression-comparison','layers':layers,'expected_document':str((OUT/'painting/expression-construction.kra').resolve())}
if REVIEW:
    two.update(stem='expression-reviewed',expected_document=str((OUT/'painting/expression-comparison.kra').resolve()))
    (OUT/'03-reviewed.json').write_text(json.dumps(two),encoding='utf-8')
else:(OUT/'02-expressions.json').write_text(json.dumps(two),encoding='utf-8')
(OUT/'elements.json').write_text(json.dumps({'topic':'anime facial expressions','independent_from':'hair material practice','main_document_mutated':False,'camera':{'yaw_approx':13,'roll':-5,'same_for_all':True},'fixed':['head envelope','eye spacing','nose position','hair silhouette','iris palette'],'variables':PARAMS,'evaluation':['read emotion without labels','compare mouth-only with coordinated smile','eyelids occlude iris consistently','inward eyebrows move in same emotional direction','mouth follows head center and cheek curve'],'limits':['simplified shallow 3/4 projection','instructional studies not finished portrait expressions','no claims of universal one-to-one emotion mapping']},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared independent expression construction and six controlled variations')
