"""Face-only iterative study. Run a single stage after reviewing the prior PNG."""
import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/test-result/20261017/krita-face-iterations'
W,H,SCALE=1920,2240,6
VB='-160 -205 320 373.3333333333333'
FACE='M-71 -65C-77 -17 -59 48 -28 74Q-6 92 7 95C53 72 74 24 72 -56C65 -111 -53 -120 -71 -65Z'
GUIDE='40 Face study nose and mouth construction'


def path(d,fill='none',stroke='none',width=1,opacity=1):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"/>'


def ellipse(x,y,rx,ry,fill,opacity=1):
    return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{opacity}"/>'


DEFS='''<defs>
<radialGradient id="tipTurn"><stop stop-color="#c797a8" stop-opacity=".48"/><stop offset=".6" stop-color="#d8a6b0" stop-opacity=".22"/><stop offset="1" stop-color="#e6b9bc" stop-opacity="0"/></radialGradient>
<radialGradient id="bridgeLight"><stop stop-color="#fff5e4" stop-opacity=".75"/><stop offset="1" stop-color="#fff0e2" stop-opacity="0"/></radialGradient>
<radialGradient id="lipSoft"><stop stop-color="#d39ba7" stop-opacity=".38"/><stop offset="1" stop-color="#eac0c0" stop-opacity="0"/></radialGradient>
<linearGradient id="noseSide" x1="0%" y1="0%" x2="100%" y2="10%"><stop stop-color="#c498aa" stop-opacity="0"/><stop offset=".55" stop-color="#b895a8" stop-opacity=".12"/><stop offset="1" stop-color="#b895a8" stop-opacity=".25"/></linearGradient>
<linearGradient id="noseCast" x1="20%" y1="0%" x2="70%" y2="100%"><stop stop-color="#b98d9e" stop-opacity=".4"/><stop offset="1" stop-color="#dcafb3" stop-opacity="0"/></linearGradient>
</defs>'''


def svg(body,defs=DEFS):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="{VB}">'+defs+body+'</svg>'


def resized(s):
    r=ET.fromstring(s)
    r.set('width',str(W));r.set('height',str(H));r.set('viewBox',VB)
    return ET.tostring(r,encoding='unicode')


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def details(stem):
    return [{'file':stem+'-face.png','rect':[84,167,154,140]},
            {'file':stem+'-nose-mouth.png','rect':[141,218,42,63]}]


def construction():
    scene=json.loads((ROOT/'docs/krita-anime-painting/examples/latest-head.scene.json').read_text(encoding='utf-8'))
    scene.update(width=W,height=H,scale=SCALE,name='Face iteration study — fixed camera and expression',stem='01-construction',blank_checkpoint='00-new-blank.kra')
    for n in scene['layers']:
        if 'svg' in n:n['svg']=resized(n['svg'])
        for k in ('svg_masks','svg_erase_masks'):
            if k in n:n[k]=[resized(s) for s in n[k]]
    guides=path('M1 10L1 71M-16 36H17M-17 62H18',stroke='#8495af',width=.35,opacity=.7)
    guides+=path('M0 14L-3 30Q-5 36 1 38Q8 35 5 29Z',stroke='#ad8495',width=.45)
    guides+=ellipse(1,32,3.5,4,'none')+path('M-8 62Q1 59.5 10 62Q1 64 -8 62Z',stroke='#ad8495',width=.4)
    scene['layers'].append({'name':GUIDE,'svg':svg(guides)})
    scene['variants']=[{'file':'00-baseline.png','visibility':{GUIDE:False}},{'file':'01-guides.png','visibility':{GUIDE:True}}]
    scene['details']=details('01-construction')
    write('01-construction.json',scene)
    write('elements.json',{'canvas':[W,H],'local_viewbox':VB,'scale':SCALE,'camera':'same near-frontal as source','expression':'neutral, tiny mouth opening; no smile','light':'broad upper viewer-left','fixed':['head contour','iris geometry','gaze','hair design'],'round1_goals':['connected nose planes instead of isolated pin light','small mouth opening and lip turn instead of straight line'],'source_scene':'docs/krita-anime-painting/examples/latest-head.scene.json','stages_are_adaptive':True})


def round1():
    scene=load('01-construction.json')
    face=copy.deepcopy(next(n for n in scene['layers'] if n['name'].startswith('Face placement')))
    r=ET.fromstring(face['svg'])
    # Last seven direct shapes are only the previous nose and lips. Preserve eyes.
    shapes=list(r)
    assert shapes[-7].get('cx')=='1' and shapes[-1].get('d').startswith('M-2 65')
    for n in shapes[-7:]:r.remove(n)
    face['svg']=ET.tostring(r,encoding='unicode')
    face['visibility']={GUIDE:False,'39 Unified cheek turn and small nose cast':False}
    # Preserve whole-face soft turn, remove the old isolated nose dab.
    light=copy.deepcopy(next(n for n in scene['layers'] if n['name'].startswith('39 ')))
    rr=ET.fromstring(light['svg'])
    for n in list(rr):
        if n.tag.split('}')[-1]=='ellipse':rr.remove(n)
    light['svg']=ET.tostring(rr,encoding='unicode');light['visible']=True
    nose=path('M2 13C3 22 4 27 6 30Q8 35 4 38Q7 34 4 30L1 25Z','url(#noseSide)')
    nose+=ellipse(2.5,35,5.5,4.4,'url(#tipTurn)',.8)
    nose+=path('M-1 35Q2 39 5 35L9 40Q5 43 2 39Z','url(#noseCast)',.55)
    nose+=ellipse(-1,26,2.5,9,'url(#bridgeLight)',.55)
    nose+=path('M-1.6 30Q-.2 28.7 1.2 30L1 32.5Q-.2 33.4 -1 32Z','#fff3e2',opacity=.75)
    nose+=path('M-.7 36Q1.1 37.4 2.6 36.4',stroke='#b58b9b',width=.27,opacity=.52)
    lips=ellipse(1,62.8,10,3.1,'url(#lipSoft)',.65)
    lips+=path('M-7.8 62Q-3 60.9 .7 61.6Q5 60.9 9 62Q4.2 64.2 .6 63.9Q-3 63.3 -7.8 62Z','#b78394',opacity=.5)
    lips+=path('M-6.7 62Q-.4 61.2 1 61.8Q4.9 61.4 8 62Q3.8 62.9 .7 62.7Z','#916d83',opacity=.55)
    lips+=path('M-1.8 65Q1 65.8 4.6 64.8',stroke='#fff4e4',width=.5,opacity=.63)
    added={'name':'41 Nose planes and quiet lip opening','svg':svg(nose+lips)}
    revision={k:scene[k] for k in ('width','height','scale','name')}
    revision.update(stem='02-round1-nose-lips',expected_document=str(OUT/'painting/01-construction.kra'),layers=[face,light],details=details('02-round1'))
    write('02-round1-revise.json',revision)
    append={**revision,'expected_document':str(OUT/'painting/02-round1-nose-lips.kra'),'stem':'02-round1-complete','layers':[added]}
    write('02-round1-append.json',append)
    for update in (face,light):
        next(n for n in scene['layers'] if n['name']==update['name']).update(copy.deepcopy(update))
    scene['layers'].append(added)
    next(n for n in scene['layers'] if n['name']==GUIDE)['visible']=False
    for k in ('variants','blank_checkpoint'):scene.pop(k,None)
    scene.update(stem='round1-snapshot',details=details('round1-snapshot'))
    write('round1-snapshot.json',scene)


def round2():
    scene=load('round1-snapshot.json')
    update=copy.deepcopy(next(n for n in scene['layers'] if n['name'].startswith('41 ')))
    nose=ellipse(3.9,29,2.8,7.5,'url(#tipTurn)',.33)
    nose+=ellipse(1.8,35,4.8,3.2,'url(#tipTurn)',.53)
    nose+=path('M0 35.2Q2.5 38 4.5 35.5L6.5 38Q4 40 2 37.8Z','url(#noseCast)',.3)
    nose+=ellipse(-.6,30.6,1.9,3.4,'url(#bridgeLight)',.9)
    nose+=ellipse(-1,23,1.7,7,'url(#bridgeLight)',.3)
    nose+=path('M.2 36.2Q1.5 37 2.8 36.3',stroke='#b98e9c',width=.24,opacity=.38)
    lips=ellipse(1,63,8.5,2.4,'url(#lipSoft)',.42)
    lips+=path('M-7 62Q-2.7 61.45 .8 61.9Q4.3 61.5 8 62Q3.5 63 .7 62.8Q-3 62.6 -7 62Z','#aa7c8e',opacity=.49)
    lips+=path('M-3.8 61.95Q.5 61.7 4.9 62.05Q1 62.5 -3.8 61.95Z','#8d687e',opacity=.45)
    lips+=path('M-.8 65Q1.5 65.4 3.9 64.9',stroke='#fff0df',width=.35,opacity=.5)
    update['svg']=svg(nose+lips)
    revision={k:scene[k] for k in ('width','height','scale','name')}
    revision.update(stem='03-round2-quiet-nose-lips',expected_document=str(OUT/'painting/02-round1-complete.kra'),layers=[update],details=details('03-round2'))
    write('03-round2-revise.json',revision)
    next(n for n in scene['layers'] if n['name']==update['name']).update(copy.deepcopy(update))
    scene.update(stem='round2-snapshot',details=details('round2-snapshot'))
    write('round2-snapshot.json',scene)


def round3():
    scene=load('round2-snapshot.json')
    face=copy.deepcopy(next(n for n in scene['layers'] if n['name'].startswith('Face placement')))
    r=ET.fromstring(face['svg'])
    old_upper=['M-60 0Q-37 -17 -14 -1L','M14 0Q37 -17 60 -1L']
    lower=['M-56 4Q-37 17 -17 3','M18 4Q37 17 57 3']
    folds=['M-57 -10Q-37 -23 -18 -10','M17 -10Q37 -23 56 -10']
    for i,n in enumerate(list(r)):
        d=n.get('d','')
        if any(d.startswith(x) for x in old_upper):r.remove(n)
        elif n.get('fill')=='#737087':r.remove(n)
        elif d in lower:
            n.set('d','M-57 3Q-52 8 -47 9.2' if d==lower[0] else 'M47 9.2Q52 8 57 3')
            n.set('stroke-width','.32');n.set('opacity','.55')
        elif d in folds:
            n.set('d','M-54 -11Q-38 -18 -22 -12' if d==folds[0] else 'M22 -12Q38 -18 54 -11')
            n.set('stroke-width','.28');n.set('opacity','.52')
        elif n.get('stroke-width')=='0.11':
            n.set('opacity',str(.08 if i%4==0 else .16+(i%3)*.045))
            n.set('stroke','#9c8c9e')
        elif n.get('fill')=='#9d9cac':n.set('fill','#9d8b9e');n.set('opacity','.58')
    defs=next(n for n in r if n.tag.split('}')[-1]=='defs')
    defs.append(ET.fromstring('<linearGradient id="eyeWhite" gradientUnits="userSpaceOnUse" x1="0" y1="-10" x2="0" y2="11"><stop stop-color="#d8ccd8"/><stop offset=".42" stop-color="#f0e7e4"/><stop offset="1" stop-color="#fff9ee"/></linearGradient>'))
    for n in r:
        if n.get('fill')=='#fcf8f0':n.set('fill','url(#eyeWhite)')
    face['svg']=ET.tostring(r,encoding='unicode')
    light=copy.deepcopy(next(n for n in scene['layers'] if n['name'].startswith('39 ')))
    rr=ET.fromstring(light['svg'])
    for n in rr:
        if n.get('fill')=='url(#cheekTurn)':n.set('opacity','.38')
    light['svg']=ET.tostring(rr,encoding='unicode')
    lid=path('M-61 -.5C-49 -11 -30 -12 -14 -1L-14 .35C-30 -7.5 -48 -7.6 -59.5 1.1Z','#615c73')
    lid+=path('M61 -.5C49 -11 30 -12 14 -1L14 .35C30 -7.5 48 -7.6 59.5 1.1Z','#615c73')
    lash=path('M-60 0Q-63 -2 -65 -5Q-63 0 -60 1Z','#615c73',opacity=.85)
    lash+=path('M-57 -3Q-61 -6 -61 -8Q-60 -4 -56 -2Z','#706780',opacity=.75)
    lash+=path('M-54 -5Q-57 -8 -56.7 -10Q-56 -7 -53 -4Z','#80728a',opacity=.55)
    lid+=lash+'<g transform="scale(-1,1)">'+lash+'</g>'
    lid+=path('M-26 8.5Q-20 7 -17 4',stroke='#b594a6',width=.25,opacity=.45)
    lid+=path('M17 4Q20 7 26 8.5',stroke='#b594a6',width=.25,opacity=.45)
    added={'name':'42 Tapered lids over iris and broken lower rim','svg':svg(lid),
           'svg_erase_masks':copy.deepcopy(light['svg_erase_masks'][:-2])}
    revision={k:scene[k] for k in ('width','height','scale','name')}
    revision.update(stem='04-round3-eye-context',expected_document=str(OUT/'painting/03-round2-quiet-nose-lips.kra'),layers=[face,light],details=details('04-round3'))
    write('04-round3-revise.json',revision)
    write('04-round3-append.json',{**revision,'expected_document':str(OUT/'painting/04-round3-eye-context.kra'),'stem':'04-round3-complete','layers':[added]})
    for update in (face,light):next(n for n in scene['layers'] if n['name']==update['name']).update(copy.deepcopy(update))
    scene['layers'].append(added)
    scene.update(stem='round3-snapshot',details=details('round3-snapshot'))
    write('round3-snapshot.json',scene)


def round4():
    scene=load('round3-snapshot.json')
    lid=copy.deepcopy(next(n for n in scene['layers'] if n['name'].startswith('42 ')))
    lid['svg']=lid['svg'].replace('C-30 -7.5 -48 -7.6 -59.5 1.1Z','C-30 -10.4 -48 -9.6 -59.5 .9Z')
    lid['svg']=lid['svg'].replace('C30 -7.5 48 -7.6 59.5 1.1Z','C30 -10.4 48 -9.6 59.5 .9Z')
    lid['svg']=lid['svg'].replace('#615c73','#6b627a').replace('M-60 0Q-63 -2 -65 -5Q-63 0 -60 1Z','M-60 0Q-62.5 -1.8 -63.8 -3.6Q-62 .1 -60 .7Z')
    revision={k:scene[k] for k in ('width','height','scale','name')}
    revision.update(stem='05-round4-lid-refined',expected_document=str(OUT/'painting/04-round3-complete.kra'),layers=[lid],details=details('05-round4')+[{'file':'05-round4-eyes.png','rect':[96,179,127,42]}])
    write('05-round4-revise.json',revision)
    next(n for n in scene['layers'] if n['name']==lid['name']).update(copy.deepcopy(lid))
    scene.update(stem='face-iterations-final-snapshot',details=details('final-snapshot'))
    write('final-scene-snapshot.json',scene)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['construction','round1','round2','round3','round4'])
    args=parser.parse_args()
    {'construction':construction,'round1':round1,'round2':round2,'round3':round3,'round4':round4}[args.stage]()
    print('Prepared '+args.stage+'; render/revise only after prior visual review.')


if __name__=='__main__':main()
