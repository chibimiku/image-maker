"""Adaptive baseline/A/B face proportion and plane study; no imports with side effects."""
import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/test-result/20261017/krita-face-proportions'
FACE='M-71 -65C-77 -17 -59 48 -28 74Q-6 92 7 95C53 72 74 24 72 -56C65 -111 -53 -120 -71 -65Z'
SHORT='M-71 -65C-77 -17 -58 42 -28 66Q-6 83 7 86C50 66 74 24 72 -56C65 -111 -53 -120 -71 -65Z'


def p(d,fill='none',stroke='none',w=1,op=1):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'


def svg(s,board=False):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="2100" height="1000" viewBox="-110 -115 630 300">' if board else '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1680" viewBox="-160 -205 320 373.3333333333333">')+s+'</svg>'


def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def construction(reviewed=False):
    s='<rect x="-110" y="-115" width="630" height="300" fill="#f6f3ee"/>'
    for i,(label,shape,chin,mouth) in enumerate([('BASELINE',FACE,95,62),('A - same proportions',FACE,95,62),('B - shorter lower face',SHORT,86,57)]):
        x=i*210
        s+=f'<g transform="translate({x} 0)"><text x="-85" y="{-108 if reviewed else -99}" fill="#4c4c60" font-size="6.5">{label}</text>'
        neck=p('M-28 72L-28 111Q-38 124 -61 132M28 72L28 111Q38 124 61 132',stroke='#a7a6ad',w=.7)
        s+=('<g transform="translate(0 -9)">'+neck+'</g>') if reviewed and i==2 else neck
        if i==2:s+=p(FACE,stroke='#bc9d9d',w=.5,op=.45)
        s+=p(shape,'#e6e1dc','#686978',.7)
        s+=p(f'M1 -78V{chin}M-69 0H70M-40 36H40M-25 {mouth}H25',stroke='#8d9bae',w=.4,op=.7)
        s+=p('M-60 0Q-37 -17 -14 -1Q-37 20 -60 0ZM14 0Q37 -17 60 -1Q37 20 14 0Z',stroke='#777987',w=.5)
        s+=p(f'M-8 {mouth}H10',stroke='#8e7c89',w=.8)
        s+=f'<text x="-70" y="155" font-size="5" fill="#666575">nose 36 / mouth {mouth} / chin {chin}</text></g>'
    filename,stem=('01b-construction-reviewed.json','01b-proportion-construction-reviewed') if reviewed else ('01-construction.json','01-proportion-construction')
    write(filename,{'name':'Face proportions - controlled construction','stem':stem,'width':2100,'height':1000,'scale':2100/630,'layers':[{'name':'Three structure views','svg':svg(s,True)}]})
    write('elements.json',{'reference':'same R4 face; no reference pixels copied','camera':'same near frontal','light':'broad upper viewer left','baseline':{'nose':36,'mouth':62,'chin':95},'A':{'same_shape':True,'new_cheek_planes':True},'B':{'nose':36,'mouth':57,'chin':86,'neck_shoulders_translate_y':-9,'same_new_planes_as_A':True},'new_brush':'d) Glazing Round - deevad 25.01; optional low-opacity independent glaze only','workflow':['construction and actual review','untextured grey planes and actual review','color A/B and actual review','revise selected candidate']})


def planes():
    s='<rect x="-110" y="-115" width="630" height="300" fill="#f6f3ee"/>'
    defs='''<defs><linearGradient id="turn" x1="0%" x2="100%"><stop stop-color="#777" stop-opacity="0"/><stop offset=".5" stop-color="#777" stop-opacity=".08"/><stop offset="1" stop-color="#777" stop-opacity=".42"/></linearGradient><radialGradient id="cheek"><stop stop-color="#777" stop-opacity=".22"/><stop offset="1" stop-color="#777" stop-opacity="0"/></radialGradient><radialGradient id="light"><stop stop-color="#fff" stop-opacity=".25"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient></defs>'''
    layers=[{'name':'Paper','svg':svg(s,True)}]
    for i,(label,shape,mouth) in enumerate([('BASELINE flat',FACE,62),('A planes / original length',FACE,62),('B planes / shorter lower face',SHORT,57)]):
        x=i*210
        base=f'<g transform="translate({x} 0)"><text x="-85" y="-99" fill="#4c4c60" font-size="6.5">{label}</text>'+p(shape,'#d8d8d8','#85858c',.55)+'</g>'
        layers.append({'name':label+' base','svg':svg(base,True)})
        if i:
            shade=p(shape,'url(#turn)')+'<ellipse cx="48" cy="48" rx="23" ry="32" fill="url(#cheek)"/><ellipse cx="-24" cy="20" rx="35" ry="43" fill="url(#light)"/><ellipse cx="40" cy="18" rx="22" ry="12" fill="url(#light)"/>'
            layers.append({'name':label+' planes','svg':svg(defs+f'<g transform="translate({x} 0)">'+shade+'</g>',True),'svg_masks':[svg(f'<g transform="translate({x} 0)">'+p(shape,'#fff')+'</g>',True)]})
        features=p('M-60 0Q-37 -17 -14 -1M14 0Q37 -17 60 -1',stroke='#666',w=.8)
        features+=p(f'M-8 {mouth}Q1 {mouth-.5} 10 {mouth}',stroke='#777',w=.5)+p('M0 36L3 37',stroke='#999',w=.5)
        layers.append({'name':label+' feature anchors','svg':svg(f'<g transform="translate({x} 0)">'+features+'</g>',True)})
    write('02-grey-planes.json',{'name':'Face planes without texture','stem':'02-grey-planes','width':2100,'height':1000,'scale':2100/630,'layers':layers})


def wrap_content(s,dy=0):
    r=ET.fromstring(s)
    group=ET.Element('g',{'transform':f'translate(0 {dy})'})
    for n in list(r):
        if n.tag.split('}')[-1]!='defs':r.remove(n);group.append(n)
    r.append(group)
    return ET.tostring(r,encoding='unicode')


def color(variant):
    scene=json.loads((ROOT/'docs/krita-anime-painting/examples/face-iterations-20261017/final-scene.json').read_text(encoding='utf-8'))
    for n in scene['layers']:
        for key in ('svg','svg_masks','svg_erase_masks'):
            if key not in n:continue
            values=[n[key]] if key=='svg' else n[key]
            mapped=[]
            for s in values:
                if variant=='B':s=s.replace(FACE,SHORT)
                r=ET.fromstring(s);r.set('width','1440');r.set('height','1680')
                mapped.append(ET.tostring(r,encoding='unicode'))
            n[key]=mapped[0] if key=='svg' else mapped
        if n['name'].startswith(('35 ','39 ')):n['visible']=False
        if variant=='B':
            if n['name'].startswith('Face placement'):
                r=ET.fromstring(n['svg'])
                shapes=[v for v in r if v.tag.split('}')[-1]!='defs']
                for v in shapes[:2]:v.set('transform','translate(0 -9)')
                n['svg']=ET.tostring(r,encoding='unicode')
            elif n['name'].startswith('37 '):
                n['svg']=wrap_content(n['svg'],-9)
                n['svg_masks']=[wrap_content(s,-9) for s in n['svg_masks']]
                n['svg_erase_masks'][1]=wrap_content(n['svg_erase_masks'][1],-9)
            elif n['name'].startswith('41 '):
                r=ET.fromstring(n['svg'])
                for v in list(r)[-4:]:v.set('transform','translate(0 -5)')
                n['svg']=ET.tostring(r,encoding='unicode')
    face=SHORT if variant=='B' else FACE
    defs='''<defs>
<linearGradient id="faceTurn" gradientUnits="userSpaceOnUse" x1="-25" y1="10" x2="78" y2="40"><stop stop-color="#9f8da8" stop-opacity="0"/><stop offset=".48" stop-color="#ae92a6" stop-opacity=".035"/><stop offset="1" stop-color="#9f8da8" stop-opacity=".3"/></linearGradient>
<radialGradient id="cheekTurn"><stop stop-color="#b58e9f" stop-opacity=".14"/><stop offset="1" stop-color="#d8acb3" stop-opacity="0"/></radialGradient>
<radialGradient id="warmFace"><stop stop-color="#fff1d9" stop-opacity=".3"/><stop offset="1" stop-color="#fff0df" stop-opacity="0"/></radialGradient>
<radialGradient id="cheekTop"><stop stop-color="#ffeddb" stop-opacity=".13"/><stop offset="1" stop-color="#fff0df" stop-opacity="0"/></radialGradient>
</defs>'''
    body=p(face,'url(#faceTurn)')+'<ellipse cx="48" cy="48" rx="23" ry="32" fill="url(#cheekTurn)"/><ellipse cx="-24" cy="20" rx="35" ry="43" fill="url(#warmFace)"/><ellipse cx="40" cy="18" rx="22" ry="12" fill="url(#cheekTop)"/>'
    old=next(n for n in scene['layers'] if n['name'].startswith('39 '))
    scene['layers'].append({'name':'43 Cheek form and restrained warm light','svg':svg(defs+body),'svg_masks':[svg(p(face,'#fff'))],'svg_erase_masks':copy.deepcopy(old['svg_erase_masks'])})
    # Small, contained marks; native strokes are not automatically masked by
    # svg_masks in the renderer, so keep the complete brush footprint inside skin.
    glaze='44 Deevad cheek glazing trial'
    scene['layers'].append({'name':glaze,'opacity':.05 if variant=='B' else 1,'strokes':[{'brush':'d) Glazing Round - deevad 25.01','color':'#bb9aa9','size':10,'opacity':.065,'path':[['M',194,230],['Q',198,242,189,252]],'taper':True,'samples':96}]})
    scene.update(name='Face proportions '+variant+' — color context',stem='03-color-'+variant,width=1440,height=1680,scale=4.5,
        variants=[{'file':'03-color-'+variant+'-no-glaze.png','visibility':{glaze:False}},{'file':'03-color-'+variant+'-with-glaze.png','visibility':{glaze:True}}],
        details=[{'file':'03-color-'+variant+'-face.png','rect':[84,167,154,140]}])
    for k in ('blank_checkpoint','expected_document'):scene.pop(k,None)
    write('03-color-'+variant+'.json',scene)


def fix_A():
    scene=json.loads((OUT/'03-color-A.json').read_text(encoding='utf-8'))
    glaze='44 Deevad cheek glazing trial'
    rev={k:scene[k] for k in ('width','height','scale','name')}
    rev.update(stem='04-A-glaze-opacity-corrected',expected_document=str(OUT/'color-A/03-color-A.kra'),layers=[{'name':glaze,'opacity':.05}],
        variants=[{'file':'04-A-no-glaze.png','visibility':{glaze:False}},{'file':'04-A-light-glaze.png','visibility':{glaze:True}}],
        details=[{'file':'04-A-face.png','rect':[84,167,154,140]}])
    write('04-A-glaze-correction.json',rev)
    next(n for n in scene['layers'] if n['name']==glaze)['opacity']=.05
    scene['stem']='A-corrected-snapshot';write('A-corrected-snapshot.json',scene)


def retain_B():
    scene=json.loads((OUT/'03-color-B.json').read_text(encoding='utf-8'))
    glaze='44 Deevad cheek glazing trial'
    rev={k:scene[k] for k in ('width','height','scale','name')}
    rev.update(stem='05-B-retained-no-glaze',expected_document=str(OUT/'color-B/03-color-B.kra'),layers=[{'name':glaze,'visible':False}],details=[{'file':'05-B-retained-face.png','rect':[84,167,154,140]}])
    write('05-B-retain.json',rev)
    # Hidden native experiment remains in the KRA. Portable final snapshot omits
    # that rejected layer, so replay does not require the optional brush bundle.
    scene['layers']=[n for n in scene['layers'] if n['name']!=glaze]
    scene['stem']='face-proportion-B-retained';scene.pop('variants',None)
    scene['details']=[{'file':'face-proportion-B-face.png','rect':[84,167,154,140]}]
    write('final-scene.json',scene)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['construction','construction-reviewed','planes','color-A','color-B','fix-A','retain-B']);args=parser.parse_args()
    {'construction':construction,'construction-reviewed':lambda:construction(True),'planes':planes,'color-A':lambda:color('A'),'color-B':lambda:color('B'),'fix-A':fix_A,'retain-B':retain_B}[args.stage]()
    print('Prepared '+args.stage)


if __name__=='__main__':main()
