"""Prepare isolated reference-inspired studies; rendering is done by Krita MCP.

No bridge connection or writes on import. Original reference pixels are excluded
from the scenes. Run one stage, inspect its export, then choose the next stage.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/test-result/20261017/krita-puracotte-head-study'
LOCK = 'M60 70C20 145 25 195 84 220C139 245 140 280 79 320C39 347 45 375 90 390C47 356 105 350 137 315C180 269 136 222 93 198C59 175 66 112 60 70Z'
OVER = 'M60 70C56 132 67 177 103 201C133 221 150 233 153 259C142 232 111 227 80 207C41 183 43 127 60 70Z'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def p(d, fill='none', stroke='none', w=1, opacity=1):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"/>'


def svg(body):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 600 450">'+body+'</svg>'


def hair(stage):
    levels = {'outline':0, 'flats':1, 'shade':2, 'shade-revised':2, 'detail':3}
    level = levels[stage]
    layers = [{'name':'Paper and labels','svg':svg('<rect width="600" height="450" fill="#f7f3f5"/>'+''.join(f'<text x="{15+i*200}" y="30" font-size="10" fill="#66566f">{t}</text>' for i,t in enumerate(['A / base silhouette','B / form + overlap','C / selective strands'])))}]
    defs = '''<defs>
<linearGradient id="mass" x1="10%" y1="0%" x2="90%" y2="100%"><stop stop-color="#fff6fe"/><stop offset=".23" stop-color="#e8d9f5"/><stop offset=".42" stop-color="#a99cc8"/><stop offset=".56" stop-color="#f0dff5"/><stop offset=".76" stop-color="#c3b4df"/><stop offset="1" stop-color="#ffe3f4"/></linearGradient>
<linearGradient id="upper" x1="0%" y1="0%" x2="100%" y2="20%"><stop stop-color="#aaa0c7"/><stop offset=".28" stop-color="#e8d7f1"/><stop offset=".58" stop-color="#fff9ff"/><stop offset="1" stop-color="#e5c6e9"/></linearGradient>
<linearGradient id="light" x1="0%" y1="0%" x2="20%" y2="100%"><stop stop-color="#fffaff" stop-opacity=".9"/><stop offset="1" stop-color="#fffaff" stop-opacity="0"/></linearGradient>
</defs>'''
    for i in range(3):
        x = i*200
        fill = '#e8dbf3' if level else 'none'
        body = p(LOCK, fill, '#a997b3', .8)+p(OVER, fill, '#a997b3', .65)
        if level == 0:
            body += p('M60 75C33 155 61 190 99 218C167 266 109 313 74 345', stroke='#c4b7ce', w=.7)
            body += p('M32 157L83 158M64 211L104 194M106 248L143 235M90 320L123 325', stroke='#c4b7ce', w=.6)
        if level >= 2 and i:
            body = p(LOCK,'url(#mass)','#a997b3',.55)
            body += p('M54 174C67 201 105 212 134 239C147 255 146 273 138 286C142 256 128 243 99 230C73 217 56 203 54 174Z','#9684b3',opacity=.6)
            body += p('M79 320C48 343 48 372 90 390C63 367 72 348 92 334C101 328 109 321 116 314C104 319 91 320 79 320Z','#a999c5',opacity=.4)
            body += p('M64 82C43 141 47 179 77 196C66 168 68 111 64 82Z','url(#light)')
            body += p('M119 234C139 253 133 277 107 299C124 287 149 261 132 239Z','#fff5ff',opacity=.8)
            body += p('M72 341C65 357 73 368 90 378C76 361 90 347 107 337Z','#ffe8f5',opacity=.8)
            body += p(OVER,'url(#upper)','#b39fbd',.5)
            body += p('M54 171C65 196 101 211 127 229',stroke='#85719c',w=1.25,opacity=.65)
            body += p('M132 306C118 327 90 343 80 356',stroke='#edc6e8',w=1.8,opacity=.85)
        item={'name':f'{i} lock planes','svg':svg(defs+f'<g transform="translate({x} 0)">'+body+'</g>')}
        if level >= 2:
            item['svg_masks']=[svg(f'<g transform="translate({x} 0)">'+p(LOCK,'#fff')+p(OVER,'#fff')+'</g>')]
        layers.append(item)
    if level == 3:
        # Sparse marks, not a comb of equal full-length lines. No taper sampling
        # accumulation: these are individual native brush paths.
        paths = [
            [['M',459,100],['C',451,140,458,174,481,191]],
            [['M',465,124],['C',464,158,477,182,503,200]],
            [['M',479,214],['C',501,226,524,241,529,253]],
            [['M',536,276],['C',529,294,515,308,501,318]],
            [['M',495,339],['C',479,352,476,363,485,374]],
        ]
        layers.append({'name':'C native deevad selective fine strands','opacity':.48,'strokes':[{'brush':'c4) Thin Brush Regular - deevad 25.01','color':'#9b83ad','size':.5,'opacity':.7,'path':path,'taper':False} for path in paths]})
    layers.append({'name':'Footer','svg':svg('<text x="18" y="426" font-size="8" fill="#88768e">White light / cool form shadow / narrow contact shadow / warm reflected edge</text>')})
    write('hair-'+stage+'.json',{'name':'Reference hair volume '+stage,'stem':'hair-'+stage,'width':1200,'height':900,'scale':2,'layers':layers})


def blush(revised=False):
    s=json.loads((ROOT/'docs/krita-anime-painting/examples/face-proportions-20261017/final-scene.json').read_text(encoding='utf-8'))
    mask=next(n for n in s['layers'] if n['name'].startswith('43 '))
    root='<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1680" viewBox="-160 -205 320 373.3333333333333">'
    defs='''<defs><radialGradient id="rose"><stop stop-color="#f5789a" stop-opacity=".38"/><stop offset=".5" stop-color="#f79aaf" stop-opacity=".20"/><stop offset="1" stop-color="#f8a8b7" stop-opacity="0"/></radialGradient><radialGradient id="bridge"><stop stop-color="#fa9bad" stop-opacity=".13"/><stop offset="1" stop-color="#f7a9ba" stop-opacity="0"/></radialGradient></defs>'''
    body=defs+''.join(f'<ellipse cx="{cx}" cy="24" rx="30" ry="18" fill="url(#rose)"/>' for cx in [-39,39])+'<ellipse cx="0" cy="22" rx="29" ry="9" fill="url(#bridge)"/>'
    s['layers'].append({'name':'45 Soft upper cheek blush','svg':root+body+'</svg>','svg_masks':copy.deepcopy(mask['svg_masks']),'svg_erase_masks':copy.deepcopy(mask['svg_erase_masks'])})
    paths=[]
    for side in [-1,1]:
        for j,(a,y,length) in enumerate([(26,18,5),(29,19,7),(33,18,8),(37,19,9),(41,20,7),(45,20,8),(49,21,6),(52,22,4)]):
            x=side*a
            if revised and side==1:y+=(-1,1,0,-1,2,0,1,0)[j]
            paths.append({'brush':'c4) Thin Brush Regular - deevad 25.01','color':'#db879d','size':(.44 if j%3 else .36) if revised else (.25 if j%3 else .2),'opacity':.9 if revised else .65,'path':[['M',x+160,y+205],['L',x+160+.8,y+length+205]],'taper':False})
    if revised=='fine':
        paths=[]
        for side,count,start in [(-1,10,22),(1,16,25)]:
            for j in range(count):
                x=160+side*(start+j*1.35)
                y=205+17+(0,1.2,.3,2,.7)[j%5]
                length=(3.4,4.8,3.9,5.5,4.3,3)[j%6]
                paths.append({'brush':'c4) Thin Brush Regular - deevad 25.01','color':'#d989a2','size':.28+(j%3)*.025,'opacity':.9,'path':[['M',x,y],['L',x+.5,y+length]],'taper':False})
    s['layers'].append({'name':'46 Native sparse blush hatching','opacity':.7 if revised else .36,'strokes':paths})
    s.update(name='Reference blush principles on independent B study',stem='blush-study',variants=[{'file':'blush-soft-only.png','visibility':{'46 Native sparse blush hatching':False}},{'file':'blush-with-hatching.png','visibility':{'46 Native sparse blush hatching':True}}],details=[{'file':'blush-face.png','rect':[84,167,154,140]}])
    write('blush-study-fine.json' if revised=='fine' else ('blush-study-r2.json' if revised else 'blush-study.json'),s)


def elements():
    write('elements.json',{'reference':'user supplied 149567189_p0.jpg; no reference pixels in study scenes','reference_landmarks_approx_pixels':{'eye_left':[345,463],'eye_right':[521,403],'nose':[433,515],'mouth':[463,546],'chin':[480,606]},'reference_orientation':{'eye_axis_degrees_image':-19,'uncertainty':'approximate hand observation, not yaw or 3D camera calibration'},'blush_study':{'baseline':'B retained near frontal; expression unchanged','soft_color_centres':[[-39,24],[39,24]],'separate_soft_blush_and_hatch_layers':True,'no_hair_restyle_in_this_trial':True},'hair_study':{'isolated_s_curve':'simplified volume exercise, not reconstruction of hairstyle','light':'broad upper light','value_order':['near-white front plane','lavender half-tone','cool violet turning plane','narrow darker contact seam'],'details':'sparse interrupted native strokes, only after planes'},'workflow':['outline and actual visual review','flats and actual visual review','shading and actual visual review','detail and actual visual review','blush soft/hatch comparison'],'new_downloads':False,'native_brush':'c4) Thin Brush Regular - deevad 25.01'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['elements','outline','flats','shade','shade-revised','detail','blush','blush-revised','blush-fine'])
    stage=parser.parse_args().stage
    if stage=='elements':elements()
    elif stage in ('blush','blush-revised','blush-fine'):blush('fine' if stage=='blush-fine' else stage=='blush-revised')
    else:hair(stage)
    print('Prepared '+stage)
