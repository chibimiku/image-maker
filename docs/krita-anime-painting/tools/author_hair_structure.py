"""Anime hair structure practice: skull -> unified masses -> selective detail."""
import json
import random
from pathlib import Path

OUT=Path('data/test-result/20261016/krita-hair-structure-study')
OUT.mkdir(exist_ok=True)
R=random.Random(161031)
BACK='M-79 -30C-96 -114 -50 -149 0 -148C77 -151 106 -100 91 -24C89 30 104 75 116 118C124 151 130 172 146 190Q121 189 110 169C122 218 115 240 107 254Q86 239 84 217C76 252 55 277 28 280Q31 261 24 240C5 256 -30 282 -59 283Q-51 263 -49 242C-75 263 -101 272 -128 270Q-112 240 -109 217Q-128 232 -145 231C-107 181 -110 122 -93 43Q-83 9 -79 -30Z'
FRONT='M-79 -30C-96 -114 -50 -149 0 -148C77 -151 106 -100 91 -24Q89 23 102 65Q84 48 76 21Q90 109 81 151Q60 116 59 58L61 -35Q49 -57 34 -86Q23 -54 -5 -35Q-11 -30 -17 -29L-11 -42Q-36 -14 -59 -13Q-62 14 -68 47Q-66 112 -79 151Q-94 122 -87 75Q-99 94 -105 103Q-99 66 -91 35Q-82 11 -79 -30Z'
FACE='M-62 -56C-70 -110 -27 -128 4 -125C54 -123 74 -90 71 -45C70 20 47 58 8 82Q-8 81 -39 58Q-71 18 -62 -56Z'
DEFS='''<defs><linearGradient id="front" x1="0%" y1="0%" x2="100%" y2="20%"><stop stop-color="#95b8cf"/><stop offset=".32" stop-color="#d7e8ed"/><stop offset=".6" stop-color="#b2cddd"/><stop offset="1" stop-color="#7e9fb9"/></linearGradient><linearGradient id="back" x1="0%" y1="0%" x2="100%" y2="35%"><stop stop-color="#87a9c1"/><stop offset=".4" stop-color="#a8c5d6"/><stop offset="1" stop-color="#6c8da8"/></linearGradient></defs>'''
def p(d,fill='none',stroke='none',w=1,op=1):return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'
def label(x,y,t,size=12):return f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Segoe UI" font-size="{size}" fill="#708194">{t}</text>'
def group(i,s):return f'<g transform="translate({[173,500,827][i]} 239) scale(.81)">{s}</g>'
def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1400" viewBox="0 0 1000 583.3333333333">'+DEFS+s+'</svg>'
paper=p('M0 0H1000V584H0Z','#f8f5ee')+label(500,35,'ANIME HAIR / STRUCTURE BEFORE TEXTURE',19)
paper+=label(500,61,'Skull and roots → connected masses → light and restrained strands',11)
for i,t in enumerate(['01 SKULL / HAIR FLOW','02 MASSES / THREE VALUES','03 SELECTIVE DETAIL']):paper+=label([173,500,827][i],509,t,12)
paper+=label(500,555,'Original practice geometry; references studied, no tutorial pixels traced or pasted',11)
base=[{'name':'01 Study paper','svg':svg(paper)}]
guides='<ellipse cx="0" cy="-33" rx="74" ry="99" fill="none" stroke="#b6acaa" stroke-width="1"/>'
guides+=p(FACE,stroke='#b6acaa',w=.8)+p(BACK,stroke='#8da5bb',w=1.1)+p(FRONT,stroke='#94aabd',w=.9)
guides+=p('M-7 -133Q-52 -118 -66 -62M-7 -133Q24 -112 35 -86M-7 -133Q73 -103 83 -31M-63 -45Q-90 62 -80 136M66 20Q99 137 112 209M-63 108Q-69 202 -93 250',stroke='#b58f91',w=.9)
for d in ['M-66 -62l-1 -10m1 10l6 -6','M35 -86l-7 -6m7 6l1 -9','M83 -31l-7 -5m7 5l2 -8','M-80 136l-5 -8m5 8l4 -9']:
    guides+=p(d,stroke='#b58f91',w=.9)
guides+='<circle cx="-7" cy="-133" r="3" fill="#ad7f87"/>'
guides+=p('M0 -131V82M-69 -13Q0 -4 70 -13',stroke='#d2c7c0',w=.65)
base.append({'name':'02 Skull envelope root point and flow arrows','svg':svg(group(0,guides)),'checkpoint':'01-flow-sketch.png'})

def skin():
    s=p('M-31 66L-34 112Q0 128 35 109L31 62Z','#efd8ce','#c0a5aa',.6)
    s+=p(FACE,'#f7e4d7','#bfa4a9',.65)
    s+=p('M-49 -3Q-31 -10 -12 -3M15 -3Q32 -11 52 -4',stroke='#b7a2aa',w=.8)
    s+=p('M-3 35h4M-6 58q8 -2 15 0',stroke='#c29ca5',w=.65)
    return s
def masses(i):
    return [
      {'name':f'03 Back mass {i}','svg':svg(group(i,p(BACK,'url(#back)','#7698b2',.9)))},
      {'name':f'04 Underlying face {i}','svg':svg(group(i,skin()))},
      {'name':f'05 Connected crown fringe and side masses {i}','svg':svg(group(i,p(FRONT,'url(#front)','#779bb7',.85)))},
      {'name':f'06 Shared form shadow {i}','svg':svg(group(i,p('M20 -140C79 -113 92 -67 80 -20Q88 30 104 65L90 68Q76 27 67 -20Q83 -70 49 -114Z','#6f91af',op=.34)+p('M-7 -136Q-18 -88 -59 -13Q-30 -36 -11 -42Q-4 -75 10 -108Z','#7296b2',op=.22))),
       'svg_masks':[svg(group(i,p(FRONT,'#fff')))]},
    ]
middle=masses(1);middle[-1]['checkpoint']='02-masses-and-values.png'
third=masses(2)
detail='';back_detail=''
for d in ['M-7 -136C-9 -100 -18 -68 -40 -40','M32 -88Q58 -41 60 24','M-64 4Q-76 64 -79 117','M77 55Q91 114 81 151']:
    detail+=p(d,stroke='#7195b2',w=.75,op=.7)
for d in ['M-104 86Q-106 155 -125 196','M95 102Q107 166 110 195','M-69 168Q-60 229 -89 250','M47 175Q58 229 29 253']:
    back_detail+=p(d,stroke='#7195b2',w=.7,op=.6)
detail+=p('M-68 -68Q-47 -105 -17 -112Q-3 -114 8 -110L5 -101Q-32 -100 -60 -64Z','#eaf3f3',op=.45)
detail+=p('M13 -110Q53 -100 64 -73L66 -61Q45 -90 8 -99Z','#e3eff0',op=.34)
for j in range(32):
    side=-1 if j<17 else 1;u=R.uniform(.12,.9)
    if side<0:
        x=-9-52*u;y=-120+75*u
        d=f'M{x:.2f} {y:.2f}Q{x-14:.2f} {y+23:.2f} {x-17:.2f} {y+R.uniform(36,61):.2f}'
    else:
        x=18+44*u;y=-115+65*u
        d=f'M{x:.2f} {y:.2f}Q{x+17:.2f} {y+20:.2f} {x+19:.2f} {y+R.uniform(34,57):.2f}'
    detail+=p(d,stroke=R.choice(['#eaf4f4','#90b4cc','#7298b7']),w=R.uniform(.22,.47),op=.45)
third.append({'name':'07 Medium divisions highlight grouping and sparse fine strands','svg':svg(group(2,detail)),
              'svg_masks':[svg(group(2,p(FRONT,'#fff')))],'checkpoint':'03-selective-detail.png'})
third.append({'name':'08 Back hair divisions stay behind the face','svg':svg(group(2,back_detail)),
              'svg_masks':[svg(group(2,p(BACK,'#fff')))],'svg_erase_masks':[svg(group(2,skin()))]})
one={'name':'Anime hair structure exercise','width':2400,'height':1400,'scale':2.4,'stem':'hair-flow-sketch','blank_checkpoint':'00-blank.kra','layers':base,'details':[]}
(OUT/'01-construction.json').write_text(json.dumps(one),encoding='utf-8')
two={**one,'stem':'hair-masses','layers':middle,'expected_document':str((OUT/'painting/hair-flow-sketch.kra').resolve())}
(OUT/'02-masses.json').write_text(json.dumps(two),encoding='utf-8')
three={**one,'stem':'anime-hair-structure','layers':third,'expected_document':str((OUT/'painting/hair-masses.kra').resolve())}
(OUT/'03-detail.json').write_text(json.dumps(three),encoding='utf-8')
(OUT/'elements.json').write_text(json.dumps({'purpose':'standalone structure practice, not a finished hairstyle for the portrait',
 'source_geometry':'original connected silhouette, not the old ribbon generator',
 'parting':[-7,-133],'skull_ellipse':[0,-33,74,99],
 'groups':['back mass','connected crown and bangs','left side','right side'],
 'checks':['readable with all fine lines hidden','no separate scalp ring','roots follow skull','continuous value groups','tip sizes vary']},indent=2),encoding='utf-8')
print('Prepared independent hair construction, masses, then selective-detail exercises')
