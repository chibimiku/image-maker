"""Hairstyle design study: explicit sections before shared color and light."""
import json
from pathlib import Path

OUT=Path('data/test-result/20261017/krita-hairstyle-sections')
OUT.mkdir(exist_ok=True)
DEFS='''<defs>
<linearGradient id="sharedHair" gradientUnits="userSpaceOnUse" x1="-140" y1="-100" x2="140" y2="240"><stop stop-color="#e5edf0"/><stop offset=".35" stop-color="#c0d3e4"/><stop offset=".7" stop-color="#96b3ce"/><stop offset="1" stop-color="#7395b9"/></linearGradient>
<linearGradient id="sharedShade" gradientUnits="userSpaceOnUse" x1="-100" y1="-110" x2="125" y2="230"><stop stop-color="#7c94b7" stop-opacity=".08"/><stop offset=".5" stop-color="#748fb1" stop-opacity=".24"/><stop offset="1" stop-color="#596e9d" stop-opacity=".5"/></linearGradient>
<radialGradient id="sharedLight" gradientUnits="userSpaceOnUse" cx="-36" cy="-91" r="157"><stop stop-color="#fff9e9" stop-opacity=".75"/><stop offset=".7" stop-color="#f2f4eb" stop-opacity=".2"/><stop offset="1" stop-color="#e1e8ee" stop-opacity="0"/></radialGradient>
</defs>'''
def p(d,fill='none',stroke='none',w=1,op=1):return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'
def label(x,y,t,size=14,color='#647a91'):return f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Segoe UI" font-size="{size}" fill="{color}">{t}</text>'
def group(i,s):return f'<g transform="translate({[250,750,1250][i]} 255) scale(.92)">{s}</g>'
def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="3000" height="1800" viewBox="0 0 1500 900">'+DEFS+s+'</svg>'

# All sections are original paths. Reference pixels are never loaded by this
# recipe. The tied region is intentionally schematic where structure is hidden.
PARTS=[
 ('A crown gathering','M-48 -132C-65 -173 -30 -191 7 -186C43 -187 66 -161 47 -132Q4 -117 -48 -132Z','#c4b3d0'),
 ('B left rear length','M-81 -83C-122 -14 -104 74 -123 147C-144 227 -189 253 -180 320C-176 363 -134 378 -147 426Q-160 460 -170 488C-132 466 -111 435 -116 400C-123 355 -163 343 -150 305C-139 260 -101 239 -83 194C-63 142 -77 65 -55 16L-35 -80Z','#a5bacd'),
 ('C right rear length','M77 -88C104 -37 96 26 110 90C128 160 181 202 171 264C164 312 122 336 144 381Q165 411 186 414C152 430 120 412 113 388C98 345 139 310 140 272C142 221 91 193 77 147C65 106 72 58 54 17L37 -88Z','#a5bacd'),
 ('D scalp envelope','M-86 -38C-105 -105 -56 -144 3 -143C74 -146 108 -95 91 -31L77 36L-78 39Z','#bccca9'),
 ('E left shoulder lock','M-69 -63C-80 -1 -60 66 -76 113C-89 153 -111 179 -107 213Q-93 183 -71 166C-42 133 -48 98 -49 71C-54 20 -38 -23 -34 -61Z','#a9c6c5'),
 ('F right shoulder lock','M69 -62C83 -4 61 46 76 105C86 151 115 162 124 200Q100 181 83 166C59 143 56 114 56 83C53 39 52 -7 43 -53Z','#a9c6c5'),
 ('G left face framing','M-49 -125C-86 -114 -96 -62 -83 -18C-77 5 -61 36 -43 53Q-57 22 -57 6Q-42 40 -24 43C-59 12 -63 -15 -59 -48C-56 -83 -25 -108 -10 -129Z','#c7b4b6'),
 ('H right swept fringe','M3 -137C49 -139 83 -111 87 -72C91 -44 79 -21 74 -7Q61 -38 39 -50C16 -69 8 -98 3 -137Z','#bdc0d6'),
 ('I central fringe','M-9 -137C-38 -120 -38 -89 -29 -58C-23 -35 -7 -20 10 -7Q-3 -24 -5 -39Q9 -20 21 -12C-1 -46 27 -91 10 -126Z','#c7b4b6'),
 ('J right gathered section','M82 -83C101 -69 95 -42 87 -30Q105 -15 88 1Q105 20 79 35L73 19Q87 8 75 -4Q91 -17 78 -31Q91 -48 78 -61Z','#d5c19e')]
FACE='M-71 -65C-77 -17 -59 48 -28 74Q-6 92 7 95C53 72 74 24 72 -56C65 -111 -53 -120 -71 -65Z'
def head():
    s=p('M-29 75L-32 132Q0 143 34 129L29 76Z','#f1d9ce','#bea6ab',.6)
    s+=p('M-32 126Q-79 139 -102 172L-90 216L92 216L104 172Q81 143 34 127Z','#e5e5e4','#aeb8c1',.7)
    s+=p(FACE,'#f9e7da','#b7a1ac',.65)
    for side in [-1,1]:
        cx=side*37
        s+=p(f'M{cx-23} 0Q{cx} -17 {cx+23} -1Q{cx} 20 {cx-23} 0Z','#fcf8f0')
        s+=f'<ellipse cx="{cx}" cy="0" rx="10" ry="12" fill="#8daac2"/>'
        s+=f'<ellipse cx="{cx}" cy="-1" rx="4" ry="7" fill="#526986"/>'
        s+=f'<ellipse cx="{cx-3}" cy="-5" rx="2.4" ry="3" fill="#eef7ee"/>'
        s+=p(f'M{cx-23} 0Q{cx} -17 {cx+23} -1',stroke='#77748a',w=1.3)
        s+=p(f'M{cx-20} -24Q{cx} -29 {cx+20} -24',stroke='#a8a2b0',w=.9)
    return s+p('M-2 35Q1 38 4 35M-8 62Q1 61 10 62',stroke='#bf98a5',w=.7)

paper=p('M0 0H1500V900H0Z','#f6f3ec')+label(750,36,'HAIRSTYLE DESIGN / SECTIONS → CONTOURS → SHARED LIGHT',23)
paper+=label(750,65,'Central fringe, face-framing locks, long rear waves and a gathered right side',14)
for i,t in enumerate(['01 SECTION MAP / CONSTRUCTION','02 ASSEMBLED LINE ART','03 ONE PALETTE / ONE LIGHT']):paper+=label([250,750,1250][i],820,t,15)
paper+=label(750,863,'Original study paths; hidden tie and bun construction are provisional, not anatomical conclusions',12)
base=[{'name':'01 Hairstyle study paper','svg':svg(paper)}]
draft=''
for name,d,color in PARTS:draft+=p(d,color,'#849aab',.75,.53)
draft+=p(FACE,stroke='#9f9aa0',w=.7)+p('M0 -140V95M-74 0Q0 8 74 0',stroke='#b3a6aa',w=.6)
draft+='<ellipse cx="0" cy="-31" rx="76" ry="101" fill="none" stroke="#9c9da7" stroke-width=".65" stroke-dasharray="4 4"/>'
for x,y,t in [(-12,-168,'A'),(-146,273,'B'),(150,265,'C'),(-37,-117,'D'),(-79,124,'E'),(89,124,'F'),(-75,-47,'G'),(52,-73,'H'),(-12,-71,'I'),(95,-28,'J')]:
    draft+=label(x,y,t,14,'#596f83')
base.append({'name':'02 Color coded hairstyle construction','svg':svg(group(0,draft)),'checkpoint':'01-section-map.png'})
one={'name':'Hairstyle sections — design before rendering','width':3000,'height':1800,'scale':2,'stem':'01-hairstyle-map','blank_checkpoint':'00-blank.kra','layers':base,'details':[]}
(OUT/'01-map.json').write_text(json.dumps(one),encoding='utf-8')

def section_layers(i,flat=False):
    ll=[]
    for n,(name,d,color) in enumerate(PARTS):
        if n==4:ll.append({'name':f'Face placeholder {i}','svg':svg(group(i,head()))})
        fill='#bdd0e2' if flat else '#f6f3ec'
        item={'name':f'{i} {name}','svg':svg(group(i,p(d,fill,'#7c90a8',.8 if n<4 else .65)))}
        if n in [4,5]:item['svg_erase_masks']=[svg(group(i,p(FACE,'#fff')))]
        ll.append(item)
    return ll
line=section_layers(1)
line[-1]['checkpoint']='02-all-section-contours.png'
two={**one,'stem':'02-hairstyle-contours','layers':line,'expected_document':str((OUT/'painting/01-hairstyle-map.kra').resolve())}
(OUT/'02-contours.json').write_text(json.dumps(two),encoding='utf-8')
flat=section_layers(2,True);flat[-1]['checkpoint']='03-unified-flat-color.png'
three={**one,'stem':'03-hairstyle-flat','layers':flat,'expected_document':str((OUT/'painting/02-hairstyle-contours.kra').resolve())}
(OUT/'03-flat.json').write_text(json.dumps(three),encoding='utf-8')

lit=[]
for name,d,color in PARTS:
    # Same absolute-coordinate color/light field for every part, rather than
    # restarting a separate highlight at the center of every closed section.
    body=p(d,'url(#sharedHair)','#748ba8',.6)+p(d,'url(#sharedLight)')
    lit.append({'name':f'2 {name}','svg':svg(group(2,body))})
    if name.startswith(('E ','F ')):lit[-1]['svg_erase_masks']=[svg(group(2,p(FACE,'#fff')))]
    shade=''
    if name.startswith('B '):shade=p('M-93 -32C-128 99 -77 155 -139 257C-159 291 -156 317 -135 343L-125 375C-174 344 -171 312 -151 271C-103 173 -143 118 -115 30Z','url(#sharedShade)')
    if name.startswith('C '):shade=p('M85 -37C115 81 92 135 150 210C179 251 145 294 130 322C161 300 184 264 170 226C150 172 113 153 110 101C104 49 112 9 101 -30Z','url(#sharedShade)')
    if name.startswith('G '):shade=p('M-18 -128C-66 -103 -77 -65 -65 -23Q-56 11 -24 43C-60 24 -87 -9 -84 -42C-85 -83 -56 -116 -18 -128Z','url(#sharedShade)')
    if name.startswith('H '):shade=p('M18 -133C20 -101 54 -65 74 -7Q76 -30 66 -51C37 -83 32 -115 18 -133Z','url(#sharedShade)')
    if name.startswith('I '):shade=p('M7 -130C20 -94 -2 -65 10 -37L21 -12C-6 -27 -14 -53 -8 -79Q1 -110 7 -130Z','url(#sharedShade)')
    if shade:lit[-1]['svg']=svg(group(2,body+shade));lit[-1]['svg_masks']=[svg(group(2,p(d,'#fff')))]
four={**one,'stem':'hairstyle-sections-lit','layers':lit,'expected_document':str((OUT/'painting/03-hairstyle-flat.kra').resolve())}
(OUT/'04-shared-light.json').write_text(json.dumps(four),encoding='utf-8')
fix={**one,'stem':'02-contours-reviewed','expected_document':str((OUT/'painting/02-hairstyle-contours.kra').resolve()),'layers':[n for n in line if n['name'].startswith(('1 E ','1 F '))]}
(OUT/'02b-root-occlusion.json').write_text(json.dumps(fix),encoding='utf-8')
three['expected_document']=str((OUT/'painting/02-contours-reviewed.kra').resolve())
(OUT/'03-flat.json').write_text(json.dumps(three),encoding='utf-8')
elements={'topic':'hairstyle design, independent from hair surface texture','source':'user reference viewed, geometry drawn anew','view':'near frontal, not yet tested in profile','style':{'fringe':'central tapered bangs, separate from face-framing pieces','rear':'very long, unequal S waves','sides':'short left cheek framing plus two shoulder locks','right':'gathered hair above ear; exact braid not inferred','crown':'raised gathered mass with provisional attachment'},'light':{'direction':'upper left','shared_coordinate_field':True},'sections':[{'id':name.split()[0],'name':name,'path':d,'depth_order':i} for i,(name,d,c) in enumerate(PARTS)],'order':['design landmarks','section outlines','assembled contour review','one flat hair color','shared light and occlusion','selective details only afterwards'],'checks':['hairstyle recognizable without texture','bangs do not unintentionally cover both eyes','rear and sides join a scalp volume','tie compresses nearby volume','all sections receive same light direction','profile and back continuity still need validation']}
(OUT/'hairstyle-template.json').write_text(json.dumps(elements,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared explicit hairstyle map, independent section contours, unified flats and shared lighting')
