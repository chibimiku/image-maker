"""Third hair study: variable lock widths and controlled material edges."""
import ast
import copy
import json
import math
import random
import re
from pathlib import Path

OLD=Path('data/test-result/20261016/krita-hair-material-study')
OUT=Path('data/test-result/20261017/krita-hair-quality-pass')
OUT.mkdir(parents=True,exist_ok=True)
R=random.Random(171037)
ns={'math':math,'re':re}
for filename,names in [('author_seated_practice.py',{'hp','hd','p'}),('author_hair_material.py',{'path'})]:
    tree=ast.parse((Path('docs/krita-anime-painting/tools')/filename).read_text(encoding='utf-8'))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),filename,'exec'),ns)
path,hp=ns['path'],ns['hp']
oldbase=json.loads((OLD/'01-construction.json').read_text(encoding='utf-8'))
layers=copy.deepcopy(oldbase['layers'])
for filename in ['02-masses.json','03-rear-texture.json','04-highlights.json','05-review-soft-planes.json','06-review-sharp-edges.json']:
    for n in json.loads((OLD/filename).read_text(encoding='utf-8'))['layers']:
        found=next((x for x in layers if x['name']==n['name']),None)
        if found is None:layers.append(copy.deepcopy(n))
        else:found.update(copy.deepcopy(n))
by={n['name']:n for n in layers}
DEFS='''<defs>
<linearGradient id="base" x2="100%" y2="25%"><stop stop-color="#6f98b6"/><stop offset=".28" stop-color="#bed9e4"/><stop offset=".57" stop-color="#9abdd3"/><stop offset="1" stop-color="#557c9d"/></linearGradient>
<linearGradient id="lock" x2="100%" y2="25%"><stop stop-color="#638dac"/><stop offset=".22" stop-color="#a9ccdf"/><stop offset=".5" stop-color="#d5e8ec"/><stop offset=".77" stop-color="#a3c6da"/><stop offset="1" stop-color="#709aba"/></linearGradient>
<linearGradient id="shade" x2="100%" y2="0%"><stop stop-color="#456d91" stop-opacity="0"/><stop offset=".65" stop-color="#4c779b" stop-opacity=".36"/><stop offset="1" stop-color="#456c8b" stop-opacity=".1"/></linearGradient>
<radialGradient id="glow"><stop stop-color="#eff8ed" stop-opacity=".55"/><stop offset=".55" stop-color="#d9ecea" stop-opacity=".19"/><stop offset="1" stop-color="#d2e7eb" stop-opacity="0"/></radialGradient>
<linearGradient id="back" x2="100%" y2="20%"><stop stop-color="#6087a6"/><stop offset=".3" stop-color="#98bbd0"/><stop offset=".65" stop-color="#6189a8"/><stop offset="1" stop-color="#4c7396"/></linearGradient>
</defs>'''
def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2600" viewBox="350 150 290 314.1666666667">'+DEFS+s+'</svg>'
FRONT='M-100 -30C-116 -111 -64 -155 5 -151C87 -153 116 -103 103 -38Q105 -8 96 13L90 -14Q78 -20 73 -39C63 -65 49 -91 30 -112C27 -81 1 -51 -27 -35Q-25 -40 -23 -44C-34 -33 -52 -27 -65 -24Q-63 25 -69 58C-68 105 -85 143 -96 164C-99 112 -95 79 -97 48Q-111 74 -120 98C-109 56 -107 9 -100 -30Z'
tree=ast.parse(Path('docs/krita-anime-painting/tools/author_hair_material.py').read_text(encoding='utf-8'))
BACK=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BACK' for t in n.targets))
mask=svg(path(FRONT,'#fff'));ornament=by['18 Embroidered hem ribbon and small jewelry']['svg']
BASE={k:oldbase[k] for k in ['name','width','height','scale','details']}
BASE['name']='Azure hair III — grouped locks and controlled edges'
def write(file,stem,expected,ll):
    s={**BASE,'stem':stem,'expected_document':str(expected.resolve()),'layers':ll}
    (OUT/file).write_text(json.dumps(s,ensure_ascii=False),encoding='utf-8')
sketch=path(FRONT,stroke='#b07e92',w=.6)
for d in ['M25 -145C5 -105 -18 -65 -57 -30','M20 -144C-32 -140 -87 -113 -92 -40','M30 -145C78 -121 99 -73 94 -20','M-89 -22C-101 50 -79 106 -96 161']:
    sketch+=path(d,stroke='#b48b97',w=.55)
write('01-flow-review.json','01-flow-review',OLD/'painting/azure-hair-final.kra',[
    {'name':'30 Third study flow corrections','svg':svg(sketch)}])

front=path(FRONT,'url(#base)','#7196b0',.38)
# Sublocks overlap inside one continuous scalp envelope; their roots taper
# together instead of becoming a row of independent rounded caps.
LOCKS=[
 'M18 -146C-18 -143 -70 -131 -91 -83C-100 -58 -95 -18 -97 21C-85 -13 -86 -56 -72 -85C-48 -124 -11 -133 18 -146Z',
 'M24 -146C4 -128 -13 -105 -24 -86C-33 -64 -50 -41 -65 -24C-39 -38 -18 -61 -6 -84C4 -112 20 -132 24 -146Z',
 'M25 -142C26 -115 10 -80 -18 -57L-27 -35C4 -54 30 -81 30 -112Q30 -132 25 -142Z',
 'M32 -146C56 -135 84 -119 94 -86C105 -56 99 -27 96 13L90 -14C94 -49 88 -81 75 -103C62 -122 46 -134 32 -146Z',
 'M-89 -17C-98 18 -88 52 -86 82C-82 111 -88 140 -96 164C-74 127 -68 104 -69 74C-71 35 -80 8 -72 -24Z']
for d in LOCKS:front+=path(d,'url(#lock)',op=.8)
rear=path(BACK,'url(#back)','#648ba8',.35)
for d in ['M-99 -55C-124 38 -92 102 -114 170C-127 211 -111 247 -123 280C-102 254 -105 215 -99 184C-81 113 -103 42 -79 -63Z',
 'M98 -44C119 29 90 91 108 162Q122 209 107 259C131 223 123 192 120 165C105 89 130 19 105 -44Z']:
    rear+=path(d,'url(#shade)',op=.65)
for d in ['M-104 4C-115 71 -92 123 -108 178Q-116 221 -111 256C-102 208 -92 192 -94 153C-95 85 -100 50 -88 10Z',
 'M101 5C114 55 99 104 107 137Q118 179 110 213C118 172 121 146 118 115C112 68 126 37 111 12Z']:
    rear+=path(d,'url(#lock)',op=.35)
mass=[{'name':'06 Long rear hair grouped by flow','svg':svg(rear)},
      {'name':'17 Rooted bangs and shoulder locks with fine fibers','svg':svg(front),'svg_masks':[mask]},
      {'name':'30 Third study flow corrections','visible':False}]
for name in ['27 Grouped silk reflections','28 Selective fibers following connected masses','29 Selective sharp glints and silhouette flyaways','19b Fine fibers for 06 Long rear hair grouped by flow']:
    mass.append({'name':name,'visible':False})
planes=copy.deepcopy(by['23 Refined cheek planes and native soft color']);planes['svg_erase_masks'][0]=mask;mass.append(planes)
ear=copy.deepcopy(by['25 Ear lobe attachment and occluded earrings']);ear['svg_erase_masks']=[mask];mass.append(ear)
write('02-lock-masses.json','02-lock-masses',OUT/'painting/01-flow-review.kra',mass)

tone=''
for d in ['M16 -143C1 -102 -23 -61 -63 -27Q-29 -64 -15 -100Q-1 -130 16 -143Z',
 'M47 -139C83 -117 95 -73 91 -16Q105 -44 99 -83C89 -116 67 -133 47 -139Z',
 'M-88 -22C-91 20 -77 64 -83 111Q-76 80 -77 58Q-83 18 -76 -24Z']:
    tone+=path(d,'url(#shade)')
# Broad glows fade at all edges, with sharp short marks added much later.
for x,y,rx,ry in [(-47,-105,41,34),(-75,-55,16,35),(61,-102,17,30),(-83,58,10,36)]:
    xx,yy=hp(x,y)
    tone+=f'<ellipse cx="{xx}" cy="{yy}" rx="{rx}" ry="{ry}" fill="url(#glow)"/>'
details=''
def bez(q,t):return tuple((1-t)**3*q[0][j]+3*(1-t)**2*t*q[1][j]+3*(1-t)*t*t*q[2][j]+t**3*q[3][j] for j in (0,1))
flows=[((15,-143),(-27,-141),(-87,-112),(-92,-37),38,9),
       ((23,-142),(1,-116),(-16,-69),(-58,-29),32,5),
       ((35,-141),(80,-121),(99,-63),(94,-18),28,5),
       ((-85,-16),(-99,38),(-73,100),(-95,158),26,5)]
for a,b,c,d,count,spread in flows:
    for j in range(count):
        off=R.uniform(-spread,spread);q=[(z[0]+off,z[1]+off*.2) for z in (a,b,c,d)]
        t0=R.uniform(.03,.38);t1=R.uniform(.7,.99)
        pts=[bez(q,t0+(t1-t0)*k/10) for k in range(11)]
        dd='M'+'L'.join(f'{x:.3f} {y:.3f}' for x,y in pts)
        details+=path(dd,stroke=R.choice(['#dcefed','#527f9f','#b3d2e0']),w=R.uniform(.11,.21),op=R.uniform(.12,.28))
glints=''
for d in ['M-76 -96Q-71 -107 -63 -112Q-65 -105 -72 -98Z',
          'M-57 -117Q-50 -123 -44 -124Q-46 -121 -54 -117Z',
          'M-39 -126Q-33 -129 -29 -129Q-33 -126 -39 -126Z',
          'M-17 -105Q-19 -98 -24 -93Q-22 -101 -17 -105Z',
          'M-30 -82Q-33 -74 -39 -69Q-35 -77 -30 -82Z',
          'M65 -115Q72 -108 76 -98Q69 -105 65 -115Z']:
    glints+=path(d,'#e4f2e9',op=.53)
write('03-material.json','03-hair-material',OUT/'painting/02-lock-masses.kra',[
 {'name':'27 Grouped silk reflections','svg':svg(tone),'svg_masks':[mask],'svg_erase_masks':[ornament],'visible':True},
 {'name':'28 Selective fibers following connected masses','svg':svg(details),'svg_masks':[mask],'svg_erase_masks':[ornament],'visible':True},
 {'name':'29 Selective sharp glints and silhouette flyaways','svg':svg(glints),'strokes':[],'svg_masks':[mask],'svg_erase_masks':[ornament],'visible':True}])
(OUT/'elements.json').write_text(json.dumps({'scope':'hair-only quality practice; original facial expression and iris kept','roots':[25,-145],'camera':{'yaw':13,'pitch':-2,'roll':-5},'light':'large upper-left source','steps':['overlay flow sketch and review','connected silhouette with variable-width sublocks','form shade and fading reflected light','selective fibers and short sharp accents','occlusion and thumbnail checks'],'checks':['no cap seam','no rectangular root shade','no hair texture crossing ribbon','front silhouette masks updated for cheeks and earrings','readable without fine detail'],'remaining':['thick main fringe','rear hair needs more curvature','native brush polish after shape review']},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared third hair study without modifying old recipes')
