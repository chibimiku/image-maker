"""Medium/small clumps tested on the complete hairstyle, not isolated strips."""
import ast
import copy
import json
import math
import random
from pathlib import Path

OUT=Path('data/test-result/20261017/krita-contextual-hair')
OUT.mkdir(exist_ok=True)
R=random.Random(171123)
tree=ast.parse(Path('docs/krita-anime-painting/tools/author_hairstyle_sections.py').read_text(encoding='utf-8'))
ns={}
for n in tree.body:
    if isinstance(n,ast.Assign):
        for t in n.targets:
            if isinstance(t,ast.Name) and t.id in {'PARTS','FACE'}:ns[t.id]=ast.literal_eval(n.value)
    if isinstance(n,ast.FunctionDef) and n.name in {'p','head'}:exec(compile(ast.Module(body=[n],type_ignores=[]),'hairstyle-helpers','exec'),ns)
p,head,FACE=ns['p'],ns['head'],ns['FACE']
DEFS='''<defs>
<linearGradient id="hair" gradientUnits="userSpaceOnUse" x1="-125" y1="-150" x2="180" y2="450"><stop stop-color="#d9e2eb"/><stop offset=".34" stop-color="#c4d5e7"/><stop offset=".7" stop-color="#9db9d4"/><stop offset="1" stop-color="#b2c8df"/></linearGradient>
<linearGradient id="shade" x2="100%" y2="30%"><stop stop-color="#8997b9" stop-opacity=".08"/><stop offset=".65" stop-color="#7185a9" stop-opacity=".44"/><stop offset="1" stop-color="#6f7ea4" stop-opacity=".16"/></linearGradient>
<radialGradient id="glow"><stop stop-color="#fff9ed" stop-opacity=".72"/><stop offset=".52" stop-color="#e9f0f0" stop-opacity=".24"/><stop offset="1" stop-color="#d3dfeb" stop-opacity="0"/></radialGradient>
</defs>'''
def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="3000" viewBox="-240 -230 480 720">'+DEFS+s+'</svg>'

# Unequal lengths and interleaving openings replace the two unbroken rear
# strips. Each medium clump has its own silhouette, not merely extra lines.
REAR=[
 ('B1 inner rear left','M-70 -92C-104 -16 -76 78 -107 157C-133 218 -148 246 -138 291C-123 333 -123 367 -149 405Q-130 356 -156 320C-178 277 -160 228 -137 181C-109 116 -128 46 -104 -29Z',
  'M-105 23C-116 102 -103 132 -137 202C-167 262 -160 290 -143 318C-165 304 -176 269 -157 224C-126 151 -132 115 -121 51Z',
  [[[-91,-29],[-109,73],[-84,122],[-125,204]],[[-126,205],[-160,263],[-146,287],[-139,318]]]),
 ('C1 inner rear right','M62 -93C111 -24 81 47 110 117C133 172 169 197 163 251C156 294 126 328 150 370Q128 357 124 335C117 295 150 270 144 234C135 193 97 169 86 123C73 70 84 14 51 -29Z',
  'M92 18C89 93 112 152 151 194C180 237 145 287 138 307C155 258 146 227 126 204C85 159 81 102 81 56Z',
  [[[84,-32],[107,67],[87,116],[132,173]],[[135,181],[174,240],[125,270],[135,327]]]),
 ('B2 outer long left','M-96 -42C-122 23 -106 102 -140 170C-162 219 -199 250 -193 299C-189 335 -148 349 -149 384C-150 426 -185 457 -183 483Q-176 455 -162 444C-124 410 -122 376 -147 349C-167 328 -175 314 -168 286C-155 245 -118 216 -105 172C-87 107 -93 45 -75 4Z',
  'M-136 186C-163 238 -193 266 -182 308C-173 333 -150 350 -146 369C-151 342 -197 334 -194 288C-191 248 -154 215 -136 186Z',
  [[[-104,27],[-113,96],[-104,136],[-135,196]],[[-142,211],[-192,279],[-178,304],[-165,323]],[[-154,349],[-114,397],[-170,442],[-180,474]]]),
 ('C2 outer long right','M89 -43C116 9 103 70 133 128C165 188 199 209 196 252C191 294 149 317 166 353C177 376 205 381 216 372Q194 396 168 382C139 366 135 340 151 311C176 272 184 256 172 231C157 200 123 179 110 144C93 99 101 48 75 2Z',
  'M143 165C172 201 202 229 188 268C180 287 160 309 155 325C163 290 183 281 178 247C175 220 157 195 143 165Z',
  [[[104,20],[116,98],[112,124],[141,164]],[[147,177],[196,232],[182,254],[166,284]],[[159,310],[130,356],[189,396],[211,375]]]),
 ('B3 inner tapered left','M-62 -1C-76 70 -61 121 -86 177C-108 222 -117 252 -104 281C-94 300 -76 306 -79 332Q-84 349 -96 355C-82 332 -110 321 -122 298C-144 258 -117 218 -104 183C-81 126 -91 76 -81 28Z',
  'M-89 154C-98 214 -133 247 -115 289Q-109 308 -92 320C-112 313 -135 286 -133 264C-132 222 -106 200 -99 167Z',
  [[[-78,34],[-87,102],[-73,132],[-96,193]],[[-103,209],[-143,273],[-109,305],[-91,321]]]),
 ('C3 inner tapered right','M64 -6C79 48 65 92 91 148C106 180 139 206 132 239C127 265 109 284 116 310Q117 323 131 327C106 326 98 309 103 289C109 260 122 244 111 219C93 190 76 180 69 151C57 107 66 69 48 23Z',
  'M78 132C80 178 108 201 121 221Q135 250 113 279C146 257 140 228 125 208C109 189 93 165 87 145Z',
  [[[63,37],[79,100],[62,119],[89,164]],[[95,179],[139,225],[116,253],[111,285]]])]
allparts=[]
for i,(name,d,c) in enumerate(ns['PARTS']):
    if name.startswith(('B ','C ')):continue
    allparts.append({'id':name,'d':d,'shade':'','flows':[]})
front=allparts[2:];crown=allparts[:2] # A and D; original B/C were removed.
crown=[n for n in allparts if n['id'].startswith('A ')]
scalp=[n for n in allparts if n['id'].startswith('D ')]
front=[n for n in allparts if n['id'][0] in 'EFGHIJ']
FLOWS={
 'A':[[[-37,-148],[-44,-188],[12,-190],[39,-153]],[[-20,-146],[-20,-181],[21,-184],[34,-149]]],
 'D':[[[0,-141],[-52,-136],[-84,-103],[-87,-43]],[[5,-141],[66,-137],[100,-85],[85,-41]]],
 'E':[[[-62,-30],[-72,40],[-47,84],[-76,135]],[[-72,130],[-89,155],[-110,185],[-105,208]]],
 'F':[[[64,-32],[82,38],[57,81],[79,128]],[[79,127],[90,158],[116,173],[122,195]]],
 'G':[[[-25,-129],[-64,-115],[-84,-77],[-77,-29]],[[-78,-32],[-73,8],[-59,33],[-43,51]]],
 'H':[[[7,-135],[56,-129],[89,-87],[77,-39]],[[19,-119],[20,-78],[61,-50],[73,-10]]],
 'I':[[[-9,-134],[-40,-108],[-26,-48],[9,-10]],[[4,-130],[22,-101],[-10,-65],[17,-19]]],
 'J':[[[87,-78],[98,-63],[80,-43],[91,-32]],[[88,-28],[103,-12],[77,-6],[89,6]]]
}
for n in crown+scalp+front:n['flows']=FLOWS[n['id'][0]]
rear=[{'id':a,'d':b,'shade':c,'flows':d} for a,b,c,d in REAR]
PARTS=crown+rear+scalp+front

def light_body(n):
    s=p(n['d'],'url(#hair)','#8897b1',.46)
    if n['shade']:s+=p(n['shade'],'url(#shade)')
    # Unequal local planes; the highlights are not a repeated cylindrical field.
    for k,q in enumerate(n['flows']):
        pts=[]
        for j in range(41):
            t=j/40;x,y=bez(q,t);nx,ny=normal(q,t);ww=(2.5 if n['id'][0] in 'BC' else 1.5)*math.sin(math.pi*t)**.7
            pts.append((x+nx*ww,y+ny*ww))
        for j in range(40,-1,-1):
            t=j/40;x,y=bez(q,t);nx,ny=normal(q,t);ww=.25*math.sin(math.pi*t)
            pts.append((x-nx*ww,y-ny*ww))
        s+=p(poly(pts),'#7385a9',op=.18 if k%2 else .24)
    return s
def bez(q,t):return tuple((1-t)**3*q[0][j]+3*(1-t)**2*t*q[1][j]+3*(1-t)*t*t*q[2][j]+t**3*q[3][j] for j in (0,1))
def normal(q,t):
    a=bez(q,max(0,t-.002));b=bez(q,min(1,t+.002));dx,dy=b[0]-a[0],b[1]-a[1];z=math.hypot(dx,dy);return dy/z,-dx/z
def poly(pts):return 'M'+'L'.join(f'{x:.3f} {y:.3f}' for x,y in pts)+'Z'
def fine(n):
    s=''
    for qi,q in enumerate(n['flows']):
        for j in range(22 if n['id'][0] in 'BC' else 15):
            off=R.uniform(-5,5);a=R.uniform(.02,.35);b=R.uniform(.62,.98)
            left=[];right=[];half=R.uniform(.07,.16)
            for k in range(33):
                t=a+(b-a)*k/32;x,y=bez(q,t);nx,ny=normal(q,t);oo=off*(.4+.6*math.sin(math.pi*t))
                w=half*math.sin(math.pi*k/32)**.65;left.append((x+nx*(oo+w),y+ny*(oo+w)));right.append((x+nx*(oo-w),y+ny*(oo-w)))
            s+=p(poly(left+right[::-1]),R.choice(['#edf1f2','#7187a9','#a0b9d1']),op=R.uniform(.2,.42))
        # Short thin reflections near turns, not solid full-length bright lines.
        for j in range(7):
            a=R.uniform(.18,.62);b=a+R.uniform(.02,.075);off=R.uniform(-4,2)
            pts=[]
            for k in range(12):
                t=a+(b-a)*k/11;x,y=bez(q,t);nx,ny=normal(q,t);pts.append((x+nx*off,y+ny*off))
            s+=p('M'+'L'.join(f'{x:.2f} {y:.2f}' for x,y in pts),stroke='#f4f5eb',w=R.uniform(.2,.48),op=.45)
    return s
def node(n,phase):
    body=p(n['d'],'#f5f2ed','#8994a9',.58) if phase=='outline' else light_body(n)
    if phase=='fine':body+=fine(n)
    item={'name':n['id'],'svg':svg(body),'svg_masks':[svg(p(n['d'],'#fff'))]}
    if n['id'][0] in 'EF':item['svg_erase_masks']=[svg(p(FACE,'#fff'))]
    return item

layers=[{'name':'00 Study paper','svg':svg(p('M-240 -230H240V490H-240Z','#f5f2ed'))}]
for n in PARTS:
    if n['id'].startswith('E '):layers.append({'name':'Face placement for hairstyle evaluation','svg':svg(head())})
    layers.append(node(n,'outline'))
flow=''
for n in PARTS:
    for q in n['flows']:
        flow+=p('M'+' '.join(map(str,q[0]))+'C'+' '.join(str(v) for pt in q[1:] for v in pt),stroke='#bf8e9c',w=.5,op=.6)
layers.append({'name':'32 Medium clump direction sketch','svg':svg(flow)})
base={'name':'Hairstyle in context — uneven medium and small clumps','width':2000,'height':3000,'scale':2000/480,'stem':'01-context-outline','blank_checkpoint':'00-blank.kra','layers':layers,'details':[]}
(OUT/'01-outline.json').write_text(json.dumps(base),encoding='utf-8')
two={**base,'stem':'02-context-values','layers':[node(n,'value') for n in PARTS]+[{'name':'32 Medium clump direction sketch','visible':False}],'expected_document':str((OUT/'painting/01-context-outline.kra').resolve())}
(OUT/'02-values.json').write_text(json.dumps(two),encoding='utf-8')
three={**base,'stem':'03-context-fine','layers':[node(n,'fine') for n in PARTS],'expected_document':str((OUT/'painting/02-context-values.kra').resolve())}
(OUT/'03-fine.json').write_text(json.dumps(three),encoding='utf-8')
(OUT/'elements.json').write_text(json.dumps({'scope':'complete hairstyle study with simplified face placeholder; original gem-eye portrait not altered','camera':'near frontal','light':'upper left; common base palette plus individually designed planes','parts':PARTS,'sequence':['outline and uneven medium clumps','review silhouette and openings','shared base and local shadows','partial fine strands','review in head context'],'checks':['no two single rear strips','distinct unequal ends and openings','fringe central, right gathered hair retained','contacts and shade belong to visible overlap','fine strands subordinate to medium shapes'],'limitations':['simplified face; not final character painting','bun and gathering structure still provisional','not validated from side or back']},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared full hairstyle study with six unequal rear clumps and context-specific detail')
