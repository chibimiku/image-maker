"""Research construction snapshot: coordinates -> sketch -> facial features.

Consumes a semantic JSON template; emits an ordinary krita_studio scene and
resolved landmarks. No reference pixels are inserted into the artwork.
"""
import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--template',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
C=json.loads(args.template.read_text(encoding='utf-8'))
assert C['schema']=='krita-facial-elements/1'
args.output_dir.mkdir(parents=True,exist_ok=True)
H=C['head']; F=C['features']; P=C['palette']; L=C['lighting']; R=random.Random(316)
for v in C['composition']['views']:
    if abs(v['yaw_deg'])>C['direction_convention']['maximum_supported_abs_yaw']:
        raise ValueError('Template view outside supported shallow-yaw range')

DEFS='''<defs>
<linearGradient id="skin" x1="15%" y1="15%" x2="85%" y2="95%"><stop stop-color="%s"/><stop offset=".7" stop-color="%s"/><stop offset="1" stop-color="%s"/></linearGradient>
<linearGradient id="iris" x1="0%%" y1="0%%" x2="0%%" y2="100%%"><stop stop-color="%s"/><stop offset=".55" stop-color="%s"/><stop offset="1" stop-color="%s"/></linearGradient>
<linearGradient id="hair" x1="0%%" y1="0%%" x2="100%%" y2="40%%"><stop stop-color="%s"/><stop offset=".5" stop-color="%s"/><stop offset="1" stop-color="%s"/></linearGradient>
<radialGradient id="blush"><stop stop-color="#e7a9b5" stop-opacity=".35"/><stop offset="1" stop-color="#f4c8cd" stop-opacity="0"/></radialGradient>
</defs>'''.replace('15%','15%%').replace('85%','85%%').replace('95%','95%%')%(P['skin_light'],P['skin_base'],P['skin_shadow'],P['iris_top'],P['iris_mid'],P['iris_bottom'],P['hair_shadow'],P['hair_light'],P['hair_base'])

def svg(body):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 1800 1200">'%(C['canvas']['width'],C['canvas']['height'])+DEFS+'<g stroke-linecap="round" stroke-linejoin="round">'+body+'</g></svg>'
def path(d,fill='none',stroke='none',w=1,opacity=1):
    return '<path d="%s" fill="%s" stroke="%s" stroke-width="%.3f" opacity="%.3f"/>'%(d,fill,stroke,w,opacity)
def label(x,y,text,size=12,color=None):
    return '<text x="%.1f" y="%.1f" text-anchor="middle" font-family="Segoe UI,Arial" font-size="%.1f" fill="%s">%s</text>'%(x,y,size,color or P['label'],text)
def screen_el(x,y,rx,ry,color,stroke='none',w=1,opacity=1):
    return '<ellipse cx="%.2f" cy="%.2f" rx="%.2f" ry="%.2f" fill="%s" stroke="%s" stroke-width="%.3f" opacity="%.3f"/>'%(x,y,rx,ry,color,stroke,w,opacity)

class Projector:
    def __init__(self,view,scale=None,offset=(0,0)):
        self.view=view;self.scale=scale or C['composition']['portrait_scale'];self.offset=offset
        self.yaw=math.radians(view.get('yaw_deg',0));self.pitch=math.radians(view.get('pitch_deg',0));self.roll=math.radians(view.get('roll_deg',0))
    def rotate(self,x,y,z):
        xx=x*math.cos(self.yaw)+z*math.sin(self.yaw)
        zz=z*math.cos(self.yaw)-x*math.sin(self.yaw)
        yy=y*math.cos(self.pitch)+zz*math.sin(self.pitch)
        z2=zz*math.cos(self.pitch)-y*math.sin(self.pitch)
        sx=xx*math.cos(self.roll)-yy*math.sin(self.roll)
        sy=xx*math.sin(self.roll)+yy*math.cos(self.roll)
        return sx,sy,z2
    def world(self,x,y,z):
        sx,sy,zz=self.rotate(x,y,z)
        camera=C.get('camera',{'projection':'orthographic'})
        factor=1
        if camera['projection']=='perspective':
            denominator=camera['distance']-zz
            if denominator<=1:raise ValueError('Geometry intersects camera plane')
            factor=camera['focal_length']/denominator
        return (self.view['origin_x']+sx*self.scale*factor,self.view.get('origin_y',C['composition']['portrait_origin_y'])+sy*self.scale*factor)
    def point(self,x,y,extra_z=0):
        x+=self.offset[0];y+=self.offset[1]
        z=H['surface_depth']*math.sqrt(max(.02,1-(x/H['half_width'])**2))*(1-.17*(y/210)**2)+extra_z
        return self.world(x,y,z)
    def vp(self,direction):
        x,y,z=self.rotate(*direction)
        f=C['camera']['focal_length'];ox=self.view['origin_x'];oy=self.view.get('origin_y',C['composition']['portrait_origin_y'])
        if abs(z)<1e-10:return {'at_infinity':True,'direction':[x,y]}
        return {'at_infinity':False,'point':[ox-self.scale*f*x/z,oy-self.scale*f*y/z]}
    def d(self,commands,z=0):
        result=''
        for cmd in commands:
            k=cmd[0]
            if k=='Z':result+='Z';continue
            vals=[]
            for i in range(1,len(cmd),2):vals.extend(self.point(cmd[i],cmd[i+1],z))
            result+=k+' '.join('%.3f'%v for v in vals)
        return result
    def ellipse(self,x,y,rx,ry,fill,stroke='none',w=1,opacity=1,z=0):
        pts=[self.point(x+rx*math.cos(2*math.pi*i/64),y+ry*math.sin(2*math.pi*i/64),z) for i in range(65)]
        d='M%.3f %.3f'%pts[0]+''.join('L%.3f %.3f'%q for q in pts[1:])+'Z'
        return path(d,fill,stroke,w*self.scale,opacity)
    def draw(self,commands,fill='none',stroke='none',w=1,opacity=1,z=0):
        return path(self.d(commands,z),fill,stroke,w*self.scale,opacity)

def face_commands():
    w=H['half_width'];top=H['crown_y'];temple=H['temple_y'];cheek=H['cheek_y'];jaw=H['jaw_y'];chin=H['chin_y']
    return [['M',0,top],['C',-w*.62,top-3,-w*1.03,top+64,-w,temple],['C',-w*1.05,temple+55,-w*.99,cheek-2,-w*.86,cheek+38],['C',-w*.68,jaw+19,-w*.26,chin-12,0,chin],['C',w*.26,chin-12,w*.68,jaw+19,w*.86,cheek+38],['C',w*.99,cheek-2,w*1.05,temple+55,w,temple],['C',w*1.03,top+64,w*.62,top-3,0,top],['Z']]
def neck_commands():
    return [['M',-62,139],['L',-60,211],['Q',-114,220,-146,248],['L',147,248],['Q',114,221,61,210],['L',61,139],['Z']]
def neck_outline():
    return [['M',-61,155],['L',-60,211],['Q',-114,220,-146,248],['L',147,248],['Q',114,221,61,210],['L',61,155]]

def brows(pr,side):
    cx=side*F['eyes']['center_spacing']/2;cy=H['brow_y'];b=F['brows'];span=b['span'];h=b['arch_height']
    cmds=[['M',cx-span/2,cy+2],['Q',cx,cy-h,cx+span/2,cy+1]]
    body=pr.draw(cmds,stroke=P['brow'],w=b['line_width'])
    for j in range(b['separate_hairs']):
        x=cx+span*(.15+j*.08)
        body+=pr.draw([['M',x,cy-.5],['L',x+4,cy-.8]],stroke=P['brow'],w=.3,opacity=.35)
    return body

def eye(pr,side,seed):
    rr=random.Random(seed);e=F['eyes'];cx=side*e['center_spacing']/2;cy=H['eye_y'];w=e['width'];hh=e['opening_height']
    # Opening height and contour are explicit template values, not copied from
    # a generic circular eye. Iris and all details share this almond mask.
    top=[['M',cx-w/2,cy],['C',cx-w*.26,cy-hh*.62,cx+w*.20,cy-hh*.55,cx+w/2,cy-hh*.04]]
    almond=top+[['Q',cx+w*.25,cy+hh*.51,cx,cy+hh*.49],['Q',cx-w*.30,cy+hh*.48,cx-w/2,cy],['Z']]
    outer=pr.d(almond)
    whites=pr.draw(almond,P['sclera'],P['lid_soft'],.45)
    shade=top+[['L',cx+w/2,cy+4],['Q',cx,cy-hh*.25,cx-w/2,cy+4],['Z']]
    whites+=pr.draw(shade,'#cabcc9',opacity=.38)
    gaze=-math.sin(pr.yaw)*e['gaze_compensation_units']
    ix=cx+gaze;iy=cy+1;rad=e['iris_diameter']/2;ry=rad*e['iris_vertical_scale']
    iris=pr.ellipse(ix,iy,rad,ry,'url(#iris)',P['iris_edge'],.65)
    for j in range(e['iris_texture_count']):
        a=2*math.pi*(j+rr.uniform(-.25,.25))/e['iris_texture_count']
        t=rr.uniform(.32,.52);end=rr.uniform(.78,.98)
        pts=[['M',ix+rad*t*math.cos(a),iy+ry*t*math.sin(a)],['Q',ix+rad*(t+end)/2*math.cos(a+.015),iy+ry*(t+end)/2*math.sin(a+.015),ix+rad*end*math.cos(a),iy+ry*end*math.sin(a)]]
        iris+=pr.draw(pts,stroke=rr.choice(['#c8d4df','#b8c6d5','#93a4bf','#d0dbe0']),w=rr.uniform(.25,.5),opacity=e['iris_texture_opacity'])
    iris+=pr.ellipse(ix,iy-1,rad*e['pupil_width_ratio'],ry*e['pupil_height_ratio'],P['pupil'],opacity=.9)
    iris+=pr.ellipse(ix,iy+1,rad*.32,ry*.40,'none','#c1cbd8',.4,.45)
    iris+=pr.draw([['M',ix-rad*.63,iy+ry*.5],['Q',ix,iy+ry*.94,ix+rad*.64,iy+ry*.5]],stroke='#dee8e9',w=.9,opacity=.7)
    # Several soft facets instead of a black pupil / white circle contrast.
    for j in range(8):
        a=rr.uniform(.3,math.pi-.3);t=rr.uniform(.45,.8)
        iris+=pr.ellipse(ix+rad*t*math.cos(a),iy+ry*t*math.sin(a),rr.uniform(.5,1.2),rr.uniform(.6,1.4),'#dfe9ee',opacity=.68)
    hx,hy=L['highlight_screen_offset'];hr=e['highlight_radius']
    iris+=pr.ellipse(ix+hx,iy+hy,*hr,'#f3f8f8',opacity=.94)
    iris+=pr.ellipse(ix+rad*.45,iy+ry*.4,2,2.3,'#f2f8f7',opacity=.9)
    # Tapered lid band. Warm muted line color is part of reference matching.
    lash=top+[['C',cx+w*.20,cy-hh*.55-e['upper_lid_width'],cx-w*.26,cy-hh*.62-e['upper_lid_width'],cx-w/2,cy-1],['Z']]
    lines=pr.draw(lash,P['lid'])
    outer_x=cx+side*w/2
    for j in range(e['outer_lash_groups']):
        xx=outer_x-side*j*4.5;yy=cy-j*1.8
        lines+=pr.draw([['M',xx,yy],['Q',xx+side*6,yy-1,xx+side*(10-j),yy-4-j],['Q',xx+side*4,yy-1,xx-side*2,yy+1],['Z']],P['lid'])
    lines+=pr.draw([['M',cx-w*.38,cy+hh*.30],['Q',cx,cy+hh*.6,cx+w*.37,cy+hh*.3]],stroke=P['lid_soft'],w=e['lower_lid_width'],opacity=.75)
    lines+=pr.draw([['M',cx-w*.35,cy-hh*.85],['Q',cx,cy-hh*1.04,cx+w*.30,cy-hh*.78]],stroke=P['lid_soft'],w=.65,opacity=.7)
    lines+=pr.draw([['M',cx-w*.32,cy+hh*.42],['Q',cx,cy+hh*.64,cx+w*.3,cy+hh*.4]],stroke='#fff6f1',w=.9)
    for j in range(2):
        xx=outer_x-side*(8+j*6)
        lines+=pr.draw([['M',xx,cy+hh*.25+j],['L',xx+side*1.1,cy+hh*.25+2+j]],stroke=P['lid_soft'],w=.35,opacity=.6)
    return {'white':whites,'iris':iris,'lines':lines,'mask':pr.draw(almond,'#ffffff'),'iris_mask':pr.ellipse(ix,iy,rad,ry,'#ffffff')}

def nose(pr):
    n=F['nose'];y=H['nose_y'];z=n['protrusion'];rx,ry=n['highlight_radius']
    body=pr.ellipse(-2,y-10,rx*2.8,ry*2.4,'url(#blush)',opacity=.28,z=z)
    body+=pr.ellipse(-3,y-10,rx,ry,'#fffaf1',opacity=.88,z=z)
    # A short warm mark, never a continuous bridge contour.
    body+=pr.draw([['M',1,y],['Q',.5,y+2,1.4,y+n['lower_mark_length']]],stroke=P['nose_mark'],w=n['line_width'],opacity=.85,z=z)
    if abs(pr.yaw)>.15:
        sign=1 if pr.yaw>0 else -1
        body+=pr.draw([['M',sign*3,y-6],['Q',sign*5,y-2,sign*2,y+1]],stroke=P['nose_mark'],w=.5,opacity=.25,z=z)
    return body

def mouth(pr):
    m=F['mouth'];y=H['mouth_y'];w=m['width'];gap=m['opening_height'];z=m['protrusion']
    body=pr.draw([['M',-w/2,y],['Q',-w*.08,y-gap*.85,w/2,y+.2],['Q',w*.1,y+gap*.7,-w/2,y],['Z']],P['mouth_gap'],opacity=.85,z=z)
    body+=pr.draw([['M',-w/2,y],['Q',-w*.15,y-gap*.75,w*.19,y-.7],['Q',w*.35,y-.4,w/2,y+.2]],stroke=P['mouth_line'],w=m['line_width'],z=z)
    body+=pr.draw([['M',-w*.22,y+5],['Q',0,y+6.3,w*.23,y+5]],stroke=P['lip_light'],w=.7,opacity=m['lower_lip_opacity'],z=z)
    body+=pr.ellipse(-1,y+gap*.2,w*.14,.55,'#fbefe8',opacity=.9,z=z)
    return body

def ears(pr,side):
    e=F['ears'];x=side*e['center_x'];y=e['center_y'];w=e['width'];h=e['height']
    # Hide / reduce the far ear in oblique views. The head projection itself
    # also changes width, position and tilt; no mirrored copy is pasted.
    far=(pr.yaw>0 and side>0) or (pr.yaw<0 and side<0)
    opacity=.32 if far and abs(pr.yaw)>.15 else 1
    commands=[['M',x-side*7,y-h*.4],['C',x+side*w*.7,y-h*.8,x+side*w*.95,y+h*.25,x+side*4,y+h*.5],['Q',x-side*9,y+h*.28,x-side*7,y-h*.4],['Z']]
    body=pr.draw(commands,'url(#skin)',P['face_outline'],.75,opacity)
    body+=pr.draw([['M',x+side*2,y-h*.27],['Q',x+side*w*.55,y-h*.33,x+side*w*.32,y+h*.12],['Q',x+side*1,y-h*.07,x-side*2,y+h*.21]],stroke='#c39ba5',w=e['inner_line_width'],opacity=opacity*.75)
    return body

views=C['composition']['views'];projectors=[Projector(v) for v in views]
landmarks=[]
for v,pr in zip(views,projectors):
    pts={}
    for side in [-1,1]:
        x=side*F['eyes']['center_spacing']/2
        pts['eye_left' if side<0 else 'eye_right']={'center':pr.point(x,H['eye_y']),'corner_left':pr.point(x-F['eyes']['width']/2,H['eye_y']),'corner_right':pr.point(x+F['eyes']['width']/2,H['eye_y']),'brow_center':pr.point(x,H['brow_y'])}
    pts['nose']=pr.point(0,H['nose_y'],F['nose']['protrusion'])
    pts['mouth']=pr.point(0,H['mouth_y'],F['mouth']['protrusion'])
    pts['chin']=pr.point(0,H['chin_y'])
    def pixels(value):
        if isinstance(value,dict):return {k:pixels(v) for k,v in value.items()}
        return [round(a*C['canvas']['coordinate_scale'],4) for a in value]
    landmarks.append({'view':v,'points_scene_units':pts,'points_pixels':pixels(pts),'pixel_scale':C['canvas']['coordinate_scale']})

layers=[]
def add(name,body=None,checkpoint=None,**kw):
    item={'name':name,**kw}
    if body is not None:item['svg']=svg(body)
    if checkpoint:item['checkpoint']=checkpoint
    layers.append(item);return item

paper=path('M0 0H1800V1200H0Z',C['canvas']['paper'])
paper+=label(900,57,'FACIAL ELEMENTS / POSITION BEFORE DETAIL',25)
paper+=label(900,87,'coordinate guides > sketch > clean lines > flats > individual features > soft light',12)
for v in views:
    x=v['origin_x'];paper+=label(x,115,v['label'],12)
    paper+='<rect x="%.1f" y="153" width="510" height="535" rx="20" fill="#f4f6f7" stroke="#dce2e5"/>'%(x-255)
add('00 Paper and view labels',paper)

# Hidden foundations are prepared below the contour and revealed only in the
# flat-color stage; this keeps the lineart layer genuinely above the flats.
skin=''
for pr in projectors:skin+=pr.draw(neck_commands(),'url(#skin)')+pr.draw(face_commands(),'url(#skin)')
add('01 Skin flats prepared below lineart',skin,visible=False)

guides='';sketch='';lineart='';pencil_strokes=[];ink_strokes=[]
for idx,(v,pr,record) in enumerate(zip(views,projectors,landmarks)):
    guides+=pr.draw([['M',0,H['crown_y']],['Q',0,-10,0,H['chin_y']]],stroke=C['palette']['guide'],w=.6,opacity=.8)
    for name,yy in [('BROW',H['brow_y']),('EYE',H['eye_y']),('NOSE',H['nose_y']),('MOUTH',H['mouth_y'])]:
        guides+=pr.draw([['M',-138,yy],['Q',0,yy,138,yy]],stroke=P['guide'],w=.55,opacity=.65)
        x,y=pr.point(-145,yy);guides+=label(x-26,y+3,name,7)
    for name,coord in [('N',record['points_scene_units']['nose']),('M',record['points_scene_units']['mouth']),('C',record['points_scene_units']['chin'])]:
        x,y=coord;guides+=screen_el(x,y,2,2,'#7494ae')+label(x+31,y+4,'%s %.0f,%.0f'%(name,x,y),7)
    for key in ['eye_left','eye_right']:
        xy=record['points_scene_units'][key]['center'];guides+=screen_el(*xy,2,2,'#7398b5')
    sketch+=pr.draw(face_commands(),stroke='#8caac0',w=.8,opacity=.75)
    sketch+=pr.ellipse(0,-91,H['half_width']*.99,124,'none','#a9bdcc',.6,.7)
    sketch+=pr.draw(neck_commands(),stroke='#98afc2',w=.7,opacity=.65)
    for side in [-1,1]:
        xx=side*F['eyes']['center_spacing']/2;ww=F['eyes']['width'];hh=F['eyes']['opening_height']
        sketch+=pr.draw([['M',xx-ww/2,-hh*.55],['L',xx+ww/2,-hh*.55],['L',xx+ww/2,hh*.5],['L',xx-ww/2,hh*.5],['Z']],stroke='#92afc3',w=.6,opacity=.65)
        sketch+=pr.ellipse(xx,0,F['eyes']['iris_diameter']/2,F['eyes']['iris_diameter']/2,'none','#a5b8c6',.5,.7)
    sketch+=pr.draw([['M',-8,H['nose_y']-19],['L',8,H['nose_y']-19],['L',3,H['nose_y']+5],['Z']],stroke='#94acbc',w=.55,opacity=.65,z=F['nose']['protrusion'])
    sketch+=pr.draw([['M',-F['mouth']['width']/2,H['mouth_y']],['L',F['mouth']['width']/2,H['mouth_y']]],stroke='#94acbc',w=.7,opacity=.65,z=F['mouth']['protrusion'])
    lineart+=pr.draw(neck_outline(),stroke=P['face_outline'],w=.85)
    def native_commands(commands):
        out=[]
        for cmd in commands:
            coords=[]
            for j in range(1,len(cmd),2):coords.extend(pr.point(cmd[j],cmd[j+1]))
            out.append([cmd[0],*coords])
        return out
    pencil_strokes.append({'brush':'c) Pencil-2','size':.85*pr.scale,'color':'#87a4bb','opacity':.55,'path':native_commands(face_commands())})
    ink_strokes.append({'brush':'d) Ink-3 Gpen','size':.95*pr.scale,'color':P['face_outline'],'opacity':.85,'path':native_commands(face_commands())})
add('02 Coordinate guides',guides,'01-coordinate-guides.png')
diagnostic='';perspective_report=[]
if C.get('perspective_verification',{}).get('enabled'):
    for v,pr in zip(views,projectors):
        limits=C['perspective_verification']['box_bounds']
        corners={(i,j,k):(limits['x'][i],limits['y'][j],limits['z'][k]) for i in range(2) for j in range(2) for k in range(2)}
        points={key:pr.world(*point) for key,point in corners.items()}
        vps={axis:pr.vp(tuple(1 if i==axis else 0 for i in range(3))) for axis in range(3)}
        checks=[]
        for axis,col in [(0,'#b58082'),(1,'#7e9f8e'),(2,'#859eb8')]:
            for key in points:
                if key[axis]!=0:continue
                end=list(key);end[axis]=1;end=tuple(end)
                a,b=points[key],points[end]
                diagnostic+=path('M%.3f %.3fL%.3f %.3f'%(*a,*b),stroke=col,w=.75,opacity=.6)
                dx,dy=b[0]-a[0],b[1]-a[1];length=max(1e-9,math.hypot(dx,dy))
                # Extend the existing line away from the camera: do not draw
                # an invented segment from the corner to the desired VP.
                za=pr.rotate(*corners[key])[2];zb=pr.rotate(*corners[end])[2]
                near,far=(a,b) if za>zb else (b,a)
                ex,ey=far[0]+(far[0]-near[0])/length*40,far[1]+(far[1]-near[1])/length*40
                diagnostic+=path('M%.3f %.3fL%.3f %.3f'%(*far,ex,ey),stroke=col,w=.6,opacity=.38)
                vp=vps[axis]
                if vp['at_infinity']:
                    ux,uy=vp['direction'];err=abs(dx*uy-dy*ux)/(length*max(1e-9,math.hypot(ux,uy)))
                    checks.append({'axis':axis,'kind':'parallel_sine_error','error':err})
                else:
                    vx,vy=vp['point'];err=abs(dy*vx-dx*vy+dx*a[1]-dy*a[0])/length
                    checks.append({'axis':axis,'kind':'vp_line_distance_scene_units','error':err})
        # A true projected side-plane circle, used for orientation checks.
        near_side=-1 if v['yaw_deg']>0 else 1
        circle=[pr.world(near_side*137,-81+105*math.sin(t),22+65*math.cos(t)) for t in [i*2*math.pi/80 for i in range(81)]]
        diagnostic+=path('M%.3f %.3f'%circle[0]+''.join('L%.3f %.3f'%q for q in circle[1:]),stroke='#8a9fac',w=.6,opacity=.6)
        eyes=F['eyes'];centers=[-eyes['center_spacing']/2,eyes['center_spacing']/2]
        widths=[math.dist(pr.point(cx-eyes['width']/2,H['eye_y']),pr.point(cx+eyes['width']/2,H['eye_y'])) for cx in centers]
        perspective_report.append({'view':v['id'],'camera':C['camera'],'vanishing_points':vps,'box_edges':checks,'max_finite_vp_distance':max((q['error'] for q in checks if q['kind'].startswith('vp_')),default=0),'max_parallel_sine_error':max((q['error'] for q in checks if q['kind'].startswith('parallel')),default=0),'left_right_eye_widths':widths,'scope':'construction self-consistency only; not independent anatomical/artistic validation'})
    add('02b Perspective box and side-plane diagnostics',diagnostic,'01b-perspective-structure.png')
add('03 Pale construction sketch',sketch,'02-construction-sketch.png',strokes=pencil_strokes)
add('04 Clean contour lines',lineart,'03-clean-contours.png',strokes=ink_strokes,visibility={'03 Pale construction sketch':False})
add('05 Reveal flats and hide construction',checkpoint='04-skin-flats.png',visibility={'01 Skin flats prepared below lineart':True,'02 Coordinate guides':False,'02b Perspective box and side-plane diagnostics':False})

for feature in C['draw_order']:
    if feature in ['construction_guides','construction_sketch','clean_face_line','skin_flats','isolated_details']:continue
    body=''
    if feature=='brows':
        for pr in projectors:
            for side in [-1,1]:body+=brows(pr,side)
        add('06 Thin quiet brows',body,'05-brows.png')
    elif feature=='eyes':
        for index,pr in enumerate(projectors):
            for side in [-1,1]:
                part=eye(pr,side,300+index*20+side)
                add('07.%d.%d Eye whites'%(index,side),part['white'],svg_masks=[svg(part['mask'])])
                add('08.%d.%d Grey blue iris'%(index,side),part['iris'],svg_masks=[svg(part['mask']),svg(part['iris_mask'])])
                add('09.%d.%d Lid hierarchy'%(index,side),part['lines'])
        add('10 Eye stage complete',checkpoint='06-eyes.png')
    elif feature=='nose':
        for pr in projectors:body+=nose(pr)
        add('11 Minimal nose marks and highlights',body,'07-nose.png')
    elif feature=='mouth':
        for pr in projectors:body+=mouth(pr)
        add('12 Slightly parted mouths',body,'08-mouth.png')
    elif feature=='ears':
        for pr in projectors:
            for side in [-1,1]:body+=ears(pr,side)
        add('13 Ear direction and inner folds',body,'09-ears.png')
    elif feature=='hair_context':
        for pr in projectors:
            # Open forehead and unoccluded features for this anatomy exercise.
            commands=[['M',-143,-80],['C',-170,-190,-90,-240,0,-223],['C',100,-239,176,-174,148,-64],['L',140,-85],['C',135,-177,91,-203,0,-197],['C',-90,-209,-137,-162,-138,-90],['Z']]
            body+=pr.draw(commands,'url(#hair)',P['hair_shadow'],.7)
            for side in [-1,1]:
                commands=[['M',side*143,-133],['C',side*169,-62,side*144,38,side*169,107],['Q',side*133,90,side*131,38],['L',side*132,-91],['Z']]
                body+=pr.draw(commands,'url(#hair)',P['hair_shadow'],.65)
                for j in range(F['hair_context']['strand_count']//2):
                    x=side*(136+j*.75)
                    body+=pr.draw([['M',x,-110+j],['C',x+side*8,-20,x-side*5,35,x+side*13,88]],stroke='#a1bfd4',w=.35,opacity=.6)
        add('14 Hair context with features kept visible',body,'10-hair-context.png')
    elif feature=='atmosphere':
        for pr in projectors:
            body+=pr.ellipse(-94,50,38,22,'url(#blush)',opacity=.7)+pr.ellipse(96,48,35,20,'url(#blush)',opacity=.6)
            body+=pr.draw([['M',-61,152],['Q',0,187,61,153],['L',61,173],['Q',0,200,-61,174],['Z']],P['skin_shadow'],opacity=.28)
        add('15 Restrained warm skin and cool hair atmosphere',body,'11-soft-light.png')

# Individual feature cards reuse the same template values, not independent
# decorative drawings unrelated to the faces.
cards=''
for card in C['composition']['detail_cards']:
    x,y,w,h=[card[k] for k in ['x','y','width','height']]
    cards+='<rect x="%s" y="%s" width="%s" height="%s" rx="14" fill="#f8f1ec" stroke="#e0d9d3"/>'%(x,y,w,h)
    cards+=label(x+w/2,y+31,card['label'],11)
add('16 Isolated feature study cards',cards)
for card in C['composition']['detail_cards']:
    feature=card['feature'];x=card['x']+card['width']/2;y=card['y']+card['height']*.53
    offset={'brows':(F['eyes']['center_spacing']/2,-H['brow_y']),'eyes':(F['eyes']['center_spacing']/2,0),'nose':(0,-H['nose_y']),'mouth':(0,-H['mouth_y']),'ears':(-F['ears']['center_x'],-F['ears']['center_y'])}[feature]
    # Offset translates projected feature coordinates; geometry and depth must
    # still be evaluated in head coordinates for non-frontal views.
    view={'origin_x':x,'origin_y':y,'yaw_deg':0,'pitch_deg':0,'roll_deg':0}
    pr=Projector(view,card['detail_scale'],offset)
    if feature=='eyes':
        part=eye(pr,-1,301)
        add('17 Eye detail white',part['white'],svg_masks=[svg(part['mask'])])
        add('18 Eye detail iris',part['iris'],svg_masks=[svg(part['mask']),svg(part['iris_mask'])])
        add('19 Eye detail lids',part['lines'])
    else:
        body=brows(pr,-1) if feature=='brows' else nose(pr) if feature=='nose' else mouth(pr) if feature=='mouth' else ears(pr,1)
        add('20 '+feature+' isolated detail',body)
footer=label(900,1141,'Same element template / three head directions / gaze compensated toward viewer / no reference pixels pasted',12)
footer+=label(900,1167,'Warm muted lines · smaller irises and pupils · short nose mark · slightly parted mouth',11)
add('21 Study footer',footer,'12-complete-study.png')

scene={'name':C['title'],'width':C['canvas']['width'],'height':C['canvas']['height'],'scale':C['canvas']['coordinate_scale'],'stem':C['deliverables']['stem'],'blank_checkpoint':C['deliverables']['blank_checkpoint'],'layers':layers,
       'details':[{'file':'front-face.png','rect':[115,177,411,455]},{'file':'turned-right-face.png','rect':[695,177,411,455]},{'file':'feature-cards.png','rect':[20,780,1760,320]}]}
(args.output_dir/'facial-study.scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
(args.output_dir/'resolved-landmarks.json').write_text(json.dumps(landmarks,ensure_ascii=False,indent=2),encoding='utf-8')
(args.output_dir/'perspective-checks.json').write_text(json.dumps(perspective_report,ensure_ascii=False,indent=2),encoding='utf-8')
reference=Path(C['reference']['path'])
record={'template':str(args.template),'template_sha256':hashlib.sha256(args.template.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'reference_pixels_inserted':False,'draw_order':C['draw_order'],'landmarks_generated_before_render':True,'views':C['composition']['views']}
(args.output_dir/'preparation-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'scene':str(args.output_dir/'facial-study.scene.json'),'layers':len(layers),'template_sha256':record['template_sha256']}))
