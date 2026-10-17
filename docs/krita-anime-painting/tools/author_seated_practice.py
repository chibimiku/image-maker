"""New seated illustration recipe: original geometry, staged Krita painting.

Generic strand construction is adapted from our earlier technique experiments;
no previous artwork or reference pixels are loaded.
"""
import json
import math
import random
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

OUT=Path('data/test-result/20261016/krita-seated-practice-v4')
OUT.mkdir(parents=True,exist_ok=True)
R=random.Random(416)
LAYERS=[]
STATS={'strand_paths':0,'locks':0}
PLAN={'title':'Azure IV — A quiet page','canvas':[2400,3360],
 'coordinate_space':[1000,1400],'new_geometry':True,'previous_artwork_loaded':False,
 'pose':{'action':'seated, knees angled toward viewer right, hands holding a closed book',
         'shoulders':[[398,442],[575,455]],'waist':[[427,616],[550,633]],
         'hips':[520,706],'knees':[[635,850],[726,864]],'ankles':[[596,1154],[733,1149]]},
 'head':{'origin':[481,308],'yaw_degrees':13,'pitch_degrees':-2,'roll_degrees':-5,
         'camera_distance':1300,'focal_length':1250,'eyes_local':[[-46,0],[46,0]],
         'brow_y':-25,'nose_y':39,'mouth_y':65,'chin_y':100},
 'chair':{'origin':[546,796],'yaw_degrees':-17,'pitch_degrees':18,
          'distance':1700,'focal_length':1500,'seat_half_width':210,'seat_half_depth':145},
 'palette':{'hair':'muted powder blue','eyes':'restrained grey blue','skin':'warm ivory',
            'dress':'ice blue and ivory','chair':'dark slate wood'},
 'light':'large soft window at upper left, warm skin and cool reflected shadows',
 'sequence':['coordinates and chair perspective','pencil construction','clean line art',
             'flat colors','form shadows and materials','individual facial features',
             'hair fibers and fabric decoration','soft light','visual review'],
 'limits':['shallow head surface, not full anatomical model','hands and cloth remain stylized',
           'projected chair consistency is not artistic approval']}
(OUT/'elements.json').write_text(json.dumps(PLAN,ensure_ascii=False,indent=2),encoding='utf-8')

DEFS='''<defs>
<linearGradient id="room" x2="85%" y2="100%"><stop stop-color="#faf6ee"/><stop offset="1" stop-color="#d1e0e8"/></linearGradient>
<linearGradient id="glass" x2="100%" y2="100%"><stop stop-color="#d0e5eb"/><stop offset=".5" stop-color="#edf4ef"/><stop offset="1" stop-color="#b8cfdc"/></linearGradient>
<linearGradient id="hair" x2="100%" y2="25%"><stop stop-color="#7899ba"/><stop offset=".24" stop-color="#bfdbe9"/><stop offset=".47" stop-color="#e5f2f7"/><stop offset=".7" stop-color="#b3d0e2"/><stop offset="1" stop-color="#6b90b2"/></linearGradient>
<linearGradient id="hairDark" x2="100%" y2="15%"><stop stop-color="#62829f"/><stop offset=".5" stop-color="#adc9dc"/><stop offset="1" stop-color="#6487a5"/></linearGradient>
<linearGradient id="skin" x2="80%" y2="100%"><stop stop-color="#fff5e7"/><stop offset=".65" stop-color="#f7dfd5"/><stop offset="1" stop-color="#dab8b8"/></linearGradient>
<linearGradient id="face" x2="80%" y2="100%"><stop stop-color="#fff7ec"/><stop offset=".7" stop-color="#fbe8dc"/><stop offset="1" stop-color="#edcbca"/></linearGradient>
<linearGradient id="dress" x2="100%" y2="35%"><stop stop-color="#7695b5"/><stop offset=".25" stop-color="#b6cedf"/><stop offset=".48" stop-color="#e9f2f7"/><stop offset=".7" stop-color="#c2d7e6"/><stop offset="1" stop-color="#829dbd"/></linearGradient>
<linearGradient id="ivory" x2="70%" y2="100%"><stop stop-color="#fffaf0"/><stop offset=".55" stop-color="#f2efe9"/><stop offset="1" stop-color="#bbc7d6"/></linearGradient>
<linearGradient id="wood" x2="100%" y2="20%"><stop stop-color="#2f4254"/><stop offset=".4" stop-color="#607488"/><stop offset="1" stop-color="#253c50"/></linearGradient>
<linearGradient id="velvet" x2="100%" y2="100%"><stop stop-color="#a2b9c9"/><stop offset="1" stop-color="#5f7e98"/></linearGradient>
<linearGradient id="iris" x2="0%" y2="100%"><stop stop-color="#5b6c83"/><stop offset=".5" stop-color="#91a8bb"/><stop offset="1" stop-color="#c9dce2"/></linearGradient>
<radialGradient id="blush"><stop stop-color="#e7a4af" stop-opacity=".42"/><stop offset="1" stop-color="#f1c5ca" stop-opacity="0"/></radialGradient>
<radialGradient id="shade"><stop stop-color="#54657f" stop-opacity=".25"/><stop offset="1" stop-color="#77899e" stop-opacity="0"/></radialGradient>
<radialGradient id="light"><stop stop-color="#fffce8" stop-opacity=".6"/><stop offset="1" stop-color="#fff8e9" stop-opacity="0"/></radialGradient>
</defs>'''
def p(d,fill='none',stroke='none',w=1,opacity=1):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{opacity}"/>'
def e(x,y,rx,ry,fill,opacity=1,stroke='none',w=1):
    return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{opacity}" stroke="{stroke}" stroke-width="{w}"/>'
def svg(s):return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="3360" viewBox="0 0 1000 1400">'+DEFS+'<g stroke-linecap="round" stroke-linejoin="round">'+s+'</g></svg>'
def add(name,s='',**kw):
    LAYERS.append({'name':name,'svg':svg(s),**kw})
def poly(points,fill,stroke='none',w=1):
    return p('M'+'L'.join(f'{x:.3f} {y:.3f}' for x,y in points)+'Z',fill,stroke,w)
def cp(x,y,z):
    a=math.radians(-17);b=math.radians(18)
    xx=x*math.cos(a)+z*math.sin(a);zz=z*math.cos(a)-x*math.sin(a)
    yy=y*math.cos(b)+zz*math.sin(b);zz=zz*math.cos(b)-y*math.sin(b)
    f=1500/(1700-zz)
    return (546+xx*f,796+yy*f)
def hp(x,y,z=None):
    if z is None:z=42-22*(x/90)**2-12*(y/115)**2
    a=math.radians(13);b=math.radians(-2);c=math.radians(-5)
    xx=x*math.cos(a)+z*math.sin(a);zz=z*math.cos(a)-x*math.sin(a)
    yy=y*math.cos(b)+zz*math.sin(b);zz=zz*math.cos(b)-y*math.sin(b)
    xx,yy=xx*math.cos(c)-yy*math.sin(c),xx*math.sin(c)+yy*math.cos(c)
    f=1250/(1300-zz)
    return (481+xx*f,308+yy*f)
def hd(cmds,fill='none',stroke='none',w=1,opacity=1):
    d=''
    for cmd in cmds:
        d+=cmd[0]
        for j in range(1,len(cmd),2):
            x,y=hp(cmd[j],cmd[j+1]);d+=f'{x:.3f} {y:.3f} '
    return p(d,fill,stroke,w,opacity)
def smooth(pts):
    d='M%.3f %.3f'%pts[0]
    for i in range(len(pts)-1):
        a,b,c,dd=pts[max(0,i-1)],pts[i],pts[i+1],pts[min(len(pts)-1,i+2)]
        d+='C%.3f %.3f %.3f %.3f %.3f %.3f'%(b[0]+(c[0]-a[0])/6,b[1]+(c[1]-a[1])/6,c[0]-(dd[0]-b[0])/6,c[1]-(dd[1]-b[1])/6,*c)
    return d
def bez(q,t):
    return tuple((1-t)**3*q[0][i]+3*(1-t)**2*t*q[1][i]+3*(1-t)*t*t*q[2][i]+t**3*q[3][i] for i in [0,1])
def strand(q,t,off,width):
    x,y=bez(q,t);a=bez(q,max(0,t-.001));b=bez(q,min(1,t+.001));dx,dy=b[0]-a[0],b[1]-a[1];n=max(.001,math.hypot(dx,dy))
    profile=(.08+.92*math.sin(math.pi*min(1,t/.22)/2))*(1-t)**.5
    return (x-dy/n*off*width*profile,y+dx/n*off*width*profile)
def lock(q,width,n=65,dark=False):
    left=[strand(q,i/35,-1,width) for i in range(36)];right=[strand(q,i/35,1,width) for i in range(35,-1,-1)]
    s=p(smooth(left)+'L%.3f %.3f'%right[0]+smooth(right)[smooth(right).find('C'):]+'Z','url(#hairDark)' if dark else 'url(#hair)','#6d8ca7',.65)
    for j in range(n):
        off=-.98+1.96*(j+R.uniform(-.2,.2))/(n-1);t0=R.uniform(.01,.09);t1=R.uniform(.85,.998)
        pts=[strand(q,t0+(t1-t0)*i/27,off,width) for i in range(28)]
        col=R.choice(['#e5f4fa','#f4fbfc','#c5dfed']) if j%4==0 else R.choice(['#7c9fb9','#94b6cd','#a9c7da'])
        s+=p(smooth(pts),stroke=col,w=R.uniform(.19,.49),opacity=R.uniform(.35,.75));STATS['strand_paths']+=1
    STATS['locks']+=1
    return s

# A real projected chair is used for the construction and the finished chair.
seat=[cp(x,0,z) for x,z in [(-210,-145),(210,-145),(210,145),(-210,145)]]
guide=poly(seat,'none','#b18272',1.5)
for x in [-210,210]:
    for z in [-145,145]:guide+=poly([cp(x,-400,z),cp(x,0,z),cp(x,410,z)],'none','#aab5c2',.7)
for yy,col in [(-25,'#b39784'),(0,'#8098b4'),(39,'#b39784'),(65,'#b39784')]:
    guide+=hd([['M',-100,yy],['Q',0,yy+4,100,yy]],stroke=col,w=.65)
guide+=hd([['M',0,-125],['Q',-1,0,0,100]],stroke='#9e8ca2',w=.8)
landmarks={k:hp(*v) for k,v in {'left_eye':(-46,0),'right_eye':(46,0),'nose':(0,39),'mouth':(0,65),'chin':(0,100)}.items()}
for k,(x,y) in landmarks.items():
    guide+=e(x,y,2,2,'#a4747d')+f'<text x="{x+7}" y="{y-4}" font-family="Segoe UI" font-size="8" fill="#9b8082">{k}</text>'
add('01 White paper',p('M0 0H1000V1400H0Z','#fffdf8'))
add('02 Coordinate and chair perspective guides',guide,checkpoint='01-coordinate-plan.png')
sketch=''
for d in ['M395 274C382 173 557 132 588 266Q595 358 502 410Q410 382 395 274Z',
          'M445 399L431 430L398 442Q377 489 407 550L427 616L380 727Q388 808 333 914Q551 1042 817 956Q832 868 712 788L550 633L575 455L526 426L520 400',
          'M400 453L353 581Q361 646 473 692L546 675M571 461L615 562L583 664L555 684',
          'M635 850Q638 1010 596 1154M726 864Q753 1035 733 1149']:
    sketch+=p(d,stroke='#9da7b2',w=1.5,opacity=.7)
sketch+=p('M431 638Q439 722 483 763M483 763L635 850L596 1154M505 759L726 864L733 1149',stroke='#bc9ca1',w=1,opacity=.75)
for x,y in [(483,763),(635,850),(726,864),(596,1154),(733,1149)]:
    sketch+=e(x,y,8,8,'none',stroke='#bc9ca1',w=.8)
sketch+=p('M426 771Q467 787 518 780',stroke='#bc9ca1',w=1.1)
add('03 Pencil pose construction',sketch,strokes=[{'brush':'c) Pencil-2','size':1.1,'color':'#a4a3ae','opacity':.45,'path':[['M',398,442],['Q',476,458,575,455]]}],checkpoint='02-pose-sketch.png')

room=p('M0 0H1000V1400H0Z','url(#room)')
room+=p('M70 1080V158Q70 76 154 76H698Q782 76 782 158V1080Z','url(#glass)','#d4cabb',4)
room+=p('M88 1070V165Q88 94 159 94H692Q764 94 764 165V1070Z',stroke='#fffaf0',w=6)
for x in [265,510]:room+=p(f'M{x} 98V1075',stroke='#f8f3e9',w=9)
for y in [371,711]:room+=p(f'M84 {y}H768',stroke='#f7f1e6',w=9)
for j in range(24):
    x=R.uniform(90,755);y=R.uniform(150,1050)
    room+=e(x,y,R.uniform(16,45),R.uniform(35,95),R.choice(['#d5e2db','#b4cad0','#e6eee5']),R.uniform(.08,.22))
room+=p('M0 1100H1000V1400H0Z','#e2e5df')+p('M0 1100H1000',stroke='#c7c9c2',w=2)
for x in [-130,140,490,900,1230]:room+=p(f'M{-500+(x+500)*.5} 1100L{x} 1400',stroke='#c9d0cf',w=1,opacity=.6)
room+=e(578,1222,292,51,'url(#shade)')+e(120,340,550,520,'url(#light)')
add('04 Quiet window room',room,visibility={'02 Coordinate and chair perspective guides':False,'03 Pencil pose construction':False},checkpoint='03-room.png')

chair=''
for x,z in [(-210,-145),(210,-145),(-210,145),(210,145)]:
    chair+=poly([cp(x-11,0,z),cp(x+11,0,z),cp(x+19,410,z+19),cp(x-7,410,z+19)],'url(#wood)','#344c60',.8)
    a,b=cp(x-3,30,z+2),cp(x+5,390,z+19);chair+=p('M%.2f %.2fL%.2f %.2f'%(*a,*b),stroke='#b5aa86',w=.8)
chair+=poly([cp(-210,-390,-145),cp(210,-390,-145),cp(210,0,-145),cp(-210,0,-145)],'url(#wood)','#53657a',2)
chair+=poly([cp(-183,-367,-143),cp(183,-367,-143),cp(183,-28,-143),cp(-183,-28,-143)],'url(#velvet)','#b1a98f',1)
for xx in [-120,0,120]:
    for yy in [-300,-160,-50]:
        x,y=cp(xx,yy,-141);chair+=e(x,y,2.4,2.4,'#637e96')
chair+=poly(seat,'url(#velvet)','#526c82',2)
chair+=poly([cp(-210,0,145),cp(210,0,145),cp(210,23,145),cp(-210,23,145)],'url(#wood)','#b0a481',1)
add('05 Projected chair with cushion and four legs',chair,checkpoint='04-chair.png')

rear=''
for q,w,n in [([(473,192),(320,237),(344,631),(261,1063)],53,100),
              ([(488,191),(349,312),(383,759),(354,1121)],41,95),
              ([(513,194),(665,263),(613,747),(794,1096)],57,110),
              ([(520,209),(601,354),(587,839),(680,1148)],42,95),
              ([(472,210),(410,418),(370,877),(420,1095)],32,75)]:rear+=lock(q,w,n,True)
add('06 Long rear hair grouped by flow',rear)

legs=p('M572 865Q615 850 664 885L640 1049Q630 1113 611 1167Q595 1191 574 1167L591 1021Z','url(#ivory)','#8297ac',1.2)
legs+=p('M690 879Q731 865 771 895Q770 1005 754 1082L751 1160Q733 1187 709 1161L709 1061Z','url(#ivory)','#8297ac',1.2)
legs+=p('M630 935Q628 1074 600 1147M745 945Q748 1073 735 1149',stroke='#fff9f0',w=7,opacity=.55)
legs+=p('M572 1149Q590 1180 615 1156L614 1190Q600 1220 553 1226Q530 1224 544 1200Z','url(#dress)','#708ca7',1.2)
legs+=p('M710 1152Q731 1178 753 1152L766 1196Q755 1221 715 1210Q698 1200 710 1152Z','url(#dress)','#708ca7',1.2)
legs+=p('M547 1215Q578 1217 611 1185M713 1201Q738 1211 760 1193',stroke='#f9f7ea',w=2)
for x,y in [(582,1187),(735,1187)]:
    legs+=p(f'M{x} {y}q-30 -23 -30 -3q11 14 30 3q24 -25 27 -8q0 13 -27 8Z','#eaf1f5','#839fb8',.8)+e(x,y,3,4,'#b8a47a')
add('07 Stockings and blue ribbon shoes',legs)

skin=p('M444 378Q446 419 424 432L403 446Q478 489 566 455L529 424L521 376Z','url(#skin)','#b48d92',.9)
skin+=p('M445 390Q482 415 522 385L524 412Q482 439 441 410Z','#d8acb1',opacity=.45)
add('08 Neck shoulders and cast shadow',skin)

dress=p('M398 442Q421 458 456 475Q500 461 544 446L575 455Q588 514 563 566L550 633Q595 681 677 732Q757 775 803 862Q829 912 817 959Q700 1019 558 993Q432 1004 323 947Q337 857 377 784Q416 699 427 616Q400 547 398 442Z','url(#dress)','#7692ad',1.25)
dress+=p('M399 443Q417 456 456 475Q505 456 544 446L563 458Q507 474 456 494Q418 473 397 465Z','url(#ivory)','#a5b9cb',.9)
dress+=p('M427 616Q477 650 550 633L564 651Q482 675 421 637Z','#819db9','#adc4d5',.8)
dress+=p('M431 622Q476 651 550 639',stroke='#faf5e9',w=2)
# Folds are tied to waist/lap tension rather than evenly spaced vertical stripes.
fold_specs=[('M435 657Q424 791 349 935Q378 955 407 958Q455 804 452 666Z','#7695b3',.48),
 ('M468 671Q486 802 434 971L472 981Q521 840 484 675Z','#f7faf9',.65),
 ('M506 671Q565 789 521 984L562 989Q591 798 521 670Z','#829ebb',.46),
 ('M540 662Q621 771 615 990L652 989Q640 793 554 666Z','#f1f7f8',.62),
 ('M585 703Q698 821 698 986L729 981Q719 830 600 715Z','#829fbc',.45),
 ('M645 740Q775 837 782 969L805 960Q795 833 665 751Z','#f2f7f7',.58)]
for d,col,op in fold_specs:dress+=p(d,col,opacity=op)
for d in ['M438 513Q449 569 450 617','M539 490Q512 568 519 635','M456 495Q477 565 472 636','M460 691Q453 751 432 793','M558 690Q601 743 619 795','M671 780Q723 833 739 887']:
    dress+=p(d,stroke='#7695b0',w=1.2,opacity=.65)
dress+=p('M325 931Q454 987 561 975Q700 1003 819 944L817 959Q700 1019 558 993Q432 1004 323 947Z','url(#ivory)','#b2bfcd',.8)
add('09 Seated dress silhouette and tension folds',dress,checkpoint='05-dress-and-pose.png')
shadows=p('M402 478Q404 539 427 616L443 623Q424 539 421 486Z','#627e9f',opacity=.22)
shadows+=p('M548 477Q533 556 530 638L550 633Q566 563 561 493Z','#6683a3',opacity=.22)
shadows+=p('M431 663Q475 711 588 726Q666 749 715 803Q631 749 566 751Q484 754 419 722Z','#6f8fac',opacity=.15)
shadows+=p('M337 913Q440 954 544 953Q697 987 817 932L817 944Q700 1003 561 975Q454 987 325 931Z','#6988aa',opacity=.18)
add('09b Bodice side planes and lap cast shadows',shadows)

# Sleeves, forearms, and a small closed book create a concrete hand action.
arms=p('M399 444Q366 450 356 488Q357 510 384 518L413 494Z','url(#ivory)','#a2b1c1',1)
arms+=p('M570 456Q598 459 613 497Q608 519 587 523L561 493Z','url(#ivory)','#a2b1c1',1)
arms+=p('M363 505Q350 551 363 591Q397 634 462 665L478 647Q419 603 397 575L398 517Z','url(#skin)','#b99196',1)
arms+=p('M592 515Q608 551 600 580L558 657L539 644Q568 584 568 568L565 522Z','url(#skin)','#b99196',1)
arms+=p('M370 523Q365 561 377 582Q411 620 455 642',stroke='#fff3e7',w=5,opacity=.7)
arms+=p('M421 632L568 616L610 687L460 709Z','#526e8c','#435f7b',1.5)
arms+=p('M430 636L563 622L600 681L465 700Z','#d6d6ce','#bcbdaE',1)
arms+=p('M427 640L563 628L603 687L463 706Z','#6f93ae','#aac3d2',1)
arms+=p('M443 649L557 639L582 678L470 692Z',stroke='#d5c394',w=.8)
arms+=p('M425 610Q452 634 479 648L462 665Q440 651 425 628Z','url(#skin)','#b99196',.7)
arms+=p('M566 615L581 620L559 661L541 644Z','url(#skin)','#b99196',.7)
arms+=p('M479 648Q491 646 508 657L528 669Q534 675 528 677L505 669L525 682Q530 689 523 690L499 677L517 692Q518 700 510 696L483 679Q466 673 462 664Z','url(#skin)','#b89096',.85)
arms+=p('M545 638Q534 644 530 656L522 670Q520 676 525 677L536 665L532 683Q534 690 539 685L547 669L544 686Q547 692 552 686L559 662Q559 650 552 644Z','url(#skin)','#b89096',.85)
add('10 Puff sleeves hands and a quiet book',arms,checkpoint='06-hands-and-book.png')

face=hd([['M',-84,-68],['C',-88,-114,-38,-131,6,-127],['C',59,-125,88,-96,87,-52],['C',91,10,76,56,40,82],['Q',18,100,0,100],['Q',-40,86,-70,52],['Q',-89,17,-84,-68],['Z']],'url(#face)','#b7979c',.9)
for x in [-80,85]:
    face+=hd([['M',x,6],['Q',x+(12 if x>0 else -12),-5,x+(12 if x>0 else -12),22],['Q',x+7,44,x,34]],'url(#skin)','#bc969b',.8)
add('11 Face and ear base aligned to head camera',face,checkpoint='07-face-base.png')
face_shadow=hd([['M',-79,-50],['Q',-62,-56,-45,-78],['Q',-57,-49,-79,-36],['Z']],'#c5a6ad',opacity=.18)
face_shadow+=hd([['M',28,-83],['Q',59,-57,80,-57],['L',82,-44],['Q',47,-48,28,-83],['Z']],'#c5a6ad',opacity=.18)
add('11b Soft shadows under the bangs',face_shadow)

brows=''
for cx in [-46,46]:
    brows+=hd([['M',cx-27,-23],['Q',cx,-31,cx+25,-24]],stroke='#9d9aa2',w=1.2)
    brows+=hd([['M',cx-27,-17],['Q',cx,-23,cx+25,-17]],stroke='#cfacb2',w=.4,opacity=.18)
add('12 Fine low arched eyebrows',brows)
eyes=''
for side in [-1,1]:
    cx=side*46
    mask=hd([['M',cx-28,0],['C',cx-15,-13,cx+12,-15,cx+28,-2],['Q',cx+9,17,cx-10,11],['Q',cx-21,8,cx-28,0],['Z']],'#ffffff')
    eye_white=hd([['M',cx-28,0],['C',cx-15,-13,cx+12,-15,cx+28,-2],['Q',cx+9,17,cx-10,11],['Q',cx-21,8,cx-28,0],['Z']],'#faf6f1')
    add(f'13 Eye {side} warm white',eye_white)
    # All iris contents are clipped by the actual projected almond aperture.
    x,y=hp(cx-2,0);ir=e(x,y,13.9,17,'url(#iris)',stroke='#788aa0',w=.65)
    for j in range(48):
        a=j*math.tau/48;rr=R.uniform(8,11);rr2=R.uniform(12,15)
        ir+=p('M%.2f %.2fL%.2f %.2f'%(x+math.cos(a)*rr,y+math.sin(a)*rr,x+math.cos(a)*rr2*.85,y+math.sin(a)*rr2),stroke=R.choice(['#657b97','#bfd3dc','#e4ebdf']),w=.4,opacity=.55)
    ir+=e(x,y-1,3.8,6.6,'#56677e')+e(x-5,y-7,3.1,2.5,'#fffdf4',.9)+e(x+6,y+8,1.3,1.2,'#f5faf8',.8)
    add(f'14 Eye {side} grey blue iris',ir,svg_masks=[svg(mask)])
    eyes+=hd([['M',cx-28,0],['C',cx-15,-13,cx+12,-15,cx+28,-2]],stroke='#646778',w=2.15)
    eyes+=hd([['M',cx-20,8],['Q',cx+4,20,cx+24,5]],stroke='#b695a1',w=.75)
    outer=cx+side*27
    for j in range(3):
        eyes+=hd([['M',outer-side*j*3,-2-j*.8],['Q',outer+side*4,-5-j*2,outer+side*(6-j),-8-j*2]],stroke='#6e6b7b',w=1.1-j*.18)
    eyes+=hd([['M',cx-26,-8],['Q',cx,-23,cx+26,-10]],stroke='#c4a4ad',w=.6,opacity=.8)
add('15 Almond eyelids and restrained lashes',eyes,checkpoint='08-eyes.png')

x,y=hp(0,39,67)
features=p(f'M{x-2:.2f} {y-4:.2f}q-3 6 2 7',stroke='#c3969b',w=.8)+e(x+3,y-3,3.2,2.2,'#fff8ef',.9)
features+=hd([['M',-10,65],['Q',0,64,10,65]],stroke='#af818d',w=.7)
features+=hd([['M',-3,70],['Q',1,71,5,70]],stroke='#fff7e9',w=.7,opacity=.6)
for xx in [-57,60]:
    x,y=hp(xx,38);features+=e(x,y,24,13,'url(#blush)',.65)
add('16 Short nose mark small parted mouth and blush',features,checkpoint='09-five-features.png')

front=''
# Crown and bangs are connected to roots; strand density follows lock widths.
for q,w,n in [([(491,181),(415,175),(391,225),(390,286)],29,70),
              ([(487,182),(464,220),(445,261),(411,287)],31,85),
              ([(491,182),(519,221),(538,236),(568,253)],30,85),
              ([(497,185),(574,196),(581,246),(581,306)],26,70),
              ([(398,261),(380,391),(416,468),(366,586)],24,75),
              ([(576,260),(600,399),(560,465),(599,571)],27,85),
              ([(400,310),(387,427),(402,475),(386,529)],12,45),
              ([(580,318),(578,424),(589,466),(616,504)],12,45)]:front+=lock(q,w,n)
add('17 Rooted bangs and shoulder locks with fine fibers',front,checkpoint='10-hair.png')

dec=''
# Dress edge embroidery follows the curved hem, not a rectangular repeat strip.
for j in range(65):
    t=j/64;x=329+484*t;y=941+44*math.sin(math.pi*t)+11*t
    dec+=e(x,y,2.2,3.5,'none',stroke='#fff9ef',w=.65)
    if j%3==0:
        dec+=p(f'M{x} {y-7}q-8 -12 -10 -4q2 9 10 4q9 -14 11 -5q0 7 -11 5',stroke='#eef2ee',w=.6)
        dec+=e(x,y-10,1.1,1.3,'#b8a77c')
for x,y in [(426,454),(548,459)]:
    dec+=p(f'M{x} {y}q-22 -22 -28 -5q7 18 28 5q25 -22 29 -4q-4 16 -29 4Z','url(#dress)','#8aa7bd',.7)+e(x,y,3.3,3.8,'#b4a078')
dec+=p('M443 423Q478 441 519 422',stroke='#d3bd91',w=.9)
dec+=p('M485 438L491 448L484 459L478 448Z','#89b5cd','#b8a67d',.7)
dec+=p('M398 255Q378 239 376 254Q381 267 398 259Q386 276 377 269Q374 259 398 255Z','url(#dress)','#7897b0',.7)+e(398,257,4,4,'#faf2df')
for x,y in [hp(-86,28),hp(88,27)]:
    dec+=p(f'M{x} {y+2}v17',stroke='#b8a680',w=.7)+e(x,y+22,3,5,'#9dc6d9',stroke='#ae9d7e',w=.6)
add('18 Embroidered hem ribbon and small jewelry',dec)

native=[]
for q in [[(406,293),(387,360),(403,453),(370,559)],[(576,285),(592,372),(572,453),(599,549)]]:
    for i in range(12):
        pts=[strand(q,j/18,(i-6)/6,15) for j in range(19)]
        cmds=[['M',*pts[0]]]+[['L',*xy] for xy in pts[1:]]
        native.append({'brush':'d) Ink-3 Gpen','size':.4,'color':'#d8edf6','opacity':.45,'path':cmds})
add('19 Native Gpen hair finishing',strokes=native)

# A few restrained foreground stems and diffuse window light give depth.
flowers=p('M91 1330Q117 1181 86 1055M110 1310Q165 1172 162 1095M95 1238Q58 1184 45 1152',stroke='#9ba99d',w=1.2)
for x,y in [(86,1062),(90,1104),(96,1148),(163,1100),(157,1137),(145,1178),(46,1157),(65,1190)]:
    for a in range(0,360,72):
        rad=math.radians(a);flowers+=e(x+math.cos(rad)*6,y+math.sin(rad)*6,5,7,'#f6ede1',.8,stroke='#d9cbbb',w=.4)
    flowers+=e(x,y,2,2,'#c6b68b')
add('20 Small foreground flowers',flowers)
light=e(180,180,470,520,'url(#light)',.45)
for j in range(35):
    x=R.uniform(90,840);y=R.uniform(160,1160)
    if 340<x<650 and y<750:continue
    light+=e(x,y,R.uniform(.5,1.8),R.uniform(.5,1.8),'#fffdf2',R.uniform(.3,.75))
add('21 Soft window atmosphere',light,checkpoint='11-complete.png')

# Explicit construction -> clean line art -> flat color checkpoints precede
# material rendering. These use exactly the same new coordinates, not a trace
# of an existing raster. Diagnostic layers remain editable in the KRA.
source=copy.deepcopy(LAYERS)
outline=[];flat=[];fiber_layers=[]
flat_palette={'wood':'#465c70','velvet':'#8aa4b8','hairDark':'#93b3ca',
              'hair':'#bdd8e8','skin':'#f6e0d5','face':'#f9e8dc',
              'dress':'#b4cddd','ivory':'#f0eee8','iris':'#9fb7c5',
              'room':'#e9eeeb','glass':'#dcebed'}
for layer_index,layer in enumerate(source[3:],start=3):
    root=ET.fromstring(layer['svg'])
    keep=[];fibers=[]
    for node in root.iter():
        tag=node.tag.split('}')[-1]
        if tag not in ['path','ellipse']:continue
        node=copy.deepcopy(node);node.tag=tag
        fill=node.get('fill','none');stroke=node.get('stroke','none')
        width=float(node.get('stroke-width','1'))
        if layer['name'].startswith(('06 ','17 ')) and fill=='none' and width<.6:
            fibers.append(ET.tostring(node,encoding='unicode'))
            continue
        keep.append(ET.tostring(node,encoding='unicode'))
        if layer['name'].startswith(('04 ','20 ','21 ')):continue
        if stroke!='none' and .6<=width<=3 and 'iris' not in layer['name']:
            line=copy.deepcopy(node);line.set('fill','#fffdf8' if fill!='none' else 'none');line.set('stroke','#8c8893');line.set('opacity','1')
            outline.append(ET.tostring(line,encoding='unicode'))
        if fill!='none' and not layer['name'].startswith(('09b','11b','12 ','15 ','16 ','18 ','19 ')):
            base=copy.deepcopy(node)
            if fill.startswith('url('):
                key=re.search(r'#([^)]*)',fill).group(1)
                if key not in flat_palette:continue
                base.set('fill',flat_palette[key])
            base.set('stroke','none')
            flat.append(ET.tostring(base,encoding='unicode'))
    if layer['name'].startswith(('06 ','17 ')):
        layer['svg']=svg(''.join(keep))
        # Fibers are painted later, but their visibility respects every opaque
        # object that originally followed their hair lock in painter order.
        mask=''
        for occluder in source[layer_index+1:]:
            if occluder['name'].startswith(('09b','11b','12 ','15 ','16 ','19 ','20 ','21 ')):continue
            for node in ET.fromstring(occluder['svg']).iter():
                if node.tag.split('}')[-1] not in ['path','ellipse'] or node.get('fill','none')=='none':continue
                nn=copy.deepcopy(node);nn.tag=nn.tag.split('}')[-1]
                nn.set('fill','#ffffff');nn.set('stroke','none');nn.set('opacity','1')
                mask+=ET.tostring(nn,encoding='unicode')
        fiber_layers.append({'name':'19b Fine fibers for '+layer['name'],'svg':svg(''.join(fibers)),'svg_erase_masks':[svg(mask)]})

flats={'name':'03b Flat color block in','svg':svg(p('M0 0H1000V1400H0Z','#f5f4ec')+''.join(flat)),
       'visible':False}
lineart={'name':'03c Clean line art before color','svg':svg(''.join(outline)),
 'visibility':{'02 Coordinate and chair perspective guides':False,'03 Pencil pose construction':False},
 'strokes':[{'brush':'d) Ink-3 Gpen','size':.8,'color':'#aaa0a7','opacity':.7,
             'path':[['M',398,442],['Q',416,456,456,475]]}],
 'checkpoint':'02b-clean-line-art.png'}
flat_checkpoint={'name':'03d Flat color checkpoint','svg':svg(''),
                 'visibility':{'03b Flat color block in':True,'03c Clean line art before color':False},'checkpoint':'02c-flat-colors.png'}
source[3].setdefault('visibility',{})['03c Clean line art before color']=False
source[3]['visibility']['03b Flat color block in']=False
fiber_layers[-1]['checkpoint']='10b-fine-hair-detail.png'
LAYERS=source[:1]+[flats]+source[1:3]+[lineart,flat_checkpoint]+source[3:-2]+fiber_layers+source[-2:]

scene={'name':PLAN['title'],'width':2400,'height':3360,'scale':2.4,'stem':'azure-quiet-page',
 'blank_checkpoint':'00-new-blank.kra','layers':LAYERS,
 'details':[{'file':'face-detail.png','rect':[365,168,258,258]},
            {'file':'hands-detail.png','rect':[340,500,285,225]},
            {'file':'hair-and-fabric-detail.png','rect':[275,675,550,365]}]}
(OUT/'scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'construction-record.json').write_text(json.dumps({'landmarks':landmarks,'chair_seat_corners':seat,**STATS,'layers':len(LAYERS)},indent=2),encoding='utf-8')
print(json.dumps({'scene':str(OUT/'scene.json'),**STATS,'layers':len(LAYERS)}))
