"""Original artwork construction snapshot for the new blank-canvas study.

No earlier illustration geometry, raster, SVG or scene is loaded. This is a
research recipe; tools/krita_studio.py remains the reusable Krita renderer.
"""
import json
import math
import random
from pathlib import Path

OUT = Path('data/test-result/20261016/krita-anime-study-v3')
OUT.mkdir(parents=True, exist_ok=True)
R = random.Random(161006)
stats = {'hair_fiber_paths':0,'hair_locks':0,'iris_texture_paths':0}

DEFS='''<defs>
<linearGradient id="paper" x1="0%" y1="0%" x2="80%" y2="100%"><stop stop-color="#fffdf9"/><stop offset=".6" stop-color="#f0f5fb"/><stop offset="1" stop-color="#e5edf8"/></linearGradient>
<linearGradient id="window" x1="0%" y1="0%" x2="100%" y2="100%"><stop stop-color="#b7d0e0"/><stop offset=".5" stop-color="#dfeaf0"/><stop offset="1" stop-color="#9fbad5"/></linearGradient>
<linearGradient id="hair" x1="0%" y1="0%" x2="100%" y2="35%"><stop stop-color="#84abd1"/><stop offset=".25" stop-color="#cee7f7"/><stop offset=".48" stop-color="#f2faff"/><stop offset=".72" stop-color="#b5d6ed"/><stop offset="1" stop-color="#749bc5"/></linearGradient>
<linearGradient id="hair2" x1="0%" y1="0%" x2="100%" y2="65%"><stop stop-color="#edf9ff"/><stop offset=".27" stop-color="#d6edf9"/><stop offset=".6" stop-color="#a4c7e2"/><stop offset="1" stop-color="#6e96bf"/></linearGradient>
<linearGradient id="hair3" x1="0%" y1="0%" x2="100%" y2="0%"><stop stop-color="#96b9d9"/><stop offset=".45" stop-color="#d8edf8"/><stop offset=".7" stop-color="#c7e1f2"/><stop offset="1" stop-color="#8caaca"/></linearGradient>
<linearGradient id="skin" x1="0%" y1="10%" x2="85%" y2="100%"><stop stop-color="#fff7ee"/><stop offset=".45" stop-color="#f9e2dc"/><stop offset=".82" stop-color="#f0cdcc"/><stop offset="1" stop-color="#daaeb7"/></linearGradient>
<linearGradient id="face" x1="10%" y1="10%" x2="85%" y2="100%"><stop stop-color="#fff6ed"/><stop offset=".6" stop-color="#fbe9e0"/><stop offset="1" stop-color="#edc6c6"/></linearGradient>
<linearGradient id="silk" x1="0%" y1="0%" x2="100%" y2="35%"><stop stop-color="#789bc5"/><stop offset=".25" stop-color="#d0e5f4"/><stop offset=".46" stop-color="#f9fcff"/><stop offset=".66" stop-color="#c1d8ee"/><stop offset="1" stop-color="#738db8"/></linearGradient>
<linearGradient id="skirt" x1="0%" y1="0%" x2="80%" y2="100%"><stop stop-color="#e9f4fd"/><stop offset=".3" stop-color="#bacfe9"/><stop offset=".6" stop-color="#edf6ff"/><stop offset="1" stop-color="#89a5c8"/></linearGradient>
<linearGradient id="stock" x1="0%" y1="0%" x2="100%" y2="25%"><stop stop-color="#b3bcd4"/><stop offset=".32" stop-color="#f5f5fd"/><stop offset=".6" stop-color="#e7e9f5"/><stop offset="1" stop-color="#abb6d1"/></linearGradient>
<linearGradient id="chair" x1="0%" y1="0%" x2="100%" y2="0%"><stop stop-color="#324766"/><stop offset=".4" stop-color="#6a7e99"/><stop offset=".7" stop-color="#526b87"/><stop offset="1" stop-color="#253b59"/></linearGradient>
<linearGradient id="velvet" x1="0%" y1="0%" x2="80%" y2="100%"><stop stop-color="#c9d9e6"/><stop offset=".4" stop-color="#a1b8cf"/><stop offset="1" stop-color="#728da9"/></linearGradient>
<linearGradient id="iris" x1="0%" y1="0%" x2="0%" y2="100%"><stop stop-color="#2b355d"/><stop offset=".32" stop-color="#525995"/><stop offset=".58" stop-color="#7c87c5"/><stop offset=".8" stop-color="#9bc8e8"/><stop offset="1" stop-color="#c3e6ef"/></linearGradient>
<linearGradient id="irisShade" x1="0%" y1="0%" x2="0%" y2="100%"><stop stop-color="#242d55" stop-opacity=".8"/><stop offset=".55" stop-color="#343d68" stop-opacity=".18"/><stop offset="1" stop-color="#5b6aa4" stop-opacity="0"/></linearGradient>
<radialGradient id="blush"><stop stop-color="#de93a8" stop-opacity=".48"/><stop offset="1" stop-color="#edb6c1" stop-opacity="0"/></radialGradient>
<radialGradient id="glow"><stop stop-color="#ffffff" stop-opacity=".8"/><stop offset="1" stop-color="#f4faff" stop-opacity="0"/></radialGradient>
<radialGradient id="shadow"><stop stop-color="#7187a8" stop-opacity=".3"/><stop offset="1" stop-color="#8595b2" stop-opacity="0"/></radialGradient>
</defs>'''

def svg(body):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="3600" viewBox="0 0 1000 1500">'+DEFS+'<g stroke-linecap="round" stroke-linejoin="round">'+body+'</g></svg>'
def p(d, fill='none', stroke='none', w=1, extra=''):
    return '<path d="%s" fill="%s" stroke="%s" stroke-width="%.3f" %s/>'%(d,fill,stroke,w,extra)
def el(x,y,rx,ry,fill,stroke='none',w=1,extra=''):
    return '<ellipse cx="%.3f" cy="%.3f" rx="%.3f" ry="%.3f" fill="%s" stroke="%s" stroke-width="%.3f" %s/>'%(x,y,rx,ry,fill,stroke,w,extra)
def native(size,color,commands,opacity=1,brush='d) Ink-3 Gpen'):
    return {'brush':brush,'size':size,'color':color,'path':commands,'opacity':opacity}
def add(name,body=None,strokes=None,**kw):
    item={'name':name,**kw}
    if body is not None:item['svg']=svg(body)
    if strokes:item['strokes']=strokes
    layers.append(item)
    return item

def bezier(q,t):
    a=(1-t)
    return (a*a*a*q[0][0]+3*a*a*t*q[1][0]+3*a*t*t*q[2][0]+t*t*t*q[3][0],a*a*a*q[0][1]+3*a*a*t*q[1][1]+3*a*t*t*q[2][1]+t*t*t*q[3][1])
def hair_point(q,t,offset,width,phase=0):
    x,y=bezier(q,t)
    x1,y1=bezier(q,max(0,t-.001));x2,y2=bezier(q,min(1,t+.001))
    dx,dy=x2-x1,y2-y1;length=max(.001,math.hypot(dx,dy))
    # Grow out of a narrow root rather than exposing a flat ribbon end.
    root=.025+.975*math.sin(math.pi*min(1,t/.20)/2)
    profile=root*(.78+.22*math.sin(math.pi*t))*(1-t)**.42
    shift=offset*width*profile+math.sin(t*9+phase)*1.1*math.sin(math.pi*t)
    return (x-dy/length*shift,y+dx/length*shift)
def smooth(points):
    d='M%.3f %.3f'%points[0]
    for i in range(len(points)-1):
        a=points[max(0,i-1)];b=points[i];c=points[i+1];dd=points[min(len(points)-1,i+2)]
        c1=(b[0]+(c[0]-a[0])/6,b[1]+(c[1]-a[1])/6)
        c2=(c[0]-(dd[0]-b[0])/6,c[1]-(dd[1]-b[1])/6)
        d+='C%.3f %.3f %.3f %.3f %.3f %.3f'%(*c1,*c2,*c)
    return d
def hair_lock(q,width,fibers=55,gradient='hair',seed=0):
    rr=random.Random(seed)
    side1=[hair_point(q,i/35,1,width) for i in range(36)]
    side2=[hair_point(q,i/35,-1,width) for i in range(35,-1,-1)]
    d=smooth(side1)+'L%.3f %.3f'%side2[0]+smooth(side2)[smooth(side2).find('C'):]+'Z'
    body=p(d,'url(#%s)'%gradient,'#718fae',.75)
    # Each group of hair has separate dark fibers, lit fibers and short filaments.
    for j in range(fibers):
        offset=-.97+1.94*(j+rr.uniform(-.25,.25))/max(1,fibers-1)
        start=rr.uniform(.015,.15);end=rr.uniform(.78,.995)
        points=[hair_point(q,start+(end-start)*i/24,offset,width,j*.8) for i in range(25)]
        lit=(j%5==0 or (abs(offset)<.27 and j%2==0))
        color=rr.choice(['#f8fdff','#e9f7ff','#cfe6f6']) if lit else rr.choice(['#668db5','#769ec2','#86adca','#9bbbd6'])
        body+=p(smooth(points),stroke=color,w=rr.uniform(.22,.55) if not lit else rr.uniform(.4,.85),extra='opacity="%.2f"'%rr.uniform(.32,.76))
        stats['hair_fiber_paths']+=1
    # A few discontinuous highlight filaments, instead of one flat white stripe.
    for j in range(max(4,fibers//10)):
        offset=rr.uniform(-.45,.4);t=rr.uniform(.17,.74);end=min(.92,t+rr.uniform(.045,.13))
        points=[hair_point(q,t+(end-t)*i/12,offset,width,j) for i in range(13)]
        body+=p(smooth(points),stroke='#f4fbff',w=rr.uniform(.65,1.15),extra='opacity=".7"')
        stats['hair_fiber_paths']+=1
    stats['hair_locks']+=1
    return body

layers=[]
paper=p('M0 0H1000V1500H0Z','url(#paper)')
paper+=p('M226 1330V362C226 193 328 103 510 103C690 103 792 193 792 362V1330Z','url(#window)','#c7bba5',1.3)
paper+=p('M240 1314V364C240 204 337 118 510 118C681 118 778 204 778 364V1314Z',stroke='#f8f6ed',w=2)
paper+=p('M510 118V1320M236 463H784M236 891H784',stroke='#f8f7ee',w=1.6,extra='opacity=".6"')
paper+=el(508,400,204,208,'#f6f9f7',extra='opacity=".34"')
paper+=el(501,1340,262,43,'url(#shadow)')
paper+='<rect x="41" y="40" width="918" height="1402" rx="160" fill="none" stroke="#d4c6b2" stroke-width="1"/>'
add('01 Ivory paper and quiet arched window',paper)

bot=''
for side in [-1,1]:
    for k in range(5):
        x=510+side*(125+k*28);y=180+k*190
        bot+=p('M%d %dQ%d %d %d %d'%(x,y+240,x-side*35,y+100,x+side*40,y),stroke='#668ea9',w=1.2,extra='opacity=".11"')
        for j in range(8):
            xx=x+side*(23-j*5);yy=y+j*28
            bot+=el(xx,yy,9,23,'#84a7bd',extra='opacity=".10" transform="rotate(%s %.1f %.1f)"'%(side*(-32+(j%2)*64),xx,yy))
add('02 Understated botanical shadows',bot)

# Individually constructed locks: no old image or previous scene is read.
back_locks=[([(455,255),(310,422),(385,1005),(181,1242)],46,70),
 ([(435,278),(330,539),(222,1019),(298,1308)],50,72),
 ([(461,290),(410,607),(287,1044),(366,1249)],55,76),
 ([(490,250),(516,644),(426,1045),(475,1315)],67,77),
 ([(524,252),(563,565),(672,982),(794,1220)],55,76),
 ([(549,284),(631,559),(763,1027),(716,1309)],50,73),
 ([(532,308),(581,650),(558,1021),(653,1265)],52,74)]
back=''
for i,(q,width,count) in enumerate(back_locks):back+=hair_lock(q,width,count,'hair' if i%2 else 'hair3',100+i)
add('03 Long back hair - individually drawn fiber groups',back)

chair=p('M312 805L293 1304Q293 1323 312 1324L347 827M636 822L683 1308Q688 1325 706 1317L674 805','url(#chair)','#3f526b',1.5)
chair+=p('M320 794V558Q320 514 357 502L569 494Q626 501 630 552L650 801Z','url(#chair)','#43546d',1.3)
chair+=p('M341 783V558Q342 532 366 523L562 515Q601 523 608 556L628 782Z','url(#velvet)','#cfb98d',1.1)
chair+=p('M354 761V560Q356 544 375 540L556 531Q585 535 590 561L609 762Z',stroke='#dfe3dc',w=1)
for i in range(5):
    for j in range(3):
        x=375+i*44+j*3;y=571+j*65
        chair+=p('M%.1f %.1fl22 32 -22 33 -22 -33Z'%(x,y),stroke='#b7c6d3',w=.7)
        chair+=el(x,y,2.3,2.3,'#e1e8eb')
chair+=p('M287 789Q448 765 684 791L694 839Q489 874 284 840Z','url(#chair)','#3c506b',1.5)
chair+=p('M290 786Q494 759 682 788L679 812Q484 844 290 814Z','url(#velvet)','#c3ad89',1.1)
chair+=p('M291 838Q489 867 689 836M310 1290L334 861M690 1292L653 860',stroke='#bfa77e',w=1.4)
chair+=p('M289 665L299 805L315 805L308 665M653 659L662 802L678 802L671 657','url(#chair)','#455875',1.2)
chair+=p('M278 648Q302 636 344 647L347 666Q311 677 280 665ZM615 642Q650 631 680 642L683 662Q651 671 618 660Z','url(#chair)','#455875',1.2)
chair+=p('M287 654Q314 646 339 655M626 648Q651 640 675 648',stroke='#c6af83',w=1.2)
for x,y in [(312,861),(668,855)]:
    chair+=p('M%d %dq16 0 17 15q-4 15 -17 7q-8 -11 2 -10q7 3 0 6'%(x,y),stroke='#c5af88',w=1.1)
add('04 Upholstered chair - seat and perspective',chair)

# Bent legs: round knees in front of seat, then diagonally receding shins.
legs=p('M443 800C447 776 493 775 528 813C560 844 578 879 570 919C558 968 536 1049 515 1136L492 1235Q491 1274 468 1294Q449 1306 439 1289Q430 1276 442 1257L456 1229C463 1135 470 1057 478 991C484 944 458 919 434 888C414 858 423 825 443 800Z','url(#stock)','#8f94b0',1)
legs+=p('M525 795C562 780 612 800 643 837C677 876 690 899 680 941C661 1004 643 1084 628 1172L620 1239Q620 1264 603 1282Q583 1298 570 1284Q561 1274 570 1256L583 1230C587 1142 592 1070 601 1001C606 952 591 937 565 908C537 880 508 850 525 795Z','url(#stock)','#8f94b0',1)
legs+=p('M449 1259Q468 1279 491 1255L490 1280Q473 1316 448 1313Q429 1307 437 1285Z','url(#silk)','#778cac',1)
legs+=p('M576 1256Q600 1277 620 1250L620 1275Q607 1307 583 1307Q563 1303 568 1284Z','url(#silk)','#778cac',1)
legs+=p('M444 1286Q467 1296 485 1277M574 1284Q598 1294 615 1273',stroke='#ecf6fd',w=2)
for x,y in [(466,1292),(595,1288)]:
    legs+=p('M%d %dq-23 -21 -26 -7q2 15 26 7q19 -25 26 -12q1 15 -26 12Z'%(x,y),'#eaf3fb','#8c9eb6',.7)+el(x,y,3.8,4.3,'#c6b18d')
leg_group={'name':'05 Bent legs and satin slippers','children':[{'name':'Leg shapes from new pose','svg':svg(legs)},{'name':'Knee form light','inherit_alpha':True,'strokes':[
native(40,'#a4a4be',[['M',541,877],['Q',555,918,527,1000]],.16,'b) Airbrush Soft'),native(36,'#b09fb7',[['M',652,876],['Q',681,906,651,972]],.14,'b) Airbrush Soft'),native(23,'#fff9f4',[['M',502,855],['Q',533,877,537,895]],.52,'b) Airbrush Soft'),native(21,'#fff9f4',[['M',613,853],['Q',642,876,647,893]],.5,'b) Airbrush Soft'),native(9,'#ffffff',[['M',506,1007],['Q',480,1163,474,1211]],.45,'b) Airbrush Soft'),native(9,'#ffffff',[['M',627,1006],['Q',606,1157,603,1215]],.4,'b) Airbrush Soft')]}]}
layers.append(leg_group)

skin=p('M451 411L451 454Q419 466 391 486C373 501 369 533 367 563L351 636Q350 660 372 676L394 666L413 558L421 608L539 623L565 557L584 637L608 628L589 518Q584 489 555 478L519 458L516 408Z','url(#skin)','#ac8b99',1)
skin+=p('M452 424Q485 448 517 421L518 447Q489 472 451 450Z','#ddb1b8',extra='opacity=".55"')
skin+=p('M451 455Q484 474 520 456L520 468Q485 484 449 468Z','#e7eff8','#91a3bc',.8)
skin+=p('M487 478L486 493',stroke='#af9a8a',w=1)+p('M487 490q11 10 0 20q-10 -10 0 -20Z','#94c5dd','#b2a07f',.8)
bodice=p('M400 510Q430 507 476 538Q521 512 563 514L552 579Q529 605 540 652L551 692Q484 727 414 688L425 645Q432 604 409 577Z','url(#silk)','#7e92b4',1)
bodice+=p('M402 518Q443 526 476 548Q520 523 560 522M408 527Q445 536 476 557Q520 532 557 531',stroke='#f6fcff',w=2)
bodice+=p('M475 550Q488 621 479 689M421 576Q450 614 437 676M546 575Q514 614 526 678',stroke='#8ca5c4',w=.8)
bodice+=p('M423 650Q482 681 542 654L551 677Q483 705 414 674Z','#8ca9c9','#7b8ead',.8)
bodice+=p('M420 663Q483 692 545 665',stroke='#f3f4ed',w=1)
bodice+=p('M482 677L495 689 482 702 470 689Z','#edf8ff','#b6a17c',1.2)
layers.append({'name':'06 Shoulders neck and silk bodice','children':[{'name':'Shoulders and bodice base','svg':svg(skin+bodice)},{'name':'Collarbone warmth and silk reflection','inherit_alpha':True,'strokes':[native(23,'#e4a9b2',[['M',407,489],['Q',424,480,448,487]],.25,'b) Airbrush Soft'),native(17,'#d0a0b1',[['M',527,478],['Q',550,485,562,501]],.22,'b) Airbrush Soft'),native(17,'#7e93ba',[['M',499,566],['Q',506,614,498,664]],.15,'b) Airbrush Soft'),native(16,'#ffffff',[['M',454,578],['Q',463,624,451,650]],.5,'b) Airbrush Soft')]}]})

skirt=p('M416 677Q480 712 547 678C566 702 585 734 618 753Q658 774 683 826L711 893Q689 916 659 907Q637 939 605 923Q576 947 547 929Q512 947 482 924Q446 939 419 916Q384 926 360 900L332 892C345 839 372 795 390 751Q407 708 416 677Z','url(#skirt)','#7e91b0',1.1)
foldpaths=[('M421 694Q397 780 355 883Q371 894 387 891L446 709Z','#9eb9d6'),('M458 709Q433 802 426 914Q446 925 463 917L477 717Z','#9bb3d3'),('M491 713Q501 812 493 919Q519 935 535 922L514 707Z','#9db7d6'),('M538 697Q565 753 597 791L638 914Q653 923 666 906Q635 806 597 765Z','#9eb7d4'),('M556 716Q588 792 605 920L575 927Q551 837 527 723Z','#c1d7eb')]
for d,col in foldpaths:skirt+=p(d,col)
for d in ['M431 719Q399 830 383 878','M468 725Q449 825 451 902','M507 728Q530 832 523 912','M545 733Q586 820 596 907','M588 786Q632 849 642 895']:
    skirt+=p(d,stroke='#f8fcff',w=1.5)
skirt+=p('M338 884Q359 908 385 899Q410 925 439 915Q465 936 491 923Q519 943 548 927Q577 943 606 920Q636 939 660 909Q688 921 704 892',stroke='#f5f9fd',w=6)
for i in range(49):
    x=347+i*7.3;y=895+28*math.sin(i/48*math.pi)+R.uniform(-1,1)
    skirt+=el(x,y,3,4,'none','#f4faff',.8)
for i in range(26):
    x=357+i*12.7;y=864+32*math.sin(i/25*math.pi)
    skirt+=p('M%.2f %.2fq-6 -8 0 -17q7 9 0 17m0 -9q-10 0 -11 -8q10 -1 11 8m0 0q10 0 11 -8q-10 -1 -11 8'%(x,y),stroke='#eff8ff',w=.65)
    skirt+=el(x,y+2,1.5,1.5,'#c2b394')
add('07 Seated skirt - independently shaped folds and embroidery',skirt)

arms=p('M377 535Q362 568 357 621Q350 653 368 680Q393 712 433 731L460 721Q415 688 404 660L403 615L410 559Z','url(#skin)','#b18f9b',.9)
arms+=p('M573 531Q590 557 599 605Q613 640 592 671Q567 703 509 735L483 720Q532 689 548 660Q559 643 547 615L543 556Z','url(#skin)','#b18f9b',.9)
arms+=p('M373 543Q392 560 413 549L411 570Q387 581 367 561ZM542 546Q564 557 581 539L588 559Q565 577 545 568Z','url(#silk)','#8196b4',.9)
arms+=p('M370 551Q392 569 412 558M545 556Q566 568 583 548',stroke='#f7fbff',w=1.4)
arms+=p('M427 717L438 739L455 730L444 709ZM502 714L513 736L529 726L517 706Z','#e8eff8','#8d9eb5',.7)
hands=p('M442 722Q455 716 472 725L505 739Q518 744 512 751Q507 755 497 750L475 740L499 755Q507 761 502 766Q498 770 490 764L467 749L486 766Q491 773 486 776Q482 779 475 772L450 753Q441 746 436 736Z','url(#skin)','#b68f9b',.75)
hands+=p('M509 720Q494 713 480 722L457 735Q449 740 455 745Q461 749 470 743L485 735L482 747Q483 753 487 750L495 736L515 734Z','url(#skin)','#b68f9b',.75)
hands+=p('M451 737L461 740M460 747L469 751M469 759L475 764',stroke='#c799a5',w=.55)
for x,y in [(448,724),(521,720)]:arms+=el(x,y,4,5,'#e9f5fb','#c2ae89',.75)
layers.append({'name':'08 Arms and gently folded fingers','children':[{'name':'Arms hands and cuffs','svg':svg(arms+hands)},{'name':'Skin reflected light','inherit_alpha':True,'strokes':[native(16,'#dba1ad',[['M',365,585],['Q',350,649,382,681]],.17,'b) Airbrush Soft'),native(15,'#d8a5b7',[['M',589,580],['Q',618,641,563,689]],.2,'b) Airbrush Soft'),native(9,'#fff9f1',[['M',375,580],['Q',362,632,381,659]],.46,'b) Airbrush Soft'),native(9,'#fff9f1',[['M',581,577],['Q',602,632,577,656]],.4,'b) Airbrush Soft')]}]})

face=p('M400 287C401 241 435 216 484 218C533 217 568 244 568 289L562 350C557 386 522 417 487 430C454 422 419 397 408 367C401 343 398 314 400 287Z','url(#face)','#a88a9b',.9)
face+=p('M408 308Q435 288 461 297Q502 286 558 306L558 315Q508 301 467 310Q438 300 408 320Z','#d6b1be',extra='opacity=".25"')
face+=el(432,367,29,15,'url(#blush)')+el(541,365,22,15,'url(#blush)')
face+=p('M487 348Q479 367 484 372Q487 373 490 372',stroke='#c69cab',w=.8)
face+=p('M480 367Q485 363 489 368',stroke='#fff7ee',w=2)
face+=p('M474 396Q486 392 498 396',stroke='#b48598',w=.8)+p('M479 401Q487 405 494 401',stroke='#f5c6cb',w=1.1)
for side in [-1,1]:
    for j in range(5):
        x=(429 if side<0 else 538)+j*2.2
        face+=p('M%.2f 366l-1 3'%x,stroke='#d89daa',w=.45,extra='opacity=".45"')
layers.append({'name':'09 Face plane and warm transparent skin','children':[{'name':'Face shape','svg':svg(face)},{'name':'Soft cheeks and bridge of nose','inherit_alpha':True,'strokes':[native(20,'#e59bab',[['M',423,364],['Q',435,368,446,366]],.22,'b) Airbrush Soft'),native(16,'#e4a5b1',[['M',528,365],['Q',539,367,548,364]],.2,'b) Airbrush Soft'),native(12,'#fffcf5',[['M',474,344],['Q',478,356,477,363]],.35,'b) Airbrush Soft')]}]})

# Eye parts are stored on independent layers. All iris detail is clipped to
# the eye almond AND to the iris, leaving sclera and lashes clean.
eye_shapes=[]
for i,(cx,cy,rx,ry,tilt) in enumerate([(444,340,33,22,-3),(524,338,30.5,21,2)]):
    upper='M%.2f %.2fC%.2f %.2f %.2f %.2f %.2f %.2f'%(cx-rx,cy,cx-rx*.5,cy-ry,cx+rx*.4,cy-ry*.95,cx+rx,cy-3)
    almond=upper+'Q%.2f %.2f %.2f %.2fQ%.2f %.2f %.2f %.2fZ'%(cx+rx*.72,cy+ry*.8,cx,cy+ry*.83,cx-rx*.75,cy+ry*.75,cx-rx,cy)
    clip='<defs><clipPath id="eye%d"><path d="%s"/></clipPath><clipPath id="ir%d"><ellipse cx="%.2f" cy="%.2f" rx="18" ry="23"/></clipPath></defs>'%(i,almond,i,cx+1,cy)
    whites=p(almond,'#fffdf7','#ac91a1',.65)
    whites+=p(upper+'L%.2f %.2fQ%.2f %.2f %.2f %.2fZ'%(cx+rx,cy+5,cx,cy-8,cx-rx,cy+5),'#c8bdd3',extra='opacity=".55"')
    add('10.%d Eye white and eyelid cast shadow'%i,clip+whites,svg_masks=[svg(p(almond,'#ffffff'))])
    iris=clip+'<g clip-path="url(#eye%d)">'%i+el(cx+1,cy,18,23,'url(#iris)','#4a567a',1)
    iris+='<g clip-path="url(#ir%d)">'%i
    for j in range(88):
        a=2*math.pi*(j+R.uniform(-.3,.3))/88
        r1=R.uniform(.36,.58);r2=R.uniform(.82,.99)
        x1=cx+1+18*r1*math.cos(a);y1=cy+23*r1*math.sin(a)
        x2=cx+1+18*r2*math.cos(a+.016);y2=cy+23*r2*math.sin(a+.016)
        col=R.choice(['#d5e4f4','#9dbde1','#6578b4','#b1a5d0','#4b5387'])
        iris+=p('M%.3f %.3fQ%.3f %.3f %.3f %.3f'%(x1,y1,(x1+x2)/2+R.uniform(-.6,.6),(y1+y2)/2,x2,y2),stroke=col,w=R.uniform(.18,.48),extra='opacity="%.2f"'%R.uniform(.3,.75))
        stats['iris_texture_paths']+=1
    iris+=el(cx+1,cy+1,5.6,10.7,'#303957')
    iris+=el(cx+1,cy+3,8.2,13.4,'none','#b8b4da',.45,extra='opacity=".6"')
    iris+=p('M%.2f %.2fQ%.2f %.2f %.2f %.2f'%(cx-11,cy+12,cx+1,cy+25,cx+13,cy+11),stroke='#c7eaf5',w=1.1,extra='opacity=".75"')
    iris+=el(cx+1,cy-7,20,17,'url(#irisShade)')
    for j in range(16):
        a=R.uniform(.18,math.pi-.18);rr=R.uniform(.56,.86)
        iris+=el(cx+1+18*rr*math.cos(a),cy+23*rr*math.sin(a),R.uniform(.25,.7),R.uniform(.3,.9),'#e6f6ff',extra='opacity=".8"')
    iris+='</g></g>'
    add('11.%d Iris fibers pupil collarette and limbal ring'%i,iris,svg_masks=[svg(p(almond,'#ffffff')),svg(el(cx+1,cy,18,23,'#ffffff'))])
    upper_lash=p(upper+'C%.2f %.2f %.2f %.2f %.2f %.2fZ'%(cx+rx*.4,cy-ry-3,cx-rx*.6,cy-ry-3,cx-rx-2,cy-1),'#51465e')
    outer=cx-rx if i==0 else cx+rx
    for j in range(5):
        xx=outer+(j*3.5 if i==0 else -j*3)
        sign=-1 if i==0 else 1
        upper_lash+=p('M%.2f %.2fQ%.2f %.2f %.2f %.2fQ%.2f %.2f %.2f %.2fZ'%(xx,cy-3-j*.9,xx+sign*6,cy-7-j*1.2,xx+sign*(9-j*.7),cy-13-j,xx+sign*2,cy-8,xx+3,cy-2-j),'#51465e')
    upper_lash+=p('M%.2f %.2fQ%.2f %.2f %.2f %.2f'%(cx-rx+9,cy-24,cx,cy-31,cx+rx-3,cy-22),stroke='#b695a9',w=.85)
    upper_lash+=p('M%.2f %.2fQ%.2f %.2f %.2f %.2f'%(cx-rx+5,cy+8,cx+3,cy+26,cx+rx-2,cy+7),stroke='#b28d9e',w=.65)
    upper_lash+=p('M%.2f %.2fQ%.2f %.2f %.2f %.2f'%(cx-rx+9,cy+14,cx+2,cy+25,cx+rx-6,cy+12),stroke='#fdf3f0',w=1.3)
    for j in range(5):
        xx=(cx-rx+7+j*4) if i==0 else (cx+rx-6-j*3.6)
        yy=cy+11+j*.65
        upper_lash+=p('M%.2f %.2fl%.2f 3.7'%(xx,yy,-1.4 if i==0 else 1.4),stroke='#796276',w=.55)
    upper_lash+=p('M%.2f %.2fQ%.2f %.2f %.2f %.2f'%(cx-rx+4,cy-39,cx,cy-46,cx+rx-3,cy-34),stroke='#8c9bb7',w=1.1)
    for j in range(14):
        xx=cx-rx+7+j*3.7
        yy=cy-39-4*math.sin(j/13*math.pi)+j*.3
        upper_lash+=p('M%.2f %.2fl3 -1.4'%(xx,yy),stroke='#96a3bd',w=.4)
    add('12.%d Sculpted lashes waterline and eyebrow hairs'%i,upper_lash)
    reflection=clip+'<g clip-path="url(#eye%d)">'%i
    reflection+=el(cx-5,cy-9,3.4,4.6,'#ffffff')+el(cx-9.5,cy-4,1.2,1.7,'#eaf8ff')
    reflection+=el(cx+10,cy+11,2.2,3,'#f8ffff')+el(cx+7,cy+15,1,1.3,'#f6fcff')
    reflection+=p('M%.2f %.2fl4 -3 5 3 -4 3Z'%(cx+3,cy-13),'#d7e9f7',extra='opacity=".75"')
    reflection+='</g>'+el(cx-rx+3,cy+1,1.3,1.1,'#efbcc5')
    add('13.%d Corneal reflections and tear duct'%i,reflection,svg_masks=[svg(p(almond,'#ffffff'))])

# A sculpted hair cap plus individually drawn bangs, not a helmet-shaped fill.
cap_shape='M392 316C366 271 390 222 424 202C451 179 506 180 545 207C584 233 598 279 577 327L561 277Q534 239 499 220Q463 207 429 244L404 310Z'
cap='<defs><clipPath id="scalp"><path d="'+cap_shape+'"/></clipPath></defs>'+p(cap_shape,'url(#hair2)','#7895b5',.85)+'<g clip-path="url(#scalp)">'
for j in range(45):
    t=j/44
    d='M%.2f %.2fC%.2f %.2f %.2f %.2f %.2f %.2f'%(486+t*10,195+t*1.8,434+t*65,194,388+t*161,227,387+t*192,298+t*8)
    cap+=p(d,stroke='#92b3ce' if j%4 else '#f4fbff',w=.35 if j%4 else .7,extra='opacity=".6"')
    stats['hair_fiber_paths']+=1
cap+='</g>'
add('14 Hair crown - root flow',cap,svg_masks=[svg(p(cap_shape,'#ffffff'))])
bang_specs=[([(481,206),(458,233),(417,270),(398,327)],23,50),
 ([(489,209),(483,242),(464,278),(437,314)],20,46),
 ([(496,213),(508,239),(528,270),(555,286)],24,50),
 ([(513,210),(552,237),(574,270),(573,322)],20,43)]
bang=''
for i,(q,w,n) in enumerate(bang_specs):bang+=hair_lock(q,w,n,'hair2',500+i)
add('15 Four overlapping bangs with individual fibers',bang)
front_specs=[([(399,277),(380,382),(416,475),(341,568)],25,58),
 ([(410,315),(399,416),(414,518),(374,604)],18,44),
 ([(570,283),(599,382),(538,493),(611,582)],28,61),
 ([(566,333),(554,416),(551,495),(580,550)],18,46)]
front=''
for i,(q,w,n) in enumerate(front_specs):front+=hair_lock(q,w,n,'hair' if i%2 else 'hair3',700+i)
add('16 Side locks falling in front of shoulders',front)

# Fine free strands interrupt the main silhouette sparingly.
fly=''
for i in range(35):
    side=-1 if i%2 else 1
    x=484+side*(90+R.uniform(0,17));y=R.uniform(245,330)
    ex=x+side*R.uniform(18,42);ey=y+R.uniform(90,235)
    fly+=p('M%.2f %.2fC%.2f %.2f %.2f %.2f %.2f %.2f'%(x,y,x+side*22,y+40,ex+side*15,ey-30,ex,ey),stroke=R.choice(['#9bbbd3','#bdd9e9','#86aac9']),w=R.uniform(.27,.55),extra='opacity=".65"')
    stats['hair_fiber_paths']+=1
add('17 Fine flyaways and broken edge strands',fly)

jewels=p('M436 209L424 171L455 188L482 154L511 185L541 171L533 208Q484 190 436 209Z','#c9e1ee','#7898b6',1)
jewels+=p('M441 200L434 180L456 195L482 165L509 193L532 181L527 200',stroke='#f4fbff',w=1.2)
jewels+=p('M482 171L492 185 482 200 472 185Z','#78add0','#b4a480',.8)+el(480,180,2,4,'#f5fdff')
jewels+=p('M411 267Q384 247 377 268Q387 285 411 271Q395 295 380 291Q377 276 411 267Z','#91b7d3','#6a8baa',.8)+el(407,270,5,6,'#eff9ff','#bdab87',.7)
for x,y in [(408,368),(563,365)]:
    jewels+=el(x,y,2,3,'#f9f0e1','#b29a7c',.65)+p('M%d %dV%d'%(x,y+4,y+26),stroke='#bba582',w=.8)+p('M%d %dq9 10 0 18q-8 -8 0 -18Z'%(x,y+25),'#aad3e8','#879db9',.65)
add('18 Diadem hair bow and blue glass earrings',jewels)

accent=[]
for side in [-1,1]:
    for j in range(12):
        x=399+j*.8 if side<0 else 570+j*.6;y=315+j*3
        accent.append(native(.55,R.choice(['#789bbd','#d8edf8']),[['M',x,y],['C',x+side*6,y+45,x-side*6,y+90,x+side*15,y+132]],.4))
for j in range(12):
    x=430+j*10
    accent.append(native(.75,'#b6cce1',[['M',x,740+j*2],['Q',x-4,797,x+13,846+j]],.42))
add('19 Native Gpen finishing on hair and folds',strokes=accent)

branch=''
branch_specs=[('M-20 187C65 212 122 282 226 241Q277 223 307 174',6),('M110 248Q99 180 126 124',2.4),('M183 254Q169 320 103 350',2.6),('M1020 168C896 166 866 276 787 283Q747 285 699 253',5.5),('M876 230Q882 304 934 331',2.5),('M834 265Q811 196 827 149',2.2),('M-20 1174C82 1152 129 1094 173 1067Q207 1051 232 995',5),('M112 1112Q75 1072 60 1016',2),('M1020 1218C906 1202 879 1111 813 1085Q779 1071 759 1020',5),('M886 1155Q918 1118 924 1072',2)]
for d,w in branch_specs:
    branch+=p(d,stroke='#97909e',w=w)+p(d,stroke='#d7bdc4',w=w*.24)
def flower(x,y,size,angle):
    b='<g transform="translate(%.1f %.1f) rotate(%s) scale(%.3f)">'%(x,y,angle,size)
    for a in range(0,360,72):
        b+='<g transform="rotate(%d)">'%a+p('M0 2C-9 -4 -16 -17 -10 -24Q-5 -29 0 -23Q5 -29 11 -24C18 -16 9 -3 0 2Z','#faeaf0','#c8aab7',.65)
        for j in [-1,0,1]:b+=p('M0 0Q%d -7 %d -15'%(j*2,j*5),stroke='#e0b7c2',w=.45)
        for j in [-1,1]:b+=el(j*3,-11,1,1,'#d7b09e')
        b+='</g>'
    return b+el(0,0,2.7,2.7,'#d9b2ba')+'</g>'
for x,y,size in [(26,202,.7),(86,225,.7),(132,249,.8),(195,250,.7),(245,229,.75),(281,205,.55),(124,142,.6),(111,185,.6),(146,313,.7),(105,350,.8),(971,175,.8),(917,192,.6),(882,225,.75),(846,260,.7),(788,282,.8),(743,275,.65),(826,164,.6),(825,220,.55),(918,323,.7),(58,1153,.7),(113,1114,.8),(173,1068,.7),(212,1034,.6),(63,1034,.7),(959,1207,.8),(908,1183,.7),(866,1129,.7),(811,1086,.8),(775,1058,.6),(919,1085,.6)]:
    branch+=flower(x,y,size,R.randrange(360))
add('20 Delicate sakura framing',branch)

petals=''
for j in range(32):
    x=R.choice([R.uniform(89,300),R.uniform(749,928)]);y=R.uniform(350,1360)
    petals+=p('M0 0Q-10 -11 -2 -19Q11 -12 0 0Z','#f4dce6','#d4b9c5',.5,extra='transform="translate(%.1f %.1f) rotate(%d)"'%(x,y,R.randrange(360)))
petals+='<text x="500" y="1421" font-family="Georgia" text-anchor="middle" font-size="10" letter-spacing="4" fill="#95a5bd">AZURE · SILKEN LIGHT</text>'
add('21 Falling petals and title',petals)
stars=[]
for x,y in [(323,398),(660,409),(285,840),(742,789),(337,1150),(741,1255)]:
    stars.append({'brush':'z) Stamp Sparkles','size':15,'color':'#ffffff','opacity':.55,'line':[x,y,x+1,y+1]})
add('22 Native scattered light accents',strokes=stars)

scene={'name':'Azure III - Silken Light - new blank canvas','width':2400,'height':3600,'scale':2.4,'stem':'azure-silken-light','blank_checkpoint':'00-new-blank-document.kra','layers':layers,
       'details':[{'file':'eyes-detail.png','rect':[402,296,164,76]},{'file':'face-hair-detail.png','rect':[355,176,266,285]},{'file':'hair-detail.png','rect':[229,753,174,332]}]}
(OUT/'new-canvas.scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'artwork-construction.json').write_text(json.dumps({**stats,'loads_previous_artwork':False,'canvas':[2400,3600],'top_level_layers':len(layers)},indent=2),encoding='utf-8')
print(json.dumps(stats))
