"""New full composition, prepared one stage at a time for Krita MCP.

No image imports, external generation, network or writes at import. Every stage
must be inspected before the next one. Paths and shading are authored anew.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/test-result/20261017/krita-garden-chair'
W,H,SCALE=1600,2200,2
HEAD='translate(411 255) rotate(-12)'
FACE='M-77 -63C-86 -25 -81 14 -63 43C-48 62 -18 80 4 86C28 80 58 62 73 37C88 13 86 -29 76 -67C62 -107 -61 -111 -77 -63Z'
CROWN='M-91 49C-119 12 -112 -75 -79 -118C-58 -146 -25 -159 -5 -144C32 -162 77 -137 98 -98C125 -50 116 6 89 62C96 18 77 -43 64 -66C32 -96 -34 -98 -60 -62C-81 -32 -76 13 -91 49Z'
NECK='M382 321C390 351 387 371 366 381C349 389 330 389 317 405C343 432 391 450 434 443C479 442 508 425 509 411C496 392 471 388 451 378C430 367 435 344 438 329Z'
BODICE='M317 420C337 444 344 460 352 493C355 512 362 530 361 550Q415 573 477 551C479 518 491 479 505 432C474 431 461 426 442 425C407 435 364 433 317 420Z'
SKIRT='M362 548C338 584 308 621 281 678C265 709 260 735 279 753Q313 775 351 768Q389 788 430 775Q465 794 505 778Q547 787 573 769Q600 771 613 748C608 698 582 651 548 620C514 587 495 559 477 550Z'
PETTICOAT='M280 710C270 733 263 748 277 765Q292 784 310 780Q330 794 350 784Q372 802 393 790Q416 805 437 792Q461 809 480 793Q507 806 526 791Q552 800 571 785Q595 788 608 769L609 730Z'
LEG_L='M443 704C460 718 493 746 509 779C522 805 516 835 507 863C494 908 472 949 469 985C466 1004 444 1003 441 986C442 944 455 895 456 856C454 831 436 797 418 765Z'
LEG_R='M481 727C505 740 553 761 567 792C583 821 573 854 565 883C556 925 544 965 544 997C542 1017 518 1018 510 999C507 957 520 916 522 877C522 844 496 814 479 788Z'
SLEEVE_L='M319 416C286 407 271 429 266 460C259 483 248 508 260 533C272 549 299 547 313 531C319 505 333 475 333 449Z'
SLEEVE_R='M491 415C512 404 534 430 546 456C563 481 580 510 572 535C561 554 530 554 514 535C504 514 495 482 481 458Z'
ARM_L='M273 521C268 504 276 470 282 443L295 389L316 386C321 418 307 452 305 479C302 505 302 530 288 541C280 545 274 537 273 521Z'
ARM_R='M544 530C547 552 534 590 517 620L511 636L530 645C555 620 573 576 570 545L567 527Z'
HAND_L='M295 390C295 374 291 364 298 352L311 333C314 329 318 332 316 338L308 355L317 348L329 319C332 312 338 314 336 321L328 347L339 334C342 331 346 334 343 339L330 361C328 369 323 378 316 388Z'
HAND_R='M515 620C504 620 499 630 489 634L468 640C462 641 461 645 467 647L490 646L469 651C464 653 465 657 471 657L494 653L478 660C473 662 475 666 481 665L504 656L490 666C487 669 490 672 495 670L520 654C530 648 533 639 528 632Z'
SHOE_L='M442 979C440 993 430 1001 426 1014C421 1025 426 1038 441 1040C460 1041 483 1031 491 1020C495 1013 484 997 475 990L472 978Q457 987 442 979Z'
SHOE_R='M511 990C506 1006 502 1023 513 1037C524 1050 556 1056 571 1048C583 1038 568 1023 554 1016L547 993Q530 1003 511 990Z'
REAR=[
 ('rear left outer','M340 135C277 156 271 246 279 342C260 425 254 495 234 573C206 659 181 729 208 797C218 827 247 848 245 878C275 841 236 816 240 779C239 734 277 680 290 627C319 539 327 445 333 365C339 282 349 188 340 135Z'),
 ('rear right outer','M471 140C552 148 560 250 558 336C551 418 591 485 614 557C644 640 661 704 635 765C611 814 652 840 663 867C612 850 590 824 607 776C634 700 572 634 555 574C527 492 491 431 503 340C507 256 481 178 471 140Z'),
 ('rear left inner','M343 237C309 339 319 444 301 530C290 596 265 648 279 705C295 758 332 776 312 827C296 868 288 900 320 929C304 879 354 853 349 811C349 770 321 743 321 700C309 647 349 575 354 511C361 429 362 318 343 237Z'),
 ('rear right inner','M489 226C522 305 502 392 531 457C555 512 560 555 545 600C529 648 550 684 579 730C613 783 566 831 578 870C583 901 609 916 602 950C642 912 612 880 610 852C601 816 640 780 615 733C585 670 572 635 584 589C597 528 561 469 553 416C539 345 555 276 489 226Z')]
FRINGES=[
 ('temple left','M-68 -126C-96 -91 -108 -12 -95 39C-93 65 -69 76 -55 71C-81 68 -73 42 -69 18C-68 -20 -49 -77 -68 -126Z'),
 ('bang left','M-66 -126C-67 -88 -77 -51 -70 -13C-67 7 -48 12 -34 2C-58 4 -57 -10 -50 -35C-44 -69 -40 -104 -40 -136Z'),
 ('bang middle','M-40 -136C-42 -87 -48 -39 -24 -14C-12 0 7 3 18 -10C-6 -9 -9 -26 -10 -45C-11 -90 -5 -120 -10 -143Z'),
 ('bang right small','M-10 -143C3 -105 12 -59 40 -36C53 -24 73 -30 77 -44C55 -36 38 -59 30 -88C21 -118 6 -139 -10 -143Z'),
 ('bang right sweep','M-5 -145C50 -147 86 -108 98 -61C113 -29 100 -2 77 10C91 -7 80 -29 68 -40C33 -64 13 -108 -5 -145Z')]
FRONT=[
 ('front left curl','M335 291C311 374 325 433 298 494C267 559 314 583 304 626C293 672 265 694 286 723C257 703 267 680 275 659C294 611 272 586 277 550C282 489 308 454 308 399C307 345 317 316 335 291Z'),
 ('front right curl','M491 291C484 346 477 390 499 433C516 465 544 484 532 516C523 544 478 554 488 587C495 608 522 616 514 642C550 617 524 598 511 582C495 563 546 554 555 522C568 484 530 456 518 429C496 380 520 330 491 291Z')]


def p(d,fill='none',stroke='none',w=1,op=1):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'


def ell(x,y,rx,ry,fill,op=1):
    return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{op}"/>'


def svg(body):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="2200" viewBox="0 0 800 1100">'+body+'</svg>'


def group(body,transform):return f'<g transform="{transform}">{body}</g>'
def head(body):return group(body,HEAD)


def eye_clip_ellipse(cx,cy,rx,ry,ex,ey,fill):
    """Convex polygon intersection avoids the local Qt SVG clipPath limitation."""
    def quad(a,b,c,t):return ((1-t)**2*a[0]+2*(1-t)*t*b[0]+t*t*c[0],(1-t)**2*a[1]+2*(1-t)*t*b[1]+t*t*c[1])
    clip=[quad((ex-25,ey),(ex-2,ey-22),(ex+25,ey-1),j/48) for j in range(49)]
    clip += [quad((ex+25,ey-1),(ex+2,ey+22),(ex-25,ey),j/48) for j in range(1,49)]
    poly=[(cx+rx*math.cos(j*math.tau/128),cy+ry*math.sin(j*math.tau/128)) for j in range(128)]
    sign=1 if sum(clip[i][0]*clip[(i+1)%len(clip)][1]-clip[(i+1)%len(clip)][0]*clip[i][1] for i in range(len(clip)))>0 else -1
    for i,a in enumerate(clip):
        b=clip[(i+1)%len(clip)]
        def side(v):return sign*((b[0]-a[0])*(v[1]-a[1])-(b[1]-a[1])*(v[0]-a[0]))
        old=poly;poly=[]
        if not old:break
        for j,v in enumerate(old):
            prev=old[j-1];sv,sp=side(v),side(prev)
            if (sv>=0)!=(sp>=0):
                t=sp/(sp-sv);poly.append((prev[0]+t*(v[0]-prev[0]),prev[1]+t*(v[1]-prev[1])))
            if sv>=0:poly.append(v)
    return p('M'+'L'.join(f'{x:.3f} {y:.3f}' for x,y in poly)+'Z',fill) if poly else ''


def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def flower(x,y,r,color='#fff5e2',angle=0):
    b=''
    for j in range(5):
        b+=group(ell(0,-r*.48,r*.48,r*.68,color,.9)+p(f'M0 {-r*.9}Q{-r*.17} {-r*.4} 0 0',stroke='#dbbdaf',w=.4,op=.4),f'rotate({angle+j*72})')
    b+=ell(0,0,r*.22,r*.22,'#d9b26b')
    for j in range(9):
        a=j*2.4;b+=ell(math.cos(a)*r*.19,math.sin(a)*r*.19,r*.03,r*.03,'#fff6c2')
    return group(b,f'translate({x} {y})')


def leaf(x,y,size,angle,color):
    b=p('M0 0C-30 -20 -23 -55 0 -77C29 -57 32 -20 0 0Z',color)
    b+=p('M0 -3Q4 -40 0 -69',stroke='#f3efc9',w=.8,op=.3)
    return group(b,f'translate({x} {y}) rotate({angle}) scale({size})')


def garden():
    rng=random.Random(261017)
    b=''
    # Far foliage: quiet, translucent shapes leave the face unobstructed.
    for j in range(36):
        x=rng.choice([rng.uniform(-30,215),rng.uniform(640,840)])
        y=rng.uniform(70,850)
        b+=ell(x,y,rng.uniform(26,70),rng.uniform(32,88),rng.choice(['#9bb7a2','#93b9af','#c5d09f']),rng.uniform(.08,.2))
    b+=p('M28 -30C120 90 184 89 157 220C133 302 180 363 145 437M767 -15C695 77 695 159 718 265C731 332 685 365 675 431',stroke='#7b9e83',w=3,op=.55)
    specs=[(66,59,.9,-80),(104,89,.85,37),(148,109,.7,-35),(159,165,.65,65),(154,217,.6,-64),(156,281,.52,72),(159,339,.5,-65),(716,77,.8,-66),(704,142,.7,48),(709,207,.6,-71),(718,276,.6,60),(704,329,.55,-60)]
    for j,(x,y,s,a) in enumerate(specs):b+=leaf(x,y,s,a,['#7d9f84','#9fb596','#a9bf94','#91afa0'][j%4])
    for x,y,r,c,a in [(105,281,18,'#f3dccf',12),(167,378,14,'#fff4d5',28),(688,228,19,'#fbefd3',-14),(706,388,14,'#f5d9cc',20)]:
        b+=flower(x,y,r,c,a)
    for j in range(28):
        x=rng.uniform(30,780);y=rng.uniform(35,950)
        b+=ell(x,y,rng.uniform(4,15),rng.uniform(5,18),'url(#white)',rng.uniform(.1,.5))
    return b


def layer(name,body,mask=None,erase=None,**kw):
    n={'name':name,'svg':svg(DEFS+body),**kw}
    if mask:n['svg_masks']=[svg(mask)]
    if erase:n['svg_erase_masks']=[svg(e) for e in erase]
    return n


DEFS='''<defs>
<linearGradient id="sky" x1="0%" y1="100%" x2="90%" y2="0%"><stop stop-color="#b7d6d3"/><stop offset=".52" stop-color="#dde9d7"/><stop offset="1" stop-color="#fff8d8"/></linearGradient>
<linearGradient id="floor" x1="0%" y1="0%" x2="80%" y2="100%"><stop stop-color="#e5e7d6"/><stop offset="1" stop-color="#bccdc7"/></linearGradient>
<linearGradient id="skin" x1="0%" y1="0%" x2="100%" y2="40%"><stop stop-color="#f7d9d3"/><stop offset=".46" stop-color="#fff0e4"/><stop offset="1" stop-color="#f9d3d4"/></linearGradient>
<linearGradient id="skinleg" x1="0%" y1="0%" x2="100%" y2="10%"><stop stop-color="#daaec0"/><stop offset=".24" stop-color="#f7d4d6"/><stop offset=".53" stop-color="#ffece3"/><stop offset=".8" stop-color="#fff2e8"/><stop offset="1" stop-color="#edc5ce"/></linearGradient>
<linearGradient id="hair" x1="5%" y1="30%" x2="100%" y2="50%"><stop stop-color="#8eacc5"/><stop offset=".17" stop-color="#b8cbe1"/><stop offset=".42" stop-color="#e0ecf6"/><stop offset=".65" stop-color="#f6f5ff"/><stop offset=".83" stop-color="#bfd1e8"/><stop offset="1" stop-color="#9daecf"/></linearGradient>
<linearGradient id="fringe" x1="0%" y1="0%" x2="100%" y2="35%"><stop stop-color="#a2b5d5"/><stop offset=".25" stop-color="#d8e5f4"/><stop offset=".52" stop-color="#f5f9ff"/><stop offset=".77" stop-color="#dae6f4"/><stop offset="1" stop-color="#b7b9d9"/></linearGradient>
<linearGradient id="crown" x1="0%" y1="100%" x2="85%" y2="0%"><stop stop-color="#a3b5d1"/><stop offset=".4" stop-color="#dce8f3"/><stop offset=".8" stop-color="#f6fbff"/><stop offset="1" stop-color="#fffbed"/></linearGradient>
<linearGradient id="cloth" x1="0%" y1="100%" x2="90%" y2="0%"><stop stop-color="#a8bfd6"/><stop offset=".45" stop-color="#e2eaf5"/><stop offset=".8" stop-color="#fcfaff"/><stop offset="1" stop-color="#fff9f0"/></linearGradient>
<linearGradient id="sleeve" x1="0%" y1="100%" x2="100%" y2="10%"><stop stop-color="#9bb4d2"/><stop offset=".6" stop-color="#e5ecf7"/><stop offset="1" stop-color="#fdf7ff"/></linearGradient>
<linearGradient id="ribbon" x1="0%" y1="0%" x2="100%" y2="70%"><stop stop-color="#94b7d6"/><stop offset=".45" stop-color="#d4e6f4"/><stop offset="1" stop-color="#7299bd"/></linearGradient>
<linearGradient id="iris" x1="0%" y1="0%" x2="15%" y2="100%"><stop stop-color="#403c6c"/><stop offset=".38" stop-color="#626ba6"/><stop offset=".72" stop-color="#a89fcf"/><stop offset="1" stop-color="#e7b4d3"/></linearGradient>
<radialGradient id="rose"><stop stop-color="#ef8da8" stop-opacity=".54"/><stop offset=".48" stop-color="#f6a4b8" stop-opacity=".29"/><stop offset="1" stop-color="#f8b4c1" stop-opacity="0"/></radialGradient>
<radialGradient id="warm"><stop stop-color="#fff7dc" stop-opacity=".75"/><stop offset="1" stop-color="#fff7dc" stop-opacity="0"/></radialGradient>
<radialGradient id="cool"><stop stop-color="#897ea4" stop-opacity=".25"/><stop offset="1" stop-color="#9d93b0" stop-opacity="0"/></radialGradient>
<radialGradient id="white"><stop stop-color="#fffdf3" stop-opacity=".9"/><stop offset="1" stop-color="#fffffa" stop-opacity="0"/></radialGradient>
</defs>'''


def plan():
    write('elements.json',{'title':'A garden pause / 花园小憩','canvas':[W,H],'logical_canvas':[800,1100],'scale':2,'reference':'user-supplied 149567189; principles only, no imported image pixels','head':{'origin':[411,255],'roll_degrees':-12,'yaw':'near frontal, stylized; no calibrated 3D claim','pitch':'near level','eye_centres_local':[[-36,0],[36,-1]],'nose_local':[0,32],'mouth_local':[4,55],'chin_local':[4,86],'expression':'new mild smile design, old expression study remains separate'},'gesture':{'neck':[411,361],'shoulders':[[318,408],[493,419]],'pelvis':[415,621],'elbows':[[279,535],[558,537]],'wrists':[[306,388],[520,633]],'knees':[[448,812],[521,817]],'ankles':[[425,986],[510,999]]},'chair':{'projection':'one point; front width horizontal, vertical axis parallel','vanishing_point':[420,340],'seat_front':[[245,700],[605,700]],'seat_back':[[287,613.6],[560.6,613.6]],'depth_scale':.76,'arm_front_y':570,'arm_back_y':514.8,'leg_front_bottom_y':1000,'leg_back_bottom_y':841.6},'hair':{'part':'off-centre swept bangs','sections':['rear outer masses','rear inner masses','crown','five uneven bangs/temple locks','two front curls'],'principles':['broad light before strands','narrow occlusion seam distinct from wide turn','restrained warm reflected edge','no evenly spaced full-length comb lines']},'light':{'main':'broad upper viewer right','fill':'cool garden ambient','face':'soft rose under eyes; not structural shadow'},'tools':{'render':'authored SVG paths/masks rasterized by Krita MCP','native_details':'c4) Thin Brush Regular - deevad 25.01','new_downloads':False,'external_image_generation':False},'workflow':['sketch','occluded contours','flats','large shading','details','visual revisions','whole/crops/mirror/gray review']})


def chair(flat=True):
    color='#e7e7d5' if flat else '#fff8e5'
    # Aligned front/back seat, armrest and leg endpoints share one VP.
    body=p('M287 614V351C286 314 323 290 422 290C521 290 562 315 561 351V614',stroke='#aaa998',w=16)
    body+=p('M287 614V351C286 314 323 290 422 290C521 290 562 315 561 351V614',stroke=color,w=11)
    for x in [317,354,391,428,465,502,539]:
        body+=p(f'M{x} 345V615',stroke='#f7f1dc',w=5)
    body+=p('M245 700V1000M605 700V1000M287 614V841.6M561 614V841.6',stroke='#a6afaa',w=10)
    body+=p('M245 700V1000M605 700V1000M287 614V841.6M561 614V841.6',stroke=color,w=6)
    body+=p('M245 700L287 613.6H560.6L605 700Z','#c1d7d7','#a5bfc5',1)
    body+=p('M245 700Q420 725 605 700L605 714Q420 738 245 714Z','#b1ccd1','#a1b5c0',1)
    body+=p('M245 700V570M605 700V570M287 614V515M561 614V515',stroke=color,w=8)
    for a,b in [((245,570),(287,514.8)),((605,570),(560.6,514.8))]:
        body+=p(f'M{a[0]} {a[1]}L{b[0]} {b[1]}',stroke='#b7baaa',w=13)+p(f'M{a[0]} {a[1]}L{b[0]} {b[1]}',stroke=color,w=9)
    return body


def sketch(reviewed=False):
    plan()
    body='<rect width="800" height="1100" fill="#f8f5ef"/>'
    body+=p('M60 340H740M245 700L420 340L605 700M287 613.6H560.6M245 700H605M245 1000H605M287 841.6H560.6',stroke='#b3c9c6',w=.9)
    body+=chair()
    body+=head(ell(0,-35,91,110,'none')+p(FACE,stroke='#8f7a84',w=1.5)+p('M-5 -115L4 86M-79 0L80 0M-48 32H47M-25 55H29',stroke='#a993a0',w=.9))
    body+=head(ell(0,-47,94,108,'none'))
    body+=p('M411 342C399 396 382 423 409 483C442 542 404 569 415 621',stroke='#a17b8e',w=2)
    body+=p('M318 408Q403 380 493 419M361 550Q410 584 477 551M318 408L279 535L306 388M493 419L558 537L520 633M391 622L497 806L455 986M447 628L555 819L528 999',stroke='#91858f',w=1.5)
    body+=p('M323 415C307 463 335 526 373 542Q421 566 472 538C513 495 522 446 490 418Q415 384 323 415ZM361 550Q415 535 477 551L484 632Q420 661 373 635Z',stroke='#b7a5b1',w=1)
    for x,y in [(318,408),(493,419),(279,535),(558,537),(306,388),(520,633),(497,806),(555,819),(455,986),(528,999)]:
        body+=f'<circle cx="{x}" cy="{y}" r="9" fill="#f8f5ef" stroke="#aa94a1" stroke-width="1"/>'
    for d in [HAND_L,HAND_R,LEG_L,LEG_R,SKIRT]:body+=p(d,stroke='#a18c9a',w=.8)
    for _,d in REAR+FRONT:body+=p(d,stroke='#a7b3c4',w=.8)
    body+=head(p(CROWN,stroke='#96aabd',w=1)+''.join(p(d,stroke='#a4b7c5',w=.7) for _,d in FRINGES))
    body+='<text x="28" y="40" font-size="11" fill="#807782">01 / gesture, contact, local head axes and chair perspective</text>'
    scene={'name':'A garden pause - from blank','stem':'01-sketch','width':W,'height':H,'scale':SCALE,'blank_checkpoint':'00-blank.kra','layers':[layer('01 Structure and perspective',body)]}
    if reviewed:
        scene.update(stem='01b-sketch-reviewed',expected_document=str(OUT/'sketch/01-sketch.kra'))
        scene.pop('blank_checkpoint')
    write('01b-sketch-reviewed.json' if reviewed else '01-sketch.json',scene)


def contour():
    body=chair()
    for _,d in REAR:body+=p(d,'#fffdf7','#8996a8',.8)
    for d in [LEG_R,LEG_L,SHOE_R,SHOE_L,PETTICOAT,NECK,SLEEVE_L,SLEEVE_R,ARM_R,BODICE,SKIRT,ARM_L]:body+=p(d,'#fffdf7','#ab9aa5',.8)
    body+=head(p(CROWN,'#fffdf7','#94a3b8',.8)+p(FACE,'#fffdf7','#b7949d',.9))
    body+=head(''.join(p(d,'#fffdf7','#92a2b8',.8) for _,d in FRINGES))
    for _,d in FRONT:body+=p(d,'#fffdf7','#94a4bc',.7)
    body+=p(HAND_L,'#fffdf7','#b698a5',.7)+p(HAND_R,'#fffdf7','#b698a5',.7)
    body+=head(p('M-62 0C-48 -13 -28 -14 -13 0M13 -1C29 -15 48 -14 62 -2M0 32L2 33M-9 56Q5 59 19 53',stroke='#8f7e94',w=1))
    n=layer('02 Occluded clean contour',body,visibility={'01 Structure and perspective':False})
    stage('02-contour',[n],OUT/'sketch-reviewed/01b-sketch-reviewed.kra')


def base_layers(shaded=False):
    skin='url(#skin)' if shaded else '#fce4df'
    hairfill='url(#hair)' if shaded else '#c3d8eb'
    cloth='url(#cloth)' if shaded else '#e8edf8'
    bg='<rect width="800" height="1100" fill="url(#sky)"/>' if shaded else '<rect width="800" height="1100" fill="#dfe9db"/>'
    bg+=p('M0 835Q405 786 800 821V1100H0Z','url(#floor)' if shaded else '#d7dfd3')
    bg+=ell(445,1019,246,34,'#789c9b',.18)
    bg+=p('M68 836V252C69 45 725 43 731 252V835M84 831V256C84 70 713 65 715 256V830',stroke='#fff9e3',w=8,op=.72)
    bg+=p('M89 435H711M103 728H702M219 128V832M611 123V824',stroke='#fff8e2',w=5,op=.45)
    if shaded:bg+=garden()+ell(645,103,245,270,'url(#warm)')
    n=[layer('03 Garden architecture',bg,visibility={'02 Occluded clean contour':False}),layer('04 Chair and seat',chair(not shaded))]
    for i,(name,d) in enumerate(REAR):n.append(layer('05 '+name,p(d,hairfill,'#91abc0',.55)))
    n.append(layer('06 Legs and shoes',p(LEG_R,'url(#skinleg)' if shaded else skin,'#c39eae',.7)+p(LEG_L,'url(#skinleg)' if shaded else skin,'#c39eae',.7)+p(SHOE_R,'url(#ribbon)' if shaded else '#9db9d6','#849db6',.7)+p(SHOE_L,'url(#ribbon)' if shaded else '#9db9d6','#849db6',.7)))
    n.append(layer('07 Petticoat',p(PETTICOAT,'#f7f4ff','#bdc9db',.7)))
    n.append(layer('08 Neck shoulders and sleeves',p(NECK,skin,'#c8a5b2',.7)+p(SLEEVE_L,'url(#sleeve)' if shaded else cloth,'#a9b9ce',.7)+p(SLEEVE_R,'url(#sleeve)' if shaded else cloth,'#a9b9ce',.7)+p(ARM_R,skin,'#bf9ba9',.7)))
    n.append(layer('09 Dress bodice and skirt',p(BODICE,cloth,'#a9b9ce',.7)+p(SKIRT,cloth,'#a9b9ce',.7)))
    n.append(layer('10 Raised forearm',p(ARM_L,skin,'#c59ca9',.7)))
    n.append(layer('11 Crown and face',head(p(CROWN,'url(#crown)' if shaded else hairfill,'#a3b2c8',.65)+p(FACE,skin,'#c59eac',.8))))
    n.append(layer('12 Front fringe',head(''.join(p(d,'url(#fringe)' if shaded else hairfill,'#9eacc4',.65) for _,d in FRINGES))))
    n.append(layer('13 Front curls',''.join(p(d,'url(#hair)' if shaded else hairfill,'#9cafc6',.65) for _,d in FRONT)))
    n.append(layer('14 Hands',p(HAND_L,skin,'#b78f9f',.65)+p(HAND_R,skin,'#b78f9f',.65)))
    eyes=''
    for cx,cy in [(-36,0),(36,-1)]:
        eyes+=p(f'M{cx-25} {cy}Q{cx-2} {cy-22} {cx+25} {cy-1}Q{cx+2} {cy+22} {cx-25} {cy}Z','#fff8f7','#bfa4b8',.65)
        eyes+=eye_clip_ellipse(cx,cy+2,13,17,cx,cy,'url(#iris)' if shaded else '#939ac2')+eye_clip_ellipse(cx,cy-1,5.2,9.5,cx,cy,'#53516f')
        eyes+=p(f'M{cx-25} {cy}Q{cx-2} {cy-22} {cx+25} {cy-1}',stroke='#65556c',w=2.3)
        eyes+=eye_clip_ellipse(cx-4.5,cy-7,3.2,4.2,cx,cy,'#fffcf4')
    eyes+=p('M-62 -29Q-39 -35 -17 -29M17 -30Q41 -38 61 -31',stroke='#9c8aa6',w=1.1)
    eyes+=p('M0 31L2 33',stroke='#c991a0',w=.8)+p('M-10 55Q6 58 20 51',stroke='#bc7e96',w=1.1)
    n.append(layer('15 Facial anchors',head(eyes),mask=head(p(FACE,'#fff')),erase=[head(''.join(p(d,'#fff') for _,d in FRINGES))]))
    return n


def stage(stem,layers,expected):
    write(stem+'.json',{'name':'A garden pause','stem':stem,'width':W,'height':H,'scale':SCALE,'expected_document':str(expected),'layers':layers,'details':[{'file':stem+'-head.png','rect':[277,80,285,285]},{'file':stem+'-hands.png','rect':[255,300,340,390]}]})


def flats():stage('03-flats',base_layers(),OUT/'contour/02-contour.kra')


def eye_repair():
    stage('03b-eye-crop-reviewed',[n for n in base_layers() if n['name']=='15 Facial anchors'],OUT/'flats/03-flats.kra')


def shading():
    layers=base_layers(True)
    # Revisions replace bases. Additional shaded layers go in a later append.
    for n in layers:n.pop('visibility',None)
    stage('04-shaded-bases',layers,OUT/'flats-reviewed/03b-eye-crop-reviewed.kra')


def foreground_mask():
    return ''.join(p(d,'#fff') for d in [NECK,LEG_L,LEG_R,SHOE_L,SHOE_R,PETTICOAT,SLEEVE_L,SLEEVE_R,ARM_L,ARM_R,BODICE,SKIRT,HAND_L,HAND_R])+head(p(CROWN,'#fff')+p(FACE,'#fff')+''.join(p(d,'#fff') for _,d in FRINGES))+''.join(p(d,'#fff') for _,d in FRONT)


def planes_layers():
    n=[]
    hairmask=''.join(p(d,'#fff') for _,d in REAR)
    body=p('M323 195C287 278 303 391 276 487C261 556 215 676 216 744C227 681 270 637 288 572C313 476 319 351 323 195Z','#8c9fbd',op=.36)
    body+=p('M318 381C308 478 317 518 286 613C281 641 278 673 287 704C296 633 328 571 334 503Z','#8d95b7',op=.3)
    body+=p('M532 270C517 366 547 426 569 483C606 571 626 646 621 700C645 635 605 549 585 494C556 416 542 364 532 270Z','#919abb',op=.38)
    body+=p('M568 688C588 738 615 767 595 812C592 824 587 850 600 876C601 839 626 812 626 782C627 742 598 721 568 688Z','#879abc',op=.35)
    body+=p('M314 215C287 295 303 381 280 460C282 402 272 346 281 283C287 250 298 228 314 215Z','#f0f4ff',op=.65)
    body+=p('M250 609C222 678 210 735 225 776C232 790 238 805 243 817C221 749 267 685 281 634Z','#f2efff',op=.55)
    body+=p('M546 404C561 462 598 528 607 580C614 617 639 674 631 714C649 677 623 608 619 585C601 502 557 441 546 404Z','#f0f1ff',op=.63)
    body+=p('M300 773C336 818 305 860 313 899C315 861 349 839 336 808Z','#efdef2',op=.65)
    body+=p('M233 787C224 819 257 833 252 859M625 775C602 815 641 835 653 852',stroke='#edc8e0',w=2.6,op=.65)
    n.append(layer('16 Rear hair broad planes',body,mask=hairmask,erase=[foreground_mask()]))
    # Each frontal region has its own mask, preventing shading jumping strands.
    shaded=[
        ('temple left',p('M-89 -78C-108 -15 -101 39 -76 61C-93 21 -71 -28 -73 -57Z','#8c9fbd',op=.35)+p('M-88 -83C-90 -29 -90 8 -82 33C-78 -12 -71 -46 -65 -74Z','#fffaff',op=.5)),
        ('bang left',p('M-65 -99C-67 -57 -80 -12 -56 -1L-39 1C-61 -4 -56 -36 -50 -64Z','#a0a4c7',op=.4)+p('M-52 -116C-57 -73 -68 -39 -62 -20C-65 -55 -42 -95 -52 -116Z','#fbfdff',op=.7)),
        ('bang middle',p('M-29 -122C-40 -81 -41 -38 -22 -19C-13 -9 -4 -7 4 -9C-20 -23 -21 -67 -14 -98Z','#9aa5c7',op=.3)+p('M-34 -113C-37 -67 -34 -45 -27 -33C-31 -69 -20 -101 -24 -119Z','#ffffff',op=.65)),
        ('bang right small',p('M-2 -116C10 -80 25 -46 49 -36C57 -32 66 -32 70 -36C36 -49 28 -81 17 -107Z','#a6a6c8',op=.45)),
        ('bang right sweep',p('M21 -131C61 -102 72 -62 86 -44C102 -18 83 -5 80 3C111 -10 108 -43 92 -66C76 -103 54 -123 21 -131Z','#9d9dbf',op=.38)+p('M13 -139C55 -128 84 -98 93 -63C87 -105 62 -136 26 -141Z','#fffcf4',op=.8))]
    for name,body in shaded:
        shape=next(d for k,d in FRINGES if k==name)
        n.append(layer('17 Plane '+name,head(body),mask=head(p(shape,'#fff'))))
    curlmask=''.join(p(d,'#fff') for _,d in FRONT)
    body=p('M322 338C301 412 319 443 292 505C278 542 289 569 294 590C279 547 286 521 302 490C326 441 317 386 322 338Z','#8d9ebb',op=.35)
    body+=p('M307 622C299 650 275 683 279 703C287 677 313 652 307 622Z','#979ac0',op=.4)
    body+=p('M497 345C482 392 501 436 525 465C547 491 542 512 529 527C551 521 565 491 541 466C513 435 499 390 507 352Z','#9ca5c4',op=.38)
    body+=p('M513 544C479 562 493 588 510 601C526 616 525 624 519 633C539 616 514 600 508 591C491 572 514 564 531 552Z','#a4a1c4',op=.35)
    body+=p('M315 398C321 444 302 480 292 519C287 539 290 552 291 562C290 524 312 490 320 460C326 437 323 413 315 398Z','#fff9ff',op=.8)
    body+=p('M499 301C485 348 486 384 495 410C486 362 509 333 499 301Z','#fffdf8',op=.8)
    body+=p('M302 619C298 650 276 675 278 689M498 562C486 576 509 596 515 603',stroke='#f3cfe1',w=2,op=.72)
    n.append(layer('18 Front curl planes',body,mask=curlmask,erase=[p(HAND_L,'#fff')+p(HAND_R,'#fff')]))
    body=p('M378 326C392 344 422 350 444 330L435 361Q414 372 389 354Z','#b58aab',op=.3)
    body+=ell(390,366,20,40,'url(#cool)')
    body+=p('M365 396Q386 394 402 410M432 408Q457 393 476 400',stroke='#b890a4',w=1.1,op=.45)
    body+=p('M405 373Q410 386 414 392',stroke='#b594ab',w=.8,op=.3)
    n.append(layer('19 Neck cast shadow and clavicles',body,mask=p(NECK,'#fff'),erase=[head(p(FACE,'#fff'))+''.join(p(d,'#fff') for _,d in FRONT)]))
    # Wide blush distinct from cast shadow, tied to the head's local axis.
    body=head(ell(-37,22,31,17,'url(#rose)')+ell(39,21,32,17,'url(#rose)')+ell(2,20,23,8,'url(#rose)',.15)+ell(-23,10,21,8,'url(#white)',.25))
    eyes=''.join(p(f'M{cx-25} {cy}Q{cx-2} {cy-22} {cx+25} {cy-1}Q{cx+2} {cy+22} {cx-25} {cy}Z','#fff') for cx,cy in [(-36,0),(36,-1)])
    n.append(layer('20 Soft blush and skin light',body,mask=head(p(FACE,'#fff')),erase=[head(''.join(p(d,'#fff') for _,d in FRINGES)+eyes)]))
    contact=head(group(''.join(p(d,'#a48aa8',op=.19) for _,d in FRINGES),'translate(0 3)'))
    n.append(layer('21 Fringe contact cast on skin',contact,mask=head(p(FACE,'#fff')),erase=[head(''.join(p(d,'#fff') for _,d in FRINGES)+eyes)]))
    folddefs='''<defs><linearGradient id="fold" x1="0%" y1="0%" x2="100%" y2="20%"><stop stop-color="#7c99bc" stop-opacity=".44"/><stop offset=".65" stop-color="#9dafd0" stop-opacity=".2"/><stop offset="1" stop-color="#bdcae0" stop-opacity="0"/></linearGradient></defs>'''
    body=folddefs
    folds=['M363 553C334 600 330 643 293 699C282 719 276 735 281 751C304 731 324 676 338 645C346 624 357 594 363 553Z',
        'M388 563C377 607 356 643 350 684C345 720 351 748 347 766C368 733 365 701 377 670C391 635 392 600 388 563Z',
        'M418 567C421 605 412 627 413 656C414 700 438 747 430 775C449 741 433 703 432 676C431 632 437 601 418 567Z',
        'M468 562C469 604 502 624 520 659C543 701 539 745 559 773C556 734 560 704 544 672C522 626 491 603 468 562Z',
        'M346 614C400 608 473 622 525 650C480 639 439 637 405 637C377 638 350 633 338 630Z',
        'M320 659C357 648 394 656 412 669C378 667 347 667 318 676Z',
        'M485 691C521 699 550 714 577 734C548 725 514 712 488 710Z']
    body+=''.join(p(d,'url(#fold)') for d in folds)
    body+=p('M363 579C344 623 336 662 311 706M387 601C373 642 368 677 367 708M438 689C439 720 453 751 445 773M496 612Q538 653 552 698M352 627Q427 615 491 643',stroke='#859fbd',w=.8,op=.42)
    body+=p('M370 586C350 630 347 658 321 706M399 595Q390 632 390 659M459 609Q474 643 483 662M450 701Q458 739 457 759',stroke='#fffafa',w=3,op=.62)
    n.append(layer('22 Dress tension and lap folds',body,mask=p(SKIRT,'#fff'),erase=[''.join(p(d,'#fff') for _,d in FRONT)+p(HAND_R,'#fff')]))
    body=p('M326 437C339 463 342 505 363 540L376 548C357 506 369 468 353 442Z','#8fa7c9',op=.34)
    body+=p('M479 445C466 480 468 519 466 546L480 548C486 511 502 469 502 443Z','#8fa7c9',op=.24)
    body+=p('M378 445Q393 486 387 539M443 443Q451 490 447 545',stroke='#a2b4ce',w=1,op=.5)
    body+=p('M385 448Q401 482 397 532',stroke='#fffcfa',w=4,op=.7)
    n.append(layer('23 Bodice fitted planes',body,mask=p(BODICE,'#fff'),erase=[''.join(p(d,'#fff') for _,d in FRONT)]))
    sleeves=p('M282 437Q293 456 282 484Q272 505 280 529M307 444Q324 469 304 514M511 437Q522 474 550 505M533 463Q561 493 560 525',stroke='#869fbf',w=1,op=.48)
    sleeves+=p('M281 449Q269 475 272 493M520 441Q531 464 537 483',stroke='#fffbff',w=4,op=.65)
    n.append(layer('24 Sleeve compression',sleeves,mask=p(SLEEVE_L,'#fff')+p(SLEEVE_R,'#fff'),erase=[p(ARM_L,'#fff')+''.join(p(d,'#fff') for _,d in FRONT)]))
    # Skin volume on knees/under hem, not a uniformly outlined leg.
    legs=p('M434 746C453 756 489 779 503 802C501 780 486 761 468 750Z','#bc8ba8',op=.35)
    legs+=p('M508 767C535 781 552 798 560 822C565 800 545 779 526 769Z','#bc8ba8',op=.32)
    legs+=ell(494,818,15,25,'url(#white)',.45)+ell(554,830,13,21,'url(#white)',.4)
    legs+=p('M467 874C464 907 448 947 449 971M535 883Q521 935 520 984',stroke='#fff3e8',w=4,op=.65)
    n.append(layer('25 Knee and shin planes',legs,mask=p(LEG_L,'#fff')+p(LEG_R,'#fff'),erase=[p(SKIRT,'#fff')+p(PETTICOAT,'#fff')+p(SHOE_L,'#fff')+p(SHOE_R,'#fff')]))
    return n


def garden_stage():stage('04b-garden',[base_layers(True)[0]],OUT/'shaded-bases/04-shaded-bases.kra')


def planes():stage('05-large-planes',planes_layers(),OUT/'garden/04b-garden.kra')


def soften_planes():
    changes=[]
    for n in planes_layers():
        name=n['name']
        if name.startswith(('16 ','17 ','18 ','22 ','23 ','25 ')):
            n['filters']=[{'name':'gaussian blur','settings':{'horizRadius':1.5,'vertRadius':1.5,'lockAspect':True}}]
            if name.startswith(('16 ','17 ','18 ')):n['opacity']=.65
            if name.startswith('22 '):n['opacity']=.68
            changes.append(n)
    stage('05b-planes-softened',changes,OUT/'planes/05-large-planes.kra')


def bezier(points,t):
    u=1-t
    return (u**3*points[0][0]+3*u*u*t*points[1][0]+3*u*t*t*points[2][0]+t**3*points[3][0],u**3*points[0][1]+3*u*u*t*points[1][1]+3*u*t*t*points[2][1]+t**3*points[3][1])


def strands(points,count,width,seed):
    rng=random.Random(seed);body=''
    for j in range(count):
        offset=(j/(count-1)-.5)*width
        start=rng.uniform(.02,.36);end=rng.uniform(.62,.98)
        coords=[]
        for k in range(36):
            t=start+(end-start)*k/35
            x,y=bezier(points,t)
            x+=offset*math.sin(math.pi*t)**.65
            coords.append((x,y))
        d='M'+'L'.join(f'{x:.2f} {y:.2f}' for x,y in coords)
        color='#fbfcff' if j%3==1 else '#879abb'
        body+=p(d,stroke=color,w=.35 if j%3 else .52,op=.4 if j%3==1 else .24)
    return body


def detail_layers():
    n=[];rng=random.Random(1706)
    # Iris reflections are clipped to both iris ellipse and eye aperture.
    body='';mask=''
    for cx,cy in [(-36,0),(36,-1)]:
        mask+=eye_clip_ellipse(cx,cy+2,13,17,cx,cy,'#fff')
        body+=ell(cx,cy+11,13,8,'#e9b7d3',.45)
        body+=p(f'M{cx-11} {cy+6}Q{cx} {cy+14} {cx+11} {cy+5}',stroke='#b7e6ef',w=1.1,op=.8)
        for j in range(30):
            a=rng.uniform(0,math.tau);rr=rng.uniform(5,12)
            x=cx+math.cos(a)*rr;y=cy+2+math.sin(a)*rr*.9
            color=rng.choice(['#b5e2f1','#ffe3c1','#a9b3e2','#f4cae5','#849cc7'])
            body+=ell(x,y,rng.uniform(.25,.8),rng.uniform(.35,1.1),color,rng.uniform(.45,.85))
        body+=ell(cx,cy-2,4.4,8.5,'#444566',.9)
        body+=ell(cx-4.4,cy-6.5,3.1,4.1,'#fffdf4',.98)+ell(cx+6.7,cy+.8,1.9,2.4,'#fff9e9',.9)
        body+=p(f'M{cx+6} {cy+4}l.7 1.8 1.7 .7-1.7 .6-.7 1.9-.7-1.9-1.7-.6 1.7-.7Z','#fefbff')
        body+=ell(cx-7,cy+7.5,1.2,1.6,'#d8eff6',.9)
    n.append(layer('26 Gemstone iris facets',head(body),mask=head(mask),erase=[head(''.join(p(d,'#fff') for _,d in FRINGES))]))
    body=''
    for side,cx,cy in [(-1,-36,0),(1,36,-1)]:
        body+=p(f'M{cx-25} {cy}Q{cx-2} {cy-23} {cx+25} {cy-1}L{cx+24} {cy+.5}Q{cx-2} {cy-19.5} {cx-25} {cy+1}Z','#60506b')
        ox=cx+side*23
        for j in range(3):
            body+=p(f'M{ox-side*j*1.5} {cy-j*1.4}Q{ox+side*(4-j*.8)} {cy-1-j*2} {ox+side*(7-j)} {cy-6-j*1.2}',stroke='#65516d',w=1.2-j*.15)
        body+=p(f'M{cx-20} {cy-9}Q{cx} {cy-21} {cx+19} {cy-10}',stroke='#c4a1b3',w=.65,op=.55)
        body+=p(f'M{cx-20} {cy+6}Q{cx-13} {cy+11} {cx-6} {cy+11}M{cx+7} {cy+10}Q{cx+17} {cy+8} {cx+21} {cy+4}',stroke='#cf9aaa',w=.8,op=.72)
    body+=ell(0,32,4,2,'url(#rose)',.3)+ell(-2,30,2,1,'#fff1e8',.4)
    n.append(layer('27 Tapered lashes and warm lower lids',head(body),mask=head(p(FACE,'#fff')),erase=[head(''.join(p(d,'#fff') for _,d in FRINGES))]))
    body=''
    flows=[([(324,170),(274,345),(315,414),(271,568)],8,25),([(268,566),(238,664),(192,730),(230,793)],8,23),([(317,555),(279,665),(341,737),(327,806)],7,18),([(537,228),(509,368),(595,478),(612,587)],8,20),([(600,566),(637,647),(649,718),(617,780)],7,17),([(577,672),(573,718),(631,765),(602,827)],7,17)]
    for j,(pts,count,width) in enumerate(flows):body+=strands(pts,count,width,80+j)
    n.append(layer('28 Rear hair interrupted strands',body,mask=''.join(p(d,'#fff') for _,d in REAR),erase=[foreground_mask()]))
    flowdefs=[('temple left',[(-74,-121),(-104,-57),(-101,38),(-64,68)],7,12),('bang left',[(-53,-126),(-65,-80),(-77,-15),(-41,1)],9,14),('bang middle',[(-26,-133),(-40,-78),(-36,-18),(10,-9)],10,17),('bang right small',[(-6,-136),(13,-95),(26,-44),(66,-36)],7,13),('bang right sweep',[(2,-140),(50,-133),(106,-42),(82,0)],10,19)]
    for j,(name,pts,count,width) in enumerate(flowdefs):
        shape=next(d for k,d in FRINGES if k==name)
        n.append(layer('29 Fine flow '+name,head(strands(pts,count,width,130+j)),mask=head(p(shape,'#fff'))))
    body=strands([(330,309),(300,383),(329,437),(293,513)],8,10,178)
    body+=strands([(293,521),(271,581),(320,615),(278,695)],7,9,179)
    body+=strands([(495,308),(479,376),(494,424),(534,475)],8,12,180)
    body+=strands([(539,498),(548,541),(479,560),(507,602)],8,10,181)
    n.append(layer('30 Face curls selective strands',body,mask=''.join(p(d,'#fff') for _,d in FRONT),erase=[p(HAND_L,'#fff')+p(HAND_R,'#fff')]))
    # Independent low-opacity native detail layer: paths stay on visible skin.
    marks=[]
    def world(x,y):
        a=math.radians(-12);return [411+x*math.cos(a)-y*math.sin(a),255+x*math.sin(a)+y*math.cos(a)]
    for side,count,start in [(-1,13,22),(1,17,23)]:
        for j in range(count):
            x=side*(start+j*1.15);y=19+(0,.5,1.3,.2,2)[j%5]
            a=world(x,y);b=world(x+.45,y+(3.3,4.8,3.9,5.1)[j%4])
            marks.append({'brush':'c4) Thin Brush Regular - deevad 25.01','color':'#c87c9b','size':.38,'opacity':.82,'path':[['M',*a],['L',*b]],'taper':False})
    n.append({'name':'31 Native deevad fine blush','opacity':.56,'strokes':marks})
    # Flounce, ribbons, delicate stitched ornament and actual shoe openings.
    body=p('M315 416Q331 427 347 426Q361 436 376 430Q394 441 410 433Q425 442 442 431Q458 436 474 430Q490 436 508 426L503 445Q489 454 474 447Q456 458 441 449Q424 461 407 450Q391 462 373 450Q356 457 341 445Q327 451 318 440Z','url(#cloth)','#aebed3',.6)
    body+=p('M320 438Q331 449 342 443Q357 455 373 448Q390 460 407 448Q423 459 441 447Q459 456 474 445Q491 453 503 442',stroke='#fffefd',w=1.6,op=.95)
    body+=p('M360 540Q417 558 478 543L481 559Q416 576 357 559Z','url(#ribbon)','#95b0ce',.55)
    body+=p('M415 555C386 537 372 542 377 556C382 571 398 565 415 555ZM424 556C446 534 468 542 458 556C451 570 435 562 424 556Z','url(#ribbon)','#87a5c3',.65)
    body+=p('M409 558Q393 580 369 601L385 604L390 616Q409 591 419 560M424 559Q433 585 453 606L455 592L469 598Q448 572 431 556','url(#ribbon)','#9bb6ce',.55)
    body+=ell(420,555,6,7,'#e2d2a7')+ell(420,553,3.5,4.5,'#f7f0d2')
    body+=p('M368 386Q412 434 461 388',stroke='#d1baa8',w=.75)
    body+=p('M414 412C404 405 402 415 416 425C430 415 426 405 419 412L416 416Z','none','#b4a9ba',1)
    body+=p('M434 1000Q455 1007 477 993L481 1000Q456 1016 431 1007Z','url(#ribbon)','#8ba8c5',.65)
    body+=p('M514 1011Q534 1019 553 1017L558 1025Q533 1028 511 1020Z','url(#ribbon)','#8ba8c5',.65)
    body+=p('M436 1000Q452 987 473 994L476 999Q457 1008 437 1004Z','url(#skin)')
    body+=p('M516 1012Q535 999 548 1014L551 1018Q533 1022 516 1017Z','url(#skin)')
    body+=p('M427 1027Q448 1041 481 1021M514 1037Q539 1052 565 1044',stroke='#f7f7f3',w=1.5,op=.9)
    body+=ell(477,1000,2.7,3.4,'#e4d4a4')+ell(552,1020,2.7,3.3,'#e4d4a4')
    n.append(layer('32 Flounce ribbon necklace and shoe finish',body,erase=[''.join(p(d,'#fff') for _,d in FRONT)+p(HAND_R,'#fff')+p(HAND_L,'#fff')]))
    body=p('M300 364Q309 357 316 363Q322 367 318 372M298 370Q309 365 316 372M300 375Q308 371 315 377M320 349Q330 350 328 358L320 368',stroke='#c392a4',w=.7,op=.65)
    body+=p('M501 637Q510 639 516 647M492 646L501 644M488 653L498 650M502 653L511 647',stroke='#c193a5',w=.7,op=.65)
    body+=p('M330 319Q334 317 334 322L331 329Q327 329 328 325Z','#d7dfed','#b4b9ce',.4)
    body+=p('M311 336Q314 335 314 339L312 345Q308 346 309 342Z','#d7dfed','#b4b9ce',.4)
    for x,y in [(470,642),(473,653),(483,661)]:body+=ell(x,y,2,1,'#f6e2e7',.75)
    n.append(layer('33 Palm finger groups and nails',body,mask=p(HAND_L,'#fff')+p(HAND_R,'#fff')))
    # Asymmetric embroidery follows the hem; no dense all-over pattern.
    body=p('M308 733C331 712 350 720 370 733C387 745 400 743 415 738M485 748Q529 735 565 750',stroke='#9cb4c8',w=.7,op=.7)
    for x,y,a in [(324,722,-45),(344,721,25),(364,732,-30),(388,742,42),(505,741,-35),(545,744,32)]:
        body+=leaf(x,y,.11,a,'#b1c7d2')
    for x,y,r in [(337,725,4),(377,739,4.8),(524,741,4.5),(554,749,3.5)]:body+=flower(x,y,r,'#fcf8f2')
    for j in range(33):
        x=283+j*9.5;y=756+12*math.sin((x-283)/330*math.pi)
        body+=ell(x,y,1,1.5,'#fffefa',.85)
    n.append(layer('34 Restrained hem embroidery',body,mask=p(SKIRT,'#fff'),erase=[''.join(p(d,'#fff') for _,d in FRONT)]))
    # Hair ornament and a few escape strands: strongest tiny details near face.
    ornament=head(flower(-89,-14,8,'#faf5e9',24)+flower(-85,-29,5,'#dcebf1',-8)+p('M-88 -15Q-110 9 -100 34Q-96 25 -88 -15M-87 -14Q-75 13 -86 41L-90 34',stroke='#7596b7',w=2))
    n.append(layer('35 Hair flower and loose silhouette strands',ornament+head(p('M-73 -119C-107 -155 -124 -130 -107 -114M-19 -144C-26 -175 34 -171 58 -145M93 -78C121 -78 133 -57 123 -23',stroke='#c3bedb',w=.65,op=.7))+p('M241 683C211 711 202 751 220 779M634 633C672 665 678 716 654 752',stroke='#b1bad4',w=.7,op=.65)))
    return n


def details():stage('06-detail',detail_layers(),OUT/'planes-softened/05b-planes-softened.kra')


def detail_repair():
    fold=next(n for n in planes_layers() if n['name'].startswith('22 '))
    fold.update(opacity=.53,filters=[{'name':'gaussian blur','settings':{'horizRadius':5,'vertRadius':5,'lockAspect':True}}])
    marks=next(n for n in detail_layers() if n['name'].startswith('31 '))
    marks['svg']=svg('')  # Clear before repainting native marks on revision.
    marks['opacity']=.6
    for mark in marks['strokes']:mark.update(size=.6,opacity=.95)
    stage('06b-detail-reviewed',[fold,marks],OUT/'details/06-detail.kra')


def finish():
    body=p('M245 700L287 613.6L303 631L282 700Z','#d1e2df','#9eb4b8',.8)
    body+=p('M245 700V713Q420 739 605 713V700',stroke='#abc4c5',w=4)
    for d in ['M245 710V574Q242 568 249 566L287 515','M605 710V574Q608 568 601 566L561 515']:
        body+=p(d,stroke='#9aa7a5',w=11)+p(d,stroke='#f6f0de',w=7)
    body+=p('M245 716V1000M605 716V1000',stroke='#a5b1aa',w=10)+p('M245 716V1000M605 716V1000',stroke='#f2eedb',w=6)
    body+=p('M252 589C262 603 268 611 267 626M597 589C587 603 581 611 582 626',stroke='#d5d9c5',w=3)
    n=[layer('36 Chair foreground contact and armrests',body,erase=[foreground_mask()])]
    # A restrained foreground branch adds depth without covering face or hands.
    body=p('M52 1110C98 1044 94 982 131 927C153 895 177 878 172 851',stroke='#8aa895',w=1.4,op=.75)
    for j,(x,y,s,a) in enumerate([(83,1030,.45,-55),(103,988,.35,53),(118,956,.35,-50),(142,916,.35,48),(158,887,.27,-40)]):
        body+=leaf(x,y,s,a,['#9cb7a0','#c0cba0','#9ebbb0'][j%3])
    for x,y,r in [(173,855,13),(137,936,9),(78,1049,11)]:body+=flower(x,y,r,'#fcf2df',18)
    body+=ell(122,994,18,23,'url(#white)',.2)
    n.append(layer('37 Quiet foreground botanical',body))
    stage('07-garden-pause',n,OUT/'details-reviewed/06b-detail-reviewed.kra')


def snapshot():
    # Portable scene preserves final layer recipes including reviewed opacities.
    all_layers=[]
    for file in ['01-sketch.json','01b-sketch-reviewed.json','02-contour.json','03-flats.json','04-shaded-bases.json','04b-garden.json','05-large-planes.json','05b-planes-softened.json','06-detail.json','06b-detail-reviewed.json','07-garden-pause.json']:
        for item in json.loads((OUT/file).read_text(encoding='utf-8'))['layers']:
            previous=next((n for n in all_layers if n['name']==item['name']),None)
            if previous is None:all_layers.append(copy.deepcopy(item))
            else:previous.update(copy.deepcopy(item))
            for name,visible in item.get('visibility',{}).items():
                for n in all_layers:
                    if n['name']==name:n['visible']=visible
    write('final-scene.json',{'name':'A garden pause - final study','stem':'garden-pause-replay','width':W,'height':H,'scale':SCALE,'layers':all_layers,'details':[{'file':'head.png','rect':[277,80,285,285]}]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['sketch','sketch-reviewed','contour','flats','eye-repair','shading','garden','planes','soften','details','detail-repair','finish','snapshot'])
    args=parser.parse_args()
    {'sketch':sketch,'sketch-reviewed':lambda:sketch(True),'contour':contour,'flats':flats,'eye-repair':eye_repair,'shading':shading,'garden':garden_stage,'planes':planes,'soften':soften_planes,'details':details,'detail-repair':detail_repair,'finish':finish,'snapshot':snapshot}[args.stage]()
    print('Prepared '+args.stage)
