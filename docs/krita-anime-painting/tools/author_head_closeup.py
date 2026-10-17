"""Re-render the reviewed head geometry at high native resolution in Krita."""
import copy
import json
import re
from pathlib import Path

ROOT=Path('data/test-result/20261016/krita-seated-practice-v4')
P=ROOT/'head-practice'
s=json.loads((ROOT/'scene.json').read_text(encoding='utf-8'))
layers=copy.deepcopy(s['layers'])
changes=[ROOT/'face-refinement/01-skin.json',ROOT/'face-refinement/03-hair.json',
         ROOT/'gem-eyes/apply-to-portrait.json',P/'01-brows.json',P/'03-mouth.json',
         P/'04-ears-rendered-gradients.json',P/'05-face-planes.json',P/'06-hair.json',
         P/'07-head-review.json',P/'07-earrings.json',P/'08-eyelids.json']
for file in changes:
    for update in json.loads(file.read_text(encoding='utf-8'))['layers']:
        found=next((n for n in layers if n['name']==update['name']),None)
        if found is None:layers.append(copy.deepcopy(update))
        else:
            if 'svg' in update and 'strokes' not in update:found.pop('strokes',None)
            found.update(copy.deepcopy(update))

def crop_svg(svg):
    svg=re.sub(r'width="2400" height="3360" viewBox="0 0 1000 1400"',
               'width="2400" height="2600" viewBox="350 150 290 314.1666666667"',svg)
    return svg
for n in layers:
    n.pop('checkpoint',None)
    if 'svg' in n:n['svg']=crop_svg(n['svg'])
    for k in ['svg_masks','svg_erase_masks']:
        if k in n:n[k]=[crop_svg(v) for v in n[k]]
    for stroke in n.get('strokes',[]):
        if 'path' in stroke:
            for cmd in stroke['path']:
                for i in range(1,len(cmd),2):cmd[i]-=350;cmd[i+1]-=150
        if 'line' in stroke:
            stroke['line']=[v-(350 if i%2==0 else 150) for i,v in enumerate(stroke['line'])]
        if 'ellipse' in stroke:
            stroke['ellipse'][0]-=350;stroke['ellipse'][1]-=150

# Neutral study paper makes head planes easier to judge than a strong window
# light overlay; both original full paintings remain saved separately.
room=next(n for n in layers if n['name']=='04 Quiet window room')
room['svg']='<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2600" viewBox="0 0 2400 2600"><rect width="2400" height="2600" fill="#edf2f0"/></svg>'
for n in layers:
    if n['name'] in ['20 Small foreground flowers','21 Soft window atmosphere']:n['visible']=False
close={'name':'Azure head — reviewed anatomy and material study','width':2400,'height':2600,
       'scale':2400/290,'stem':'azure-head-study','blank_checkpoint':'00-head-blank.kra',
       'layers':layers,'details':[
        {'file':'head-detail.png','rect':[5,5,280,285]},
        {'file':'brows-detail.png','rect':[49,115,180,36]},
        {'file':'nose-detail.png','rect':[123,179,42,36]},
        {'file':'mouth-detail.png','rect':[120,209,48,33]},
        {'file':'ear-detail.png','rect':[198,144,61,99]}]}
(P/'head-closeup.json').write_text(json.dumps(close,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared high-resolution head study from the reviewed vector/brush recipe')
