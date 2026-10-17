"""Record corrections found by viewing the actual neck/ear study."""
import copy
import json
from pathlib import Path

out=Path('data/test-result/20261017/krita-tutorial-volume')
two=json.loads((out/'02-neck-ear.json').read_text())
three=json.loads((out/'03-lighting.json').read_text())
neck,ear=copy.deepcopy(two['layers'])
neck['svg']=neck['svg'].replace('opacity="0.15"','opacity="0.075"')
defs=ear['svg'].split('</defs>')[0]+'</defs>'
ear['svg']=defs+'''<path d="M71 -13C83 -19 87 -7 83 4Q79 14 79 22C77 30 72 32 70 27Q68 21 72 15L71 5Q66 -5 71 -13Z" fill="url(#shoulderSkin)" stroke="#b893a3" stroke-width=".5"/>
<ellipse cx="77" cy="10" rx="3.6" ry="7" fill="url(#earBowl)"/>
<path d="M73 -9Q83 -13 83 -3Q83 3 78 7M77 7Q73 10 75 16Q78 20 73 25M77 7L80 3" fill="none" stroke="#bf97a6" stroke-width=".6" stroke-linecap="round"/>
<path d="M74 -9Q81 -12 82 -5" fill="none" stroke="#fff0df" stroke-width=".85"/>
<path d="M70 11Q76 9 75 15L72 18Z" fill="#edc9c5"/></svg>'''
repair={k:v for k,v in three.items() if k!='layers'}
repair.update(stem='tutorial-volume-refined',expected_document=str((out/'painting/tutorial-volume-reviewed.kra').resolve()),layers=[neck,ear])
(out/'04-review-repair.json').write_text(json.dumps(repair),encoding='utf-8')
snapshot=json.loads((out/'01-construction.json').read_text())
snapshot['stem']='tutorial-volume-final-snapshot'
snapshot.pop('blank_checkpoint',None)
for n in two['layers']+three['layers']+repair['layers']:
    old=next((x for x in snapshot['layers'] if x['name']==n['name']),None)
    if old is None:snapshot['layers'].append(copy.deepcopy(n))
    else:old.update(copy.deepcopy(n))
for n in snapshot['layers']:
    if n['name']=='36 Tutorial neck cylinder and ear landmarks':n['visible']=False
(out/'final-scene-snapshot.json').write_text(json.dumps(snapshot),encoding='utf-8')
print('Prepared smaller ear and subtler neck anatomy after visual review')

# Original-size inspection exposed a hard wedge on the cheek and an isolated
# ear lobe beneath the gathered lock. Replace these at their source layers.
ear2=copy.deepcopy(ear)
ear2['svg']=defs+'''<path d="M71 -13C83 -19 87 -7 83 4Q80 11 76 17Q70 15 71 5Q66 -5 71 -13Z" fill="url(#shoulderSkin)" stroke="#b893a3" stroke-width=".5"/>
<ellipse cx="77" cy="5" rx="3" ry="5" fill="url(#earBowl)" opacity=".6"/>
<path d="M73 -9Q83 -13 83 -3Q83 3 78 7M77 7Q73 10 75 14" fill="none" stroke="#bf97a6" stroke-width=".6" stroke-linecap="round"/>
<path d="M74 -9Q81 -12 82 -5" fill="none" stroke="#fff0df" stroke-width=".85"/></svg>'''
light=copy.deepcopy(three['layers'][0])
defs2=light['svg'].split('</defs>')[0]+'</defs>'
face='M-71 -65C-77 -17 -59 48 -28 74Q-6 92 7 95C53 72 74 24 72 -56C65 -111 -53 -120 -71 -65Z'
light['svg']=defs2+f'<path d="{face}" fill="url(#cheekTurn)" opacity=".65"/><ellipse cx="6" cy="35" rx="3.5" ry="4" fill="url(#earBowl)" opacity=".25"/></svg>'
for cx in [-37,37]:
    light['svg_erase_masks'][-2 if cx==-37 else -1]=defs2+f'<path d="M{cx-23} 0Q{cx} -17 {cx+23} -1Q{cx} 20 {cx-23} 0Z" fill="#fff"/></svg>'
final={**repair,'stem':'tutorial-volume-final','expected_document':str((out/'painting/tutorial-volume-refined.kra').resolve()),'layers':[ear2,light]}
(out/'05-edge-repair.json').write_text(json.dumps(final),encoding='utf-8')
edge_check={**final,'stem':'tutorial-volume-eye-edge-checked','expected_document':str((out/'painting/tutorial-volume-final.kra').resolve()),'layers':[light]}
(out/'06-eye-edge-check.json').write_text(json.dumps(edge_check),encoding='utf-8')
for n in final['layers']:
    next(x for x in snapshot['layers'] if x['name']==n['name']).update(copy.deepcopy(n))
(out/'final-scene-snapshot.json').write_text(json.dumps(snapshot),encoding='utf-8')
