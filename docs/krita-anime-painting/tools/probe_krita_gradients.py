"""Check actual embedded Qt SVG gradient behavior without touching a canvas."""
import json
import ast
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tools.krita_studio import bridge_call,RENDER

code=r'''
from PyQt5.QtCore import QByteArray,QRectF
from PyQt5.QtGui import QImage,QPainter
from PyQt5.QtSvg import QSvgRenderer
variants={
 'implicit_radial':'<radialGradient id="g">',
 'explicit_radial':'<radialGradient id="g" cx="50%" cy="50%" r="50%">',
 'user_radial':'<radialGradient id="g" gradientUnits="userSpaceOnUse" cx="50" cy="50" r="45">',
 'explicit_linear':'<linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">'}
report={}
for key,definition in variants.items():
    closing='</linearGradient>' if 'linear' in key else '</radialGradient>'
    s='<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><defs>'+definition+'<stop offset="0" stop-color="#c08090" stop-opacity=".8"/><stop offset="1" stop-color="#c08090" stop-opacity="0"/>'+closing+'</defs><ellipse cx="50" cy="50" rx="45" ry="45" fill="url(#g)"/></svg>'
    im=QImage(100,100,QImage.Format_ARGB32);im.fill(0)
    renderer=QSvgRenderer(QByteArray(s.encode()));p=QPainter(im);renderer.render(p,QRectF(0,0,100,100));p.end()
    report[key]={'valid':renderer.isValid(),'center_rgba':im.pixelColor(50,50).getRgb(),'offset_rgba':im.pixelColor(70,50).getRgb()}
result=report
'''
fn=next(n for n in ast.parse(RENDER).body if isinstance(n,ast.FunctionDef) and n.name=='compatible_radials')
code += '\nimport copy\nimport xml.etree.ElementTree as ET\n'+ast.get_source_segment(RENDER,fn)
code += r'''
def raster_sample(s,x,y,w=200,h=100):
    im=QImage(w,h,QImage.Format_ARGB32);im.fill(0)
    r=QSvgRenderer(QByteArray(s.encode()));p=QPainter(im);r.render(p,QRectF(0,0,w,h));p.end()
    return im.pixelColor(x,y).getRgb()
start='<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><defs><radialGradient id="g"><stop offset="0" stop-color="#c08090" stop-opacity=".8"/><stop offset="1" stop-color="#c08090" stop-opacity="0"/></radialGradient></defs>'
shared=start+'<ellipse cx="50" cy="50" rx="35" ry="20" fill="url(#g)"/><ellipse cx="150" cy="50" rx="20" ry="35" fill="url(#g)"/></svg>'
fixed=compatible_radials(shared)
path=start+'<path d="M15 15H85V85H15Z" fill="url(#g)"/></svg>'
transformed=start+'<g transform="translate(100 0)"><ellipse cx="50" cy="50" rx="25" ry="35" fill="url(#g)"/></g></svg>'
samples={'shared_left':raster_sample(fixed,50,50),'shared_right':raster_sample(fixed,150,50),
         'shared_edge':raster_sample(fixed,80,50),'path_center':raster_sample(compatible_radials(path),50,50),
         'transformed_center':raster_sample(compatible_radials(transformed),150,50)}
assert all(samples[k][3]>150 for k in ['shared_left','shared_right','path_center','transformed_center']),samples
assert samples['shared_edge'][3]<samples['shared_left'][3],samples
user=shared.replace('<radialGradient id="g">','<radialGradient id="g" gradientUnits="userSpaceOnUse" cx="50" cy="50" r="20">')
assert compatible_radials(user)==user
report['normalization_regression']={'samples':samples,'user_space_preserved':True,'passed':True}
result=report
'''
result=bridge_call('run_python',{'code':code})
out=Path('data/test-result/20261016/krita-seated-practice-v4/head-practice/gradient-probe.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
