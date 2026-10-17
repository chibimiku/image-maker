"""Reusable local Krita bridge CLI: inventory and declarative layered studies.

Artwork, paths and brush choices live in a JSON scene, not in this tool.
Krita's own Python/Qt and paint engine perform all rendering.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.request


def bridge_call(operation, params, timeout=120):
    state = Path(os.environ['APPDATA']) / 'krita' / 'krita_mcp_bridge.json'
    info = json.loads(state.read_text(encoding='utf-8'))
    request = urllib.request.Request(
        'http://127.0.0.1:%d/rpc' % int(info['port']),
        data=json.dumps({'op': operation, 'params': params, 'timeout': timeout}).encode(),
        headers={'Content-Type': 'application/json', 'X-Krita-MCP-Token': info['token']})
    with urllib.request.urlopen(request, timeout=timeout+15) as response:
        payload = json.load(response)
    if not payload.get('ok') or payload.get('result', {}).get('ok') is False:
        raise RuntimeError(json.dumps(payload, ensure_ascii=True))
    return payload


INVENTORY = '''
view=krita.activeWindow().activeView()
presets=krita.resources('preset')
filters={}
for name in ['gaussian blur','unsharp','hsvadjustment','desaturate','colorbalance']:
    f=krita.filter(name)
    if f:
        c=f.configuration()
        filters[name]={str(k):str(v) for k,v in c.properties().items()}
result={'version':krita.version(),'presets':sorted(presets),
        'native_paint_api':{name:hasattr(doc,name) if name=='createNode' else hasattr(doc.activeNode(),name) for name in ['createNode','paintLine','paintPath','paintEllipse']} if doc else {},
        'filters':filters,'brush':view.currentBrushPreset().name() if view and view.currentBrushPreset() else None}
'''


RENDER = r'''
from PyQt5.QtCore import QByteArray, QRectF, QPoint, QRect
from PyQt5.QtGui import QImage,QPainter,QPainterPath,QColor
from PyQt5.QtSvg import QSvgRenderer
from krita import ManagedColor
import json, math, hashlib
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
scene=json.loads(SCENE_JSON)
out=Path(OUTPUT_DIR)
out.mkdir(parents=True,exist_ok=True)
presets=krita.resources('preset')
def specs(nodes):
    for item in nodes:
        yield item
        yield from specs(item.get('children',[]))
required={s['brush'] for item in specs(scene['layers']) for s in item.get('strokes',[])}
missing=required-set(presets)
if missing: raise ValueError('Missing presets: '+repr(sorted(missing)))
W,H=scene['width'],scene['height']
scale=scene.get('scale',1)
if REVISION or CONTINUE:
    doc=krita.activeDocument()
    expected=scene.get('expected_document')
    if not expected or not doc or Path(doc.fileName()).resolve()!=Path(expected).resolve():
        raise ValueError('Revision requires the explicitly expected active document')
    if [doc.width(),doc.height()]!=[W,H]:raise ValueError('Revision dimensions mismatch')
    if CONTINUE:
        def existing_nodes(parent):
            for n in parent.childNodes():
                yield n.name()
                yield from existing_nodes(n)
        collisions=set(existing_nodes(doc.rootNode())) & {i['name'] for i in specs(scene['layers'])}
        if collisions:raise ValueError('Append layer names already exist: '+repr(sorted(collisions)))
else:
    doc=krita.createDocument(W,H,scene['name'],'RGBA','U8','',300)
    krita.activeWindow().addView(doc)
doc.setBatchmode(True)
view=krita.activeWindow().activeView()
previous=(view.currentBrushPreset(),view.brushSize(),view.paintingOpacity(),view.foregroundColor())
log=[]
def color(value): return ManagedColor.fromQColor(QColor(value),view.canvas())
def compatible_radials(svg):
    # This embedded QSvgRenderer accepts objectBoundingBox radial gradients
    # as valid SVG but paints them transparent. Resolve each use against its
    # actual geometry bounds and supply an equivalent user-space transform.
    root=ET.fromstring(svg)
    gradients={n.get('id'):n for n in root.iter()
               if n.tag.split('}')[-1]=='radialGradient'
               and n.get('gradientUnits','objectBoundingBox')=='objectBoundingBox'}
    targets=[]
    for n in root.iter():
        fill=n.get('fill','')
        if fill.startswith('url(#') and fill.endswith(')') and fill[5:-1] in gradients:
            old_id=n.get('id');temp_id='krita_radial_bounds_'+str(len(targets))
            n.set('id',temp_id);targets.append((n,gradients[fill[5:-1]],temp_id,old_id))
    if not targets:return svg
    probe=QSvgRenderer(QByteArray(ET.tostring(root,encoding='utf-8')))
    defs=next((n for n in root if n.tag.split('}')[-1]=='defs'),None)
    if defs is None:return svg
    def fraction(value):
        value=str(value)
        return float(value[:-1])/100 if value.endswith('%') else float(value)
    for i,(n,g,temp_id,old_id) in enumerate(targets):
        bounds=probe.boundsOnElement(temp_id)
        if old_id:n.set('id',old_id)
        else:n.attrib.pop('id',None)
        if bounds.isEmpty():continue
        clone=copy.deepcopy(g);new_id=g.get('id')+'_krita_user_'+str(i)
        clone.set('id',new_id);clone.set('gradientUnits','userSpaceOnUse')
        for key,default in [('cx','.5'),('cy','.5'),('r','.5')]:
            clone.set(key,str(fraction(g.get(key,default))))
        clone.set('fx',str(fraction(g.get('fx',g.get('cx','.5')))))
        clone.set('fy',str(fraction(g.get('fy',g.get('cy','.5')))))
        transform='matrix(%s 0 0 %s %s %s)'%(bounds.width(),bounds.height(),bounds.x(),bounds.y())
        clone.set('gradientTransform',transform+' '+g.get('gradientTransform',''))
        defs.append(clone);n.set('fill','url(#'+new_id+')')
    return ET.tostring(root,encoding='unicode')

def write_svg(node,svg,masks=None,erase_masks=None):
    svg=compatible_radials(svg)
    renderer=QSvgRenderer(QByteArray(svg.encode('utf-8')))
    if not renderer.isValid(): raise ValueError('Invalid SVG: '+node.name())
    image=QImage(W,H,QImage.Format_ARGB32);image.fill(0)
    p=QPainter(image);p.setRenderHint(QPainter.Antialiasing)
    renderer.render(p,QRectF(0,0,W,H));p.end()
    # QSvgRenderer's SVG subset may ignore clipPath. Explicit raster masks
    # guarantee the same clipped bounds for SVG content in this Krita version.
    mask_operations=[(s,QPainter.CompositionMode_DestinationIn) for s in masks or []]
    mask_operations += [(s,QPainter.CompositionMode_DestinationOut) for s in erase_masks or []]
    for mask_svg,mode in mask_operations:
        mask_svg=compatible_radials(mask_svg)
        mask_renderer=QSvgRenderer(QByteArray(mask_svg.encode('utf-8')))
        if not mask_renderer.isValid():raise ValueError('Invalid SVG alpha mask')
        mask=QImage(W,H,QImage.Format_ARGB32);mask.fill(0)
        mp=QPainter(mask);mp.setRenderHint(QPainter.Antialiasing)
        mask_renderer.render(mp,QRectF(0,0,W,H));mp.end()
        p=QPainter(image);p.setCompositionMode(mode)
        p.drawImage(0,0,mask);p.end()
    ptr=image.constBits();ptr.setsize(image.byteCount())
    node.setPixelData(QByteArray(bytes(ptr)),0,0,W,H)
def curve(commands):
    p=QPainterPath()
    for c in commands:
        v=[x*scale for x in c[1:]]
        if c[0]=='M':p.moveTo(*v)
        elif c[0]=='L':p.lineTo(*v)
        elif c[0]=='C':p.cubicTo(*v)
        elif c[0]=='Q':p.quadTo(*v)
        elif c[0]=='Z':p.closeSubpath()
        else:raise ValueError('Unknown path command '+c[0])
    return p
def make(items,parent):
    for item in items:
        if REVISION:
            def walk(p):
                for n in p.childNodes():
                    yield n
                    yield from walk(n)
            matches=[n for n in walk(doc.rootNode()) if n.name()==item['name']]
            if len(matches)!=1:raise ValueError('Revision layer must be unique: '+item['name'])
            node=matches[0]
        else:
            node=doc.createNode(item['name'],'grouplayer' if 'children' in item else 'paintlayer')
            parent.addChildNode(node,None)
        if 'children' in item:make(item['children'],node)
        if 'svg' in item:write_svg(node,item['svg'],item.get('svg_masks'),item.get('svg_erase_masks'))
        doc.setActiveNode(node)
        doc.waitForDone()
        for stroke in item.get('strokes',[]):
            view.setCurrentBrushPreset(presets[stroke['brush']])
            view.setBrushSize(stroke['size']*scale)
            view.setPaintingOpacity(stroke.get('opacity',1))
            view.setForeGroundColor(color(stroke['color']))
            if node.paintAbility()!='PAINT':raise RuntimeError('Native brush not paintable')
            if 'path' in stroke:
                p=curve(stroke['path'])
                if stroke.get('taper'):
                    count=stroke.get('samples',36)
                    for j in range(count):
                        a,b=j/count,(j+1)/count
                        pressure=lambda t:max(.05,math.sin(math.pi*t)**.45)
                        node.paintLine(p.pointAtPercent(a).toPoint(),p.pointAtPercent(b).toPoint(),pressure(a),pressure(b))
                else:node.paintPath(p,'ForegroundColor','None')
            elif 'ellipse' in stroke:
                x,y,w,h=stroke['ellipse']
                node.paintEllipse(QRect(round(x*scale),round(y*scale),round(w*scale),round(h*scale)),'ForegroundColor','None')
            elif 'line' in stroke:
                x,y,xx,yy=stroke['line']
                node.paintLine(QPoint(round(x*scale),round(y*scale)),QPoint(round(xx*scale),round(yy*scale)),stroke.get('p1',1),stroke.get('p2',1))
            doc.waitForDone()
        for setting in item.get('filters',[]):
            f=krita.filter(setting['name']);cfg=f.configuration()
            for k,v in setting.get('settings',{}).items():cfg.setProperty(k,v)
            f.setConfiguration(cfg);f.apply(node,0,0,W,H);doc.waitForDone()
            log.append({'filter':setting['name'],'settings':setting.get('settings',{}),'layer':node.name()})
        if not REVISION or 'blend' in item:node.setBlendingMode(item.get('blend','normal'))
        if not REVISION or 'opacity' in item:node.setOpacity(round(item.get('opacity',1)*255))
        if not REVISION or 'inherit_alpha' in item:node.setInheritAlpha(item.get('inherit_alpha',False))
        if not REVISION or 'visible' in item:node.setVisible(item.get('visible',True))
        if item.get('visibility'):
            def visibility_walk(p):
                for n in p.childNodes():
                    if n.name() in item['visibility']:n.setVisible(bool(item['visibility'][n.name()]))
                    visibility_walk(n)
            visibility_walk(doc.rootNode())
        log.append({'layer':node.name(),'brushes':sorted({s['brush'] for s in item.get('strokes',[])}),'native_strokes':len(item.get('strokes',[])),'blend':node.blendingMode(),'inherit_alpha':node.inheritAlpha()})
        if item.get('checkpoint'):
            doc.refreshProjection();doc.waitForDone()
            if not doc.exportImage(str(out/item['checkpoint']),InfoObject()):raise RuntimeError('Checkpoint export failed')
try:
    if not REVISION and not CONTINUE and scene.get('blank_checkpoint'):
        if not doc.saveAs(str(out/scene['blank_checkpoint'])):raise RuntimeError('Blank document save failed')
    make(scene['layers'],doc.rootNode())
    doc.refreshProjection();doc.waitForDone()
    for item in scene.get('variants',[]):
        for n in doc.rootNode().childNodes():
            if n.name() in item.get('visibility',{}):n.setVisible(item['visibility'][n.name()])
        doc.refreshProjection();doc.waitForDone()
        if not doc.exportImage(str(out/item['file']),InfoObject()):raise RuntimeError('Variant export failed')
    stem=scene.get('stem','study')
    if not doc.saveAs(str(out/(stem+'.kra'))):raise RuntimeError('Save KRA failed')
    doc.waitForDone()
    if not doc.exportImage(str(out/(stem+'.png')),InfoObject()):raise RuntimeError('Export PNG failed')
    doc.waitForDone()
    for detail in scene.get('details',[]):
        x,y,w,h=[round(v*scale) for v in detail['rect']]
        if not doc.projection(x,y,w,h).save(str(out/detail['file']),'PNG'):raise RuntimeError('Detail export failed')
    record={'name':scene['name'],'size':[W,H],'created_new_document':not REVISION and not CONTINUE,'continued_document':CONTINUE,'blank_checkpoint':scene.get('blank_checkpoint'),'presets':sorted(required),'log':log,'kra':str(out/(stem+'.kra')),'png':str(out/(stem+'.png'))}
    (out/'render-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    result=record
finally:
    view.setCurrentBrushPreset(previous[0]);view.setBrushSize(previous[1]);view.setPaintingOpacity(previous[2]);view.setForeGroundColor(previous[3])
    doc.setBatchmode(False)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status', 'inventory', 'render', 'revise', 'append'])
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    if args.command == 'status':
        result = bridge_call('status', {})
    elif args.command == 'inventory':
        result = bridge_call('run_python', {'code': INVENTORY})
    else:
        if not args.scene or not args.output_dir:
            parser.error('render requires --scene and --output-dir')
        scene = json.loads(args.scene.read_text(encoding='utf-8'))
        code = 'REVISION='+repr(args.command=='revise')+'\nCONTINUE='+repr(args.command=='append')+'\nSCENE_JSON='+repr(json.dumps(scene, ensure_ascii=True))+'\nOUTPUT_DIR='+repr(str(args.output_dir.resolve()))+'\n'+RENDER
        result = bridge_call('run_python', {'code': code}, 300)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
