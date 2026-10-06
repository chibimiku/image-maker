import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import torch  # Load before Qt to avoid Windows c10 DLL initialization order issues.
from PIL import Image, ImageDraw
from utils import style_face_metrics as face
from utils import style_similarity as similarity


def fixture(path, broken=False, iris_gray=110):
    size = 512
    annotation = {"version": "style-regions/1", "confirmed": True, "pose": "frontal",
        "face_outline": [[.3,.25],[.7,.25],[.75,.9],[.25,.9]],
        "hair_regions": [[[.08,.04],[.28,.04],[.28,.43],[.08,.43]]],
        "hair_strands": [{"points": [[x,.08],[x,.4]], "polarity": "dark"} for x in (.13,.18,.23)],
        "eyes": {}, "nose": [.5,.67], "mouth": [.5,.77]}
    image = Image.new("RGB", (size,size), "white")
    draw = ImageDraw.Draw(image)
    def pixel(vertices):
        return [(round(x*(size-1)), round(y*(size-1))) for x,y in vertices]
    for strand in annotation["hair_strands"]:
        draw.line(pixel(strand["points"]), fill="black", width=2)
    if broken:
        draw.rectangle((45,118,140,150), fill="white")
    for name, start in (("viewer_left", .35),("viewer_right",.56)):
        xs = [start + i*.12/8 for i in range(9)]
        upper = [[x,.52 - .035*math.sin(i*math.pi/8)] for i,x in enumerate(xs)]
        lower = [[x,.52 + .025*math.sin(i*math.pi/8)] for i,x in enumerate(xs)]
        iris = [[start+.04,.49],[start+.08,.49],[start+.08,.54],[start+.04,.54]]
        lashes = [[[start+.02,.502],[start+.008,.479]], [[start+.035,.493],[start+.023,.47]]]
        annotation["eyes"][name] = {"state":"open", "upper_lid":upper,"lower_lid":lower,"iris":iris,"lashes":lashes}
        draw.polygon(pixel(iris), fill=(iris_gray,iris_gray,iris_gray))
        draw.line(pixel(upper), fill="black", width=3)
        draw.line(pixel(lower), fill="black", width=1)
        for lash in lashes:
            draw.line(pixel(lash), fill="black", width=2)
    image.save(path)
    annotation["image_sha256"] = face.digest(path)
    return annotation


class FaceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = str(self.root/'a.png')
        self.b = str(self.root/'b.png')
        self.aa = fixture(self.a)
        self.bb = fixture(self.b, broken=True, iris_gray=45)
    def tearDown(self):
        self.temp.cleanup()

    def test_identity_all_thirteen_and_normalized_geometry(self):
        measured = face.compare_face_features(self.a,self.a,{self.a:self.aa})
        self.assertEqual(len(measured),13)
        for metric, outcome in measured.items():
            self.assertEqual(outcome["status"], "ok", (metric,outcome))
            self.assertAlmostEqual(outcome["value"],1)
        descriptors = face.descriptors(self.a,self.aa)
        self.assertAlmostEqual(descriptors["eye_width"]["components"]["viewer_left"], .24, places=2)
        self.assertGreater(descriptors["eye_curvature"]["components"]["viewer_left_upper_arc"],0)

    def test_only_marked_hair_responds_to_hair_gaps(self):
        a = face.descriptors(self.a,self.aa)
        b = face.descriptors(self.b,self.bb)
        self.assertGreater(a["hair_continuity"]["components"]["coverage"],b["hair_continuity"]["components"]["coverage"])
        changed = str(self.root/'outside.png')
        with Image.open(self.a) as image:
            ImageDraw.Draw(image).rectangle((300,30,460,140),fill="black")
            image.save(changed)
        c = face.descriptors(changed,self.aa)
        self.assertEqual(a["hair_continuity"]["components"],c["hair_continuity"]["components"])
        self.assertEqual(a["hair_fineness"]["components"],c["hair_fineness"]["components"])

    def test_eye_brightness_and_symmetry(self):
        annotations = {self.a:self.aa,self.b:self.bb}
        forward = face.compare_face_features(self.a,self.b,annotations)
        reverse = face.compare_face_features(self.b,self.a,annotations)
        self.assertLess(forward["eye_brightness"]["value"],1)
        self.assertEqual(forward["eye_width"]["value"],1)
        for metric in face.METRICS:
            self.assertAlmostEqual(forward[metric]["value"],reverse[metric]["value"])

    def test_provisional_excluded_from_formal_mean(self):
        self.bb["confirmed"] = False
        pair = face.compare_face_features(self.a,self.b,{self.a:self.aa,self.b:self.bb})
        self.assertEqual(pair["eye_width"]["status"],"provisional")
        self.assertIsNotNone(pair["eye_width"]["value"])
        self.assertIsNone(similarity.aggregate([pair],"eye_width",1)["mean"])

    def test_pose_closed_eyes_hair_outside_and_low_resolution(self):
        changed = copy.deepcopy(self.bb)
        changed["pose"] = "profile"
        result = face.compare_face_features(self.a,self.b,{self.a:self.aa,self.b:changed})
        self.assertEqual(result["eye_width"]["status"],"not_comparable")
        self.assertEqual(result["hair_continuity"]["status"],"ok")
        changed["eyes"]["viewer_left"]["state"] = "closed"
        self.assertEqual(face.descriptors(self.b,changed)["eye_height"]["status"],"unavailable")
        changed["hair_strands"][0]["points"] = [[.8,.1],[.8,.4]]
        self.assertEqual(face.descriptors(self.b,changed)["hair_continuity"]["status"],"unavailable")
        tiny = str(self.root/'tiny.png')
        Image.open(self.a).resize((64,64)).save(tiny)
        self.assertTrue(all(v["status"]=="unavailable" for v in face.descriptors(tiny,self.aa).values()))

    def test_hash_bound_annotations_and_snapshot(self):
        from utils import style_regions
        def location(path):
            return self.root/(face.digest(path)+'.json')
        with patch.object(face,"sidecar",side_effect=location),patch.object(style_regions,"sidecar",side_effect=location):
            style_regions.save_annotation(self.a,self.aa)
            self.assertTrue(face.read_annotation(self.a)["confirmed"])
            before = face.annotation_snapshot([self.a])
            changed = copy.deepcopy(self.aa);changed["confirmed"]=False
            style_regions.save_annotation(self.a,changed)
            self.assertNotEqual(before,face.annotation_snapshot([self.a]))
            wrong = copy.deepcopy(changed);wrong["image_sha256"]="wrong"
            location(self.a).write_text(json.dumps(wrong),encoding="utf-8")
            with self.assertRaises(ValueError):
                face.read_annotation(self.a)

    def test_auto_proposal_cannot_claim_human_confirmation(self):
        from utils import style_regions
        response = MagicMock()
        malicious = copy.deepcopy(self.aa)
        malicious["confirmed"] = True
        response.choices[0].message.content = json.dumps(malicious)
        client=MagicMock();client.chat.completions.create.return_value=response
        with patch('openai.OpenAI',return_value=client),patch.object(style_regions,'sidecar',return_value=self.root/'proposal.json'),patch.object(style_regions,'save_annotation') as save:
            proposal=style_regions.propose_regions(self.a,('https://example.test','test-key','vision-model'))
        self.assertFalse(proposal["confirmed"])
        save.assert_called_once()

    def test_shared_engine_keeps_face_results_and_coverage(self):
        from utils.style_metrics import inventory
        def location(path):
            return self.root / (face.digest(path) + '.regions.json')
        for path, annotation in ((self.a,self.aa),(self.b,self.bb)):
            location(path).write_text(json.dumps(annotation),encoding='utf-8')
        keys=('vgg19','lpips_alex','lpips_alex_trunk','csd','clip_vit_l14')
        with patch.object(face,'sidecar',side_effect=location),patch.object(inventory,'list_inventory',return_value={k:{'sha256_matches':False} for k in keys}),patch.object(inventory,'preflight',return_value={m:[] for m in similarity.METRICS}):
            result=similarity.compare_images(similarity.image_manifest([self.a],[self.b]),'cpu')
        expected=face.compare_face_features(self.a,self.b,{self.a:self.aa,self.b:self.bb})
        self.assertEqual(result['rows'][0]['pairs'][0]['face_features'],expected)
        self.assertEqual(result['rows'][0]['face_summary']['eye_gap']['count'],1)
        self.assertEqual(result['face_contract']['version'],face.VERSION)

    def test_region_editor_manual_and_invalid_json_do_not_crash(self):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt6.QtWidgets import QApplication
        from modules.image_analysis.style_regions_dialog import StyleRegionsDialog
        app=QApplication.instance() or QApplication([])
        with patch('modules.image_analysis.style_regions_dialog.read_annotation',return_value=copy.deepcopy(self.aa)):
            dialog=StyleRegionsDialog(self.a)
        try:
            dialog.canvas.pending=[[.3,.2],[.7,.2],[.7,.9],[.3,.9]]
            dialog.commit()
            self.assertFalse(dialog.confirm.isChecked())
            dialog.json_editor.setPlainText('{"eyes":42}')
            dialog.show();app.processEvents()
            dialog.grab()
            dialog.show_annotations.setChecked(False)
            dialog.canvas.zoom = 3
            dialog.canvas.fit()
            self.assertEqual(dialog.canvas.zoom,1.0)
        finally:
            dialog.close()


if __name__ == '__main__':
    unittest.main()
