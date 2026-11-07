import csv
import hashlib
import pytest
from PIL import Image
from evaluate_photos import evaluate


class Stub:
    metadata = {'artifact_sha256': 'test-only', 'model_id': 'stub'}
    def predict(self, image):
        return {'label': 'plastic'}


def manifest(tmp_path, rows):
    path = tmp_path / 'labels.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'photo', 'label'])
        writer.writerows(rows)
    return path


def test_missing_classes_are_not_reported_as_full_coverage(tmp_path):
    Image.new('RGB', (40, 40), 'red').save(tmp_path / 'one.png')
    Image.new('RGB', (40, 40), 'blue').save(tmp_path / 'two.png')
    report = evaluate(manifest(tmp_path, [('one','one.png','plastic'),('two','two.png','paper')]), Stub(), set())
    assert report['accuracy'] == .5
    assert report['six_class_macro_f1'] is None
    assert report['per_class']['glass']['recall'] is None
    assert report['per_class']['plastic']['precision'] == .5
    assert report['per_class']['paper']['recall'] == 0
    assert sum(map(sum,report['confusion_matrix'])) == 2


def test_rejects_reused_study_pixels_and_duplicate_images(tmp_path):
    im=Image.new('RGB',(40,40),'red')
    im.save(tmp_path/'one.png')
    im.save(tmp_path/'copy.png')
    fingerprint=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    one=manifest(tmp_path,[('one','one.png','plastic')])
    with pytest.raises(ValueError,match='recorded study'):
        evaluate(one,Stub(),{fingerprint})
    two=manifest(tmp_path,[('one','one.png','plastic'),('two','copy.png','plastic')])
    with pytest.raises(ValueError,match='duplicate image pixels'):
        evaluate(two,Stub(),set())


def test_rejects_empty_or_invalid_labels(tmp_path):
    with pytest.raises(ValueError,match='template is empty'):
        evaluate(manifest(tmp_path,[]),Stub(),set())
    with pytest.raises(ValueError,match='label must be'):
        evaluate(manifest(tmp_path,[('one','missing.png','unknown')]),Stub(),set())
