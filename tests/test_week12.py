import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from wastesort.common import ROOT, read_json


def test_intro_prediction_evidence_matches_frozen_validation():
    predictions = pd.read_csv(ROOT / 'reports/intro_validation_predictions.csv')
    manifest = pd.read_csv(ROOT / 'data/splits.csv')
    expected = manifest[manifest.split == 'validation'].set_index('path').label.to_dict()
    assert len(predictions) == 742
    for name, frame in predictions.groupby('model'):
        assert len(frame) == 371 and frame.path.is_unique
        assert frame.set_index('path').actual.to_dict() == expected
        assert set(frame.split) == {'validation'}
        assert np.isclose(f1_score(frame.actual, frame.predicted, average='macro'),
                          read_json(ROOT / f'reports/{name}_training.json')['validation_macro_f1'])
        assert (frame.correct == (frame.actual == frame.predicted)).all()
    samples = pd.read_csv(ROOT / 'reports/intro_sample_predictions.csv')
    assert len(samples) == 24
    assert samples.groupby(['model','actual']).size().eq(2).all()


def test_review_has_traceable_decisions_for_every_candidate():
    review = pd.read_csv(ROOT / 'reports/manual_review_worksheet.csv')
    pairs = pd.read_csv(ROOT / 'reports/near_duplicate_candidates.csv')
    reviewed = review[review.kind == 'near_duplicate_candidate']
    assert len(reviewed) == len(pairs) == 45
    assert list(zip(reviewed.left, reviewed.right)) == list(zip(pairs.left, pairs.right))
    assert review[['reviewer','decision','notes']].notna().all().all()
    assert set(reviewed.decision) == {'likely_same_item','visibly_distinct','uncertain'}
