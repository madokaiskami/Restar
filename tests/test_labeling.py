import numpy as np

from restar.labeling import class_to_text, postprocess_logits, rating_to_class


def test_rating_to_class_edges():
    assert rating_to_class(1) == 0
    assert rating_to_class(2) == 0
    assert rating_to_class(3) == 1
    assert rating_to_class(4) == 2
    assert rating_to_class(5) == 2


def test_class_to_text():
    assert class_to_text(0) == "negative"
    assert class_to_text(1) == "neutral"
    assert class_to_text(2) == "positive"


def test_postprocess_logits():
    logits = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    labels, conf = postprocess_logits(logits, threshold=0.5)
    assert labels[0] == "positive"
    assert labels[1] == "abstain"
    assert 0.0 <= conf[1] <= 1.0
