import numpy as np

from restar.labeling import postprocess_logits


def test_postprocess_abstain_and_label():
    logits = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    labels, conf = postprocess_logits(logits, threshold=0.5)
    assert labels[0] == "positive"
    assert labels[1] == "abstain"
    assert 0.0 <= conf[1] <= 1.0
