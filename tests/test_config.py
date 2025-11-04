from restar.config import load_config


def test_load_config_tiny():
    cfg = load_config("configs/tiny.yaml")
    assert cfg.model.num_labels == 3
    assert cfg.data.source == "synthetic"
