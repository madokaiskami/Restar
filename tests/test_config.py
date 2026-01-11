from restar.config import load_config


def test_load_config_smoke():
    cfg = load_config("configs/dvc_smoke.yaml")
    assert cfg.model.pretrained_name == "prajjwal1/bert-tiny"
    assert cfg.model.num_labels == 3
    assert cfg.train_stream.local_parquet == "data/frozen/train.parquet"
    assert not hasattr(cfg, "prepare")
