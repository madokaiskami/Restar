from restar.config import load_config


def test_load_config_smoke():
    cfg = load_config("configs/dvc_smoke.yaml")
    assert cfg.model.pretrained_name == "prajjwal1/bert-tiny"
    assert cfg.model.num_labels == 3
    assert cfg.train_stream.local_parquet == "data/raw/train.parquet"
    assert cfg.prepare.out_train_parquet == "data/raw/train.parquet"
