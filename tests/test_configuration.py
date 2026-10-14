from geoctgan.config import Config


def test_reported_generation_settings():
    cfg = Config()
    assert cfg.EMB_DIM == 16
    assert cfg.N_WALKS == 15
    assert cfg.WALK_LEN == 10
    assert cfg.W2V_WINDOW == 10
    assert cfg.W2V_EPOCHS == 6
    assert cfg.W2V_WORKERS == 1
    assert cfg.COHORT_KIND == "global"


def test_reported_capacity_settings():
    cfg = Config()
    assert cfg.CANN_COST_SCALE == 100000
    assert cfg.CANN_MAX_EDGES == 5000000
