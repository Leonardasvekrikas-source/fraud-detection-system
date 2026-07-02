import textwrap

from fraud_detection import config


def test_defaults_match_thesis():
    assert config.WINDOW_SIZE == 3
    assert config.INFERENCE_THETA == 0.5
    assert config.TARGET_FRAUD_RATIO == 0.05
    assert config.LGBM_PARAMS["n_estimators"] == 5000
    assert config.LGBM_PARAMS["max_depth"] == 16


def test_reload_overrides_then_restores(tmp_path):
    original_window = config.WINDOW_SIZE
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        textwrap.dedent(
            """
            WINDOW_SIZE: 7
            INFERENCE_THETA: 0.3
            """
        ),
        encoding="utf-8",
    )
    try:
        config.reload(custom)
        assert config.WINDOW_SIZE == 7
        assert config.INFERENCE_THETA == 0.3
    finally:
        # restore the default config for other tests
        config.reload(None)

    assert config.WINDOW_SIZE == original_window
