from ftanomaly import config


def test_input_files_are_present():
    assert config.PATHS.shipments.exists()
    assert config.PATHS.notes.exists()
    assert config.PATHS.sample_output.exists()


def test_thresholds_are_the_locked_values():
    assert config.OWN_HISTORY_FLOOR_PCT == 10.0
    assert config.PEER_THRESHOLD_PCT == 20.0
    assert config.MAD_MULTIPLIER == 3.0
    assert config.TRAILING_WEEKS == 8
