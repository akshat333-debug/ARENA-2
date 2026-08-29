"""M4 unit tests: observation encoders and baseline feature views."""

from __future__ import annotations

import numpy as np

from arena.features import (
    ALLOW,
    CALL_FEATS,
    FLAG,
    RED_EXTRA,
    SINGLE_CALL_FEATS,
    SIDE_EFFECTS,
    encode_call,
    history_matrix,
    sequence_features,
    sequence_features_from_rows,
    single_call_features,
)
from arena.tools import COMMON_TOOLS

BY = {t.name: t for t in COMMON_TOOLS}


# --- encode_call ----------------------------------------------------


def test_encode_call_width_matches_constants():
    v = encode_call(BY["read_file"], 0, 12, resolved=False, verdict=None, for_red=False)
    assert v.shape == (CALL_FEATS,)
    vr = encode_call(BY["read_file"], 0, 12, resolved=False, verdict=None, for_red=True)
    assert vr.shape == (CALL_FEATS + RED_EXTRA,)


def test_encode_call_side_effect_and_category_onehot():
    v = encode_call(BY["post_webhook"], 3, 12, resolved=True, verdict=ALLOW, for_red=False)
    assert v[SIDE_EFFECTS.index("external_sink")] == 1.0
    assert v[:6].sum() == 1.0  # exactly one side-effect bit


def test_encode_call_position_and_verdict():
    v = encode_call(BY["read_file"], 6, 12, resolved=True, verdict=FLAG, for_red=False)
    pos = v[6 + 8 + 3]
    assert abs(pos - 0.5) < 1e-6
    # flag bit set in the (allow, flag) pair
    assert v[6 + 8 + 5 + 1] == 1.0


def test_pending_proposal_has_resolved_flag_zero():
    v = encode_call(BY["read_file"], 2, 12, resolved=False, verdict=None, for_red=False)
    assert v[6 + 8 + 4] == 0.0


def test_single_call_features_is_the_observable_prefix_of_encode_call():
    for name in ("read_env_file", "summarize_text", "post_webhook", "fetch_url"):
        row = encode_call(BY[name], 4, 12, resolved=True, verdict=ALLOW, for_red=False)
        assert np.array_equal(single_call_features(BY[name]), row[:SINGLE_CALL_FEATS])


def test_single_call_features_identical_for_a_call_regardless_of_context():
    # the whole point: a lone read looks the same everywhere
    a = single_call_features(BY["read_env_file"])
    b = single_call_features(BY["read_env_file"])
    assert np.array_equal(a, b)


# --- history_matrix ----------------------------------------------


def test_history_matrix_lays_out_calls_then_pending():
    hist = [(BY["read_env_file"], 0, ALLOW), (BY["summarize_text"], 1, ALLOW)]
    mat, length = history_matrix(hist, (BY["post_webhook"], 2), 12, for_red=False)
    assert length == 3
    assert mat.shape == (12, CALL_FEATS)
    assert mat[2, SIDE_EFFECTS.index("external_sink")] == 1.0
    assert mat[2, 6 + 8 + 4] == 0.0  # pending -> resolved flag 0
    assert mat[3:].sum() == 0.0      # zero-padded


def test_history_matrix_no_pending():
    hist = [(BY["read_file"], 0, ALLOW)]
    mat, length = history_matrix(hist, None, 12, for_red=True)
    assert length == 1
    assert mat.shape == (12, CALL_FEATS + RED_EXTRA)


# --- sequence_features -----------------------------------------


def test_sequence_features_flags_sensitive_read_before_sink():
    canon = [BY["read_env_file"], BY["summarize_text"], BY["post_webhook"]]
    v = sequence_features(canon)
    names = _seq_names()
    assert v[names.index("sensitive_read_before_sink")] == 1.0
    assert v[names.index("has_external_sink")] == 1.0


def test_sequence_features_no_false_positive_when_sink_precedes_read():
    seq = [BY["post_webhook"], BY["read_env_file"], BY["summarize_text"]]
    v = sequence_features(seq)
    assert v[_seq_names().index("sensitive_read_before_sink")] == 0.0


def test_sequence_features_untrusted_before_privileged():
    seq = [BY["fetch_url"], BY["summarize_text"], BY["read_credentials"]]
    v = sequence_features(seq)
    assert v[_seq_names().index("untrusted_read_before_privileged")] == 1.0


def test_sequence_features_empty_is_zero():
    assert not sequence_features([]).any()


def test_sequence_features_from_rows_agrees_with_from_tools():
    seqs = [
        [BY["read_env_file"], BY["summarize_text"], BY["post_webhook"]],
        [BY["read_file"], BY["translate_text"], BY["write_file"]],
        [BY["fetch_url"], BY["run_script"], BY["upload_file"]],
        [BY["list_directory"]],
    ]
    for seq in seqs:
        rows = np.stack([
            encode_call(t, i, 12, resolved=True, verdict=ALLOW, for_red=False)
            for i, t in enumerate(seq)
        ])
        a = sequence_features(seq)
        b = sequence_features_from_rows(rows)
        assert np.allclose(a, b), seq


def _seq_names():
    from arena.features import SEQUENCE_FEATURE_NAMES

    return list(SEQUENCE_FEATURE_NAMES)
