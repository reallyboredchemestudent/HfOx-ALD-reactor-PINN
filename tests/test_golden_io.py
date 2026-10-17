"""The stdout comparison itself, on the exact lines that differed between CI runners (macOS Accelerate and
Linux OpenBLAS builds, Python 3.12 and 3.13), and on changes it must still catch."""
from golden_io import compare_stdout


def same(got, want):
    return not compare_stdout([got], [want])


def test_round_off_residuals_agree():
    assert same("      top vs bottom plate max difference 2.2e-16", "      top vs bottom plate max difference 6.7e-16")
    assert same("mediated, purge to t=200: max|dtheta| 7.5e-01 (11.29 nm) | pool left -7e-21",
                "mediated, purge to t=200: max|dtheta| 7.5e-01 (11.29 nm) | pool left 7e-21")
    assert same("grid 24x32: same-grid time vs steady max|dtheta| 1.3e-08 | gas left 9e-18 | <t>",
                "grid 24x32: same-grid time vs steady max|dtheta| 1.1e-08 | gas left 9e-18 | <t>")


def test_eigenvector_sign_is_free():
    want = "  sloppiest direction: {'log_lam': 0.05, 'log_zb': 0.02, 'Ea': -0.0, 'log_l': -0.06, 'U': 1.0, 'g1': -0.02}"
    got = "  sloppiest direction: {'log_lam': -0.05, 'log_zb': -0.02, 'Ea': 0.0, 'log_l': 0.06, 'U': -1.0, 'g1': 0.02}"
    assert same(got, want)


def test_real_changes_still_fail():
    # a discretisation error above the noise floor
    assert not same("N= 800: time vs steady max|dtheta| 2.6e-06", "N= 800: time vs steady max|dtheta| 2.3e-06")
    # a residual that grew past the floor
    assert not same("max|dtheta| 2.0e-07 (0.0000 nm)", "max|dtheta| 4.4e-08 (0.0000 nm)")
    # a direction with only some components flipped is a different direction
    assert not same("  weakest direction: {'log_lam': -0.71, 'U': 0.7}", "  weakest direction: {'log_lam': 0.71, 'U': 0.7}")
    # sign is free only on direction lines
    assert not same("  log_lam  fit -3.005", "  log_lam  fit 3.005")


def test_words_containing_inf_or_nan_are_text():
    line = "  information by wafer (increase in total physics variance if the wafer is removed), top 6:"
    assert same(line, line)
    assert same("250 C  3.85 +-  nan  (not resolved)", "250 C  3.85 +-  nan  (not resolved)")
