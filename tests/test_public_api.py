import ffspaces


def test_public_api_exposes_sumset_helpers_but_not_fwht():
    assert hasattr(ffspaces, "compute_sumset_fwht")
    assert not hasattr(ffspaces, "fwht")
