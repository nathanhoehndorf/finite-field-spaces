import ffspaces


def test_public_api_exposes_sumset_helpers_but_not_fwht():
    assert hasattr(ffspaces, "compute_sumset_fwht")
    assert hasattr(ffspaces, "generate_space_chunked")
    assert hasattr(ffspaces, "compute_sumset_lowmem")
    assert hasattr(ffspaces, "compute_sumset_lowmem_from_ints")
    assert hasattr(ffspaces, "generate_ball_ints_lowmem")
    assert hasattr(ffspaces, "generate_covering_ints_lowmem")
    assert hasattr(ffspaces, "complement_ints_lowmem")
    assert hasattr(ffspaces, "find_maximum_subspace_dimension_lowmem")
    assert hasattr(ffspaces, "estimate_memory_gb")
    assert not hasattr(ffspaces, "fwht")
