def assert_binary_target(y):
    unique = set(y.unique())
    assert unique <= {0, 1}, "Target não é binário"
