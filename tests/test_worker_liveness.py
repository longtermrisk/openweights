from openweights.cluster.liveness import worker_looks_alive

KW = dict(threshold=120, grace=600, hard_limit=3600)


def test_fresh_ping_is_alive_whatever_the_log_says():
    assert worker_looks_alive(30, None, **KW)
    assert worker_looks_alive(120, 5000, **KW)


def test_stale_ping_with_unreadable_log_is_dead():
    assert not worker_looks_alive(121, None, **KW)


def test_stale_ping_but_growing_log_is_alive():
    assert worker_looks_alive(125, 0, **KW)
    assert worker_looks_alive(1800, 599, **KW)


def test_stale_ping_and_stalled_log_is_dead():
    assert not worker_looks_alive(125, 600, **KW)
    assert not worker_looks_alive(125, 3000, **KW)


def test_hard_limit_wins_over_a_growing_log():
    assert not worker_looks_alive(3601, 0, **KW)
