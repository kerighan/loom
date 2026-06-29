"""Shared sampling helper used by Collection and the data structures."""


def reservoir_sample(iterable, n, random=True, seed=None):
    """Return up to ``n`` elements drawn from ``iterable``.

    With ``random=True`` (default) this is a uniform reservoir sample — a single
    pass, no need to know the population size up front. With ``random=False`` it
    returns the first ``n`` elements (stops early, no full scan). ``seed`` makes
    a random sample reproducible.
    """
    n = max(0, int(n))
    if n == 0:
        return []
    if not random:
        out = []
        for item in iterable:
            out.append(item)
            if len(out) >= n:
                break
        return out

    import random as _random

    rng = _random.Random(seed)
    reservoir = []
    for i, item in enumerate(iterable):
        if i < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = item
    return reservoir
