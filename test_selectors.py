from rlflow.selectors import FifoScheme, UniformSampleScheme, DensitySampleScheme

SCHEME_SIZE = 100

def test_selector(selector):
    batch_size = 11
    res = selector.sample(batch_size)
    assert res is None
    for i in range(SCHEME_SIZE//2):
        selector.add(i)

    res = selector.sample(batch_size)
    assert len(res) == batch_size

    if hasattr(selector, "update_priorities"):
        selector.update_priorities(res, np.ones(batch_size, dtype=np.float32))
    selector.remove(res[0])

def test_all():
    test_selector(FifoScheme())
    test_selector(UniformSampleScheme(SCHEME_SIZE))
    test_selector(DensitySampleScheme(SCHEME_SIZE,0.9))

test_all()
