from data_store.selectors import FifoScheme, UniformSampleScheme, PrioritizedSampleScheme
def test_selector(selector):
    selector.add(0)
    selector.add(1)
    res = selector.sample()
    print(res)
    selector.update_priorities(0, 0.5)
    selector.remove(res)

def test_all():
    test_selector(FifoScheme())
    test_selector(UniformSampleScheme())
    test_selector(PrioritizedSampleScheme(10,0.9))

test_all()
