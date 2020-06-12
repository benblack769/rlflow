from data_store.selectors import FifoScheme, UniformSampleScheme, PrioritizedSampleScheme
def test_selector(selector):
    selector.add(0, None)
    selector.add(1, None)
    res = selector.sample()
    selector.remove(res)

def test_all():
    test_selector(FifoScheme())
    test_selector(UniformSampleScheme())
    #test_selector(PrioritizedSampleScheme(10,0.9))
