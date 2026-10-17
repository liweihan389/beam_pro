from beamic_pro.stream import AdaptiveRate


def test_sustained_underperformance_lowers_rate_without_increasing_backlog():
    rate = AdaptiveRate(20)
    start = rate.last_check
    assert rate.observe(5, 250, start + 4) is None
    assert rate.observe(5, 250, start + 8) == 15
    assert rate.requested == 20
    rate.configure(30)
    assert rate.current == rate.requested == 30

