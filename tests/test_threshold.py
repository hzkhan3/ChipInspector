from scripts.tune_threshold import THRESHOLDS, costs

def test_costs_count_misses_and_false_alarms():
    probs, labels = [0.1, 0.4, 0.6, 0.9], [0, 1, 0, 1]
    at = lambda t, ratio: costs(probs, labels, ratio)[list(THRESHOLDS).index(t)]  # noqa: E731
    assert at(0.5, 1) == 2   # misses the 0.4 defect + flags the 0.6 good chip
    assert at(0.3, 1) == 1   # catches both defects, one false alarm
    assert at(0.5, 10) == 11  # a miss now costs 10
