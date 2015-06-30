from ..collecting import crescendo_replacement


def test_crescendo_replacement():
    elements = range(10)
    replacement_tuples = [(x, -x) for x in [6, 3, 8]]
    new_lists = crescendo_replacement(elements, replacement_tuples)
    assert(new_lists[0] == range(6) + [-6] + range(7, 10))
    assert(new_lists[1] == range(3) + [-3, 4, 5, -6] + range(7, 10))
    assert(new_lists[2] == range(3) + [-3, 4, 5, -6, 7, -8] + range(9, 10))
