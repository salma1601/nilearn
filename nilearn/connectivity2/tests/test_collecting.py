from collections import Counter
from collecting import crescendo_replacement, combination_replacement


def test_crescendo_replacement():
    elements = range(10)
    replacement_tuples = [(x, -x) for x in [6, 3, 8]]
    new_lists = crescendo_replacement(elements, replacement_tuples)
    assert(new_lists[0] == range(6) + [-6] + range(7, 10))
    assert(new_lists[1] == range(3) + [-3, 4, 5, -6] + range(7, 10))
    assert(new_lists[2] == range(3) + [-3, 4, 5, -6, 7, -8] + range(9, 10))


def test_combination_replacement():
    # Simple case
    elements = [1, 3, 5, 6, 8, 9]
    replacement_tuples = [(x, -x) for x in [6, 3, 8]]
    new_dict = combination_replacement(elements, replacement_tuples)
    assert(set(new_dict.keys()) == set([0, 1, 2, 3]))
    assert(new_dict[0] == [[1, 3, 5, 6, 8, 9]])
    print new_dict[1]
    assert(new_dict[1] == [[1, 3, 5, -6, 8, 9], [1, -3, 5, 6, 8, 9],
                           [1, 3, 5, 6, -8, 9]])
    assert(new_dict[2] == [[1, -3, 5, -6, 8, 9], [1, 3, 5, -6, -8, 9],
                           [1, -3, 5, 6, -8, 9]])
    assert(new_dict[3] == [[1, -3, 5, -6, -8, 9]])

    # Border case: empty list
#    assert(shuffled_replacement([], replacement_tuples))

