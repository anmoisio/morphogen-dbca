#!/usr/bin/env python
import unittest
from utils.detokenisation import make_detokenise

class TestDetokenisation(unittest.TestCase):

    def test_make_detokenise(self):
        detokenise = make_detokenise('This: is a test.'.split(),
            ['This', ':', 'is', 'a', 'test', '.'])
        assert detokenise == {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3}

        detokenise = make_detokenise('This is a test'.split(),
            ['This', 'is', 'a', 'test'])
        assert detokenise == {0: 0, 1: 1, 2: 2, 3: 3}

        detokenise = make_detokenise(["This:", "is,", "you're", "test.'"],
            ["This", ":", "is", ",", "you", "'re", "test", ".", "'"])
        assert detokenise == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}

if __name__ == '__main__':
    unittest.main()
