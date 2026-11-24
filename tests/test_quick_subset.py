import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quick_compare_ml100k import select_users


class QuickSubsetTests(unittest.TestCase):
    def test_fixed_stratified_user_sample(self):
        train={user:list(range(user+20)) for user in range(943)}
        selected=select_users(train,150,1997)
        self.assertEqual(len(selected),150)
        self.assertEqual(len(set(selected)),150)
        self.assertEqual(selected,select_users(train,150,1997))
        ordered=sorted(train,key=lambda u:(len(train[u]),u))
        for group in range(10):
            bucket=set(ordered[943*group//10:943*(group+1)//10])
            self.assertEqual(len(bucket.intersection(selected)),15)


if __name__=='__main__':
    unittest.main()
