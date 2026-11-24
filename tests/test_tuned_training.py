import contextlib
import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_interest_lifecycle import config
from model import Att_FM_model,create_model_FM
from trainer import model_train
from utils import Data_Train,Data_Val,Data_Test


class TunedTrainerTests(unittest.TestCase):
    def test_cpu_update_checkpoint_and_disabled_train_inference(self):
        args=config(True)
        args.item_num=128
        for key,value in dict(epochs=2,device='cpu',metric_ks=[5,10,20],
            Loss_Alpha=.2,Loss_Beta=.4,optimizer='Adam',optimizer_device='cpu',
            lr=.001,weight_decay=0.,grad_clip_value=.5,decay_step=15,gamma=.5,
            eval_interval=1,patience=2,selection_metric='NDCG@20',
            random_seed=1997,report_train_metrics=0,batch_size=2,dataset='synthetic').items():
            setattr(args,key,value)
        train={1:[1,2,3],2:[2,3,4]}
        val={1:[4],2:[5]}
        test={1:[5],2:[6]}
        with tempfile.TemporaryDirectory() as directory:
            args.log_file=directory+'/'
            model=Att_FM_model(create_model_FM(args),args)
            with contextlib.redirect_stdout(io.StringIO()):
                selected,metrics=model_train(Data_Train(train,args).get_pytorch_dataloaders(),
                    Data_Val(train,val,args).get_pytorch_dataloaders(),
                    Data_Test(train,val,test,args).get_pytorch_dataloaders(),
                    model,args,logging.getLogger('test'))
            self.assertEqual(len(metrics),6)
            self.assertIsNotNone(selected)
            self.assertEqual(len(list((Path(directory)/'synthetic').glob('*.json'))),1)
            self.assertEqual(len(list((Path(directory)/'synthetic').glob('*.pt'))),1)


if __name__=='__main__':
    unittest.main()
