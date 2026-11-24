"""构造下一物品预测样本，严格区分当前标签与可见历史。

生命周期关闭时保留原双张量接口；开启时额外返回完整前缀。
近期 tokens 左补零，完整 history 在 collate 时右补零，两者不能混淆。
"""
import torch.utils.data as data_utils
import torch
from torch.nn.utils.rnn import pad_sequence


def lifecycle_collate(batch):
    """Keep the complete observed prefix; right-pad independently of max_len."""
    tokens, labels, histories = zip(*batch)
    return torch.stack(tokens), torch.stack(labels), pad_sequence(histories, batch_first=True)


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len, lifecycle=False):
        self.id2seq = id2seq
        self.max_len = max_len
        self.lifecycle = lifecycle

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        labels = [seq[-1]]
        # 完整历史必须排除本次标签；只截断 tokens，不能截断长期兴趣记忆。
        history = seq[:-1]
        tokens = history
        tokens = tokens[-self.max_len:]
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        result = torch.LongTensor(tokens), torch.LongTensor(labels)
        return result + (torch.LongTensor(history),) if self.lifecycle else result

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.lifecycle = bool(getattr(args, 'interest_lifecycle', False))
        self.split_onebyone()

    def split_onebyone(self):
        """逐前缀展开训练样本；每个用户第一条行为只作为后续历史。"""
        self.id_seq = {}
        self.id_seq_user = {}
        idx = 0
        for user_temp, seq_temp in self.u2seq.items():
            for star in range(len(seq_temp)-1):
                self.id_seq[idx] = seq_temp[:star+2]
                self.id_seq_user[idx] = user_temp
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len, self.lifecycle)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                     collate_fn=lifecycle_collate if self.lifecycle else None)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, lifecycle=False):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.lifecycle = lifecycle

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        history = seq
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        result = torch.LongTensor(seq), torch.LongTensor(answer)
        return result + (torch.LongTensor(history),) if self.lifecycle else result


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.lifecycle = bool(getattr(args, 'interest_lifecycle', False))
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len, self.lifecycle)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                                          collate_fn=lifecycle_collate if self.lifecycle else None)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len, lifecycle=False):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.lifecycle = lifecycle

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_add[user]
        # 测试时验证交互已经可见；测试答案仍不允许进入历史。
        history = seq
        # seq = self.u2seq[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        result = torch.LongTensor(seq), torch.LongTensor(answer)
        return result + (torch.LongTensor(history),) if self.lifecycle else result


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.lifecycle = bool(getattr(args, 'interest_lifecycle', False))
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len, self.lifecycle)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                                          collate_fn=lifecycle_collate if self.lifecycle else None)
        return dataloader


class CHLSDataset(data_utils.Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_temp = self.data[index]
        seq = data_temp[:-1]
        answer = [data_temp[-1]]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_CHLS():
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.data = data

    def get_pytorch_dataloaders(self):
        dataset = CHLSDataset(self.data, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader
