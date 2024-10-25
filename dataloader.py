import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples#训练三元组的数组
        self.nentity = nentity#（实体的数量）
        self.negative_sample_size = negative_sample_size#（负样本数目）作为输入参数。

        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.hr2t[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, idx

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])
        return positive_sample, negative_sample, sample_idx


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, rel_mask = None):
        self.len = len(triples)
        self.triple_set = all_true_triples#接收 triples（包含测试或验证三元组的数组）
        self.triples = triples
        self.nentity = nentity

        self.rel_mask = rel_mask
        #这是一个可选的参数，用于控制在特定客户端的验证和测试数据中是否考虑某些关系。如果提供了
        #rel_mask，它会被用于过滤掉在客户端的训练数据中存在的关系类型。这些类用于处理数据集，其中
        #TrainDataset主要用于训练数据，而TestDataset
        #主要用于验证和测试数据。它们提供了一种组织和访问数据的方式，以便在训练和评估模型时使用。

        self.hr2t_all = ddict(set)#该字典类似于 TrainDataset 中的 self.hr2t，用于存储从头实体-关系对 (h, r) 到尾实体 t 的映射。
        # 它是一个 defaultdict，对于每个 (h, r) 键，值是一个包含整个数据集中所有尾实体 t 的数组。
        for h, r, t in all_true_triples:#包含整个数据集中所有正确三元组的数组
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):#
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = self.hr2t_all[(head, relation)]
        trp_label = self.get_label(label)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        y = np.zeros([self.nentity], dtype=np.float32)

        for e2 in label:
            y[e2] = 1.0

        return torch.FloatTensor(y)


def get_task_dataset(data, args):
    nentity = len(np.unique(data['train']['edge_index'].reshape(-1)))
    nrelation = len(np.unique(data['train']['edge_type']))

    train_triples = np.stack((data['train']['edge_index'][0],
                              data['train']['edge_type'],
                              data['train']['edge_index'][1])).T

    valid_triples = np.stack((data['valid']['edge_index'][0],
                              data['valid']['edge_type'],
                              data['valid']['edge_index'][1])).T

    test_triples = np.stack((data['test']['edge_index'][0],
                             data['test']['edge_type'],
                             data['test']['edge_index'][1])).T

    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset(valid_triples, all_triples, nentity)
    test_dataset = TestDataset(test_triples, all_triples, nentity)

    return train_dataset, valid_dataset, test_dataset, nrelation, nentity


def get_all_clients(all_data, args):
    all_rel = np.array([], dtype=int)#数组 all_rel 用于存储所有客户端训练数据中的关系类型。
    for data in all_data:#遍历所有客户端的数据。
        all_rel = np.union1d(all_rel, data['train']['edge_type_ori']).reshape(-1)
        #将当前客户端的训练数据中的关系类型与之前收集的关系类型进行合并（去重），并更新 all_rel 数组。
    nrelation = len(all_rel) # all relations of training set in all clients#
    # 计算所有客户端训练数据中的不同关系类型数量，并将结果存储在变量 nrelation 中。

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []#初始化用于存储数据加载器

    ent_embed_list = []#初始化实体嵌入

    rel_freq_list = []
    #初始化关系频率的列表

    for data in tqdm(all_data): # in a client再次遍历所有客户端的数据，并为每个客户端构建数据加载器和相关数据。
        nentity = len(np.unique(data['train']['edge_index'])) # entities of training in a client计算当前客户端训练数据中的不同实体数量。

        train_triples = np.stack((data['train']['edge_index'][0],
                                  data['train']['edge_type_ori'],
                                  data['train']['edge_index'][1])).T

        valid_triples = np.stack((data['valid']['edge_index'][0],
                                  data['valid']['edge_type_ori'],
                                  data['valid']['edge_index'][1])).T

        test_triples = np.stack((data['test']['edge_index'][0],
                                 data['test']['edge_type_ori'],
                                 data['test']['edge_index'][1])).T

        client_mask_rel = np.setdiff1d(np.arange(nrelation),
                                       np.unique(data['train']['edge_type_ori'].reshape(-1)), assume_unique=True)
        #这段代码的作用是计算在整个数据集中存在的所有可能关系类型（np.arange(nrelation)）中，
        # 但在特定客户端的训练数据中缺失的关系类型。计算结果将存储在变量 client_mask_rel 中。(客户端中特有的数据集)


        all_triples = np.concatenate([train_triples, valid_triples, test_triples]) # in a client
        train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
        valid_dataset = TestDataset(valid_triples, all_triples, nentity, client_mask_rel)
        test_dataset = TestDataset(test_triples, all_triples, nentity, client_mask_rel)

        # dataloader数据划分
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_list.append(train_dataloader)

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        valid_dataloader_list.append(valid_dataloader)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_list.append(test_dataloader)

        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])

        '''use n of entity in train or all (train, valid, test)?'''
        if args.model in ['RotatE', 'ComplEx']:
            ent_embed = torch.zeros(nentity, args.hidden_dim*2).to(args.gpu).requires_grad_()
        else:
            ent_embed = torch.zeros(nentity, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=ent_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        ent_embed_list.append(ent_embed)

        rel_freq = torch.zeros(nrelation)
        for r in data['train']['edge_type_ori'].reshape(-1):
            rel_freq[r] += 1
        rel_freq_list.append(rel_freq)#根据客户端训练数据中的关系类型，统计每个关系类型在整个数据集中出现的频率，
        # 将结果存储在 rel_freq 中，并添加到 rel_freq_list 列表中。

    rel_freq_mat = torch.stack(rel_freq_list).to(args.gpu)#将关系频率列表转换为一个 PyTorch 张量 rel_freq_mat，并将其移到 GPU（如果可用）。

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
           rel_freq_mat, ent_embed_list, nrelation
