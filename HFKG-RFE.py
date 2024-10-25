from dataloader import *
import pickle
import os
import copy
import logging
from kge_model import KGEModel
from torch import optim
import torch.nn.functional as F


class Server(object):
    def __init__(self, args, nrelation):
        self.args = args
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        #这行代码计算了关系嵌入向量初始化的范围 embedding_range。
        # 参数 args.gamma 和 args.epsilon 是模型的一些超参数，用于控制关系嵌入向量初始化范围的大小。args.hidden_dim 是模型中嵌入向量的维度。
        if args.model in ['ComplEx']:
            self.rel_embed = torch.zeros(nrelation, args.hidden_dim*2).to(args.gpu).requires_grad_()
            #如果模型类型是 'ComplEx'，则创建一个形状为 (nrelation, args.hidden_dim2) 的全零张量 self.rel_embed，
            # 用于存储关系嵌入向量。nrelation 是关系的数量，args.hidden_dim2 是每个关系嵌入向量的维度。
            # 通过 .to(args.gpu) 将张量放置在指定的 GPU 上（如果使用了 GPU）。
            # 最后，通过 requires_grad_() 方法指定张量需要计算梯度，用于后续的模型训练和优化。
        else:
            self.rel_embed = torch.zeros(nrelation, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=self.rel_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()#这行代码使用均匀分布初始化关系嵌入向量 self.rel_embed。
            # 关系嵌入向量的值被随机采样自均匀分布，范围是从-embedding_range.item() 到 embedding_range.item()。
        )
        self.nrelation = nrelation

    def send_emb(self):
        return copy.deepcopy(self.rel_embed)

    def aggregation(self, clients, rel_update_weights):
        agg_rel_mask = rel_update_weights#agg_rel_mask 是一个与 rel_update_weights 维度相同的张量，它用于保存每个关系在三个客户端中的权重。
        agg_rel_mask[rel_update_weights != 0] = 1
        #将 rel_update_weights 中非零元素对应的位置标记为1，即将关系在三个客户端中存在的标记为1，不存在的标记为0。

        rel_w_sum = torch.sum(agg_rel_mask, dim=0)#对 agg_rel_mask 沿着维度0（关系维度）求和，得到每个关系在三个客户端中出现的次数。
        rel_w = agg_rel_mask / rel_w_sum#将每个关系在三个客户端中出现的次数进行归一化，得到每个关系的权重。
        rel_w[torch.isnan(rel_w)] = 0#由于某些关系在三个客户端中都不存在，可能导致归一化时得到 NaN（Not a Number）。
        # 这里将这些 NaN 设置为0，表示这些关系的权重为0。
        if self.args.model in ['ComplEx']:
            update_rel_embed = torch.zeros(self.nrelation, self.args.hidden_dim * 2).to(self.args.gpu)
        else:
            update_rel_embed = torch.zeros(self.nrelation, self.args.hidden_dim).to(self.args.gpu)
        for i, client in enumerate(clients):
            local_rel_embed = client.rel_embed.clone().detach()
            client_similarity_score = F.cosine_similarity(update_rel_embed, local_rel_embed, dim=1)
            # 将余弦相似度限定在[1, 1.5]范围内
            client_similarity_score = torch.clamp(client_similarity_score, min=1.0, max=1.1)
            weight = client_similarity_score.view(client_similarity_score.shape[0], 1)

            update_rel_embed += local_rel_embed  * rel_w[i].reshape(-1, 1)* weight


            #local_rel_embed = client.rel_embed.clone().detach()
            #update_rel_embed += local_rel_embed * rel_w[i].reshape(-1, 1)
            # 这行代码将当前客户端的局部关系嵌入向量 local_rel_embed 与其对应的关系权重 rel_w[i] 相乘，
            # 并加到 update_rel_embed 中。
            # 由于 local_rel_embed 的形状为 (self.nrelation, self.args.hidden_dim)，
            # 而 rel_w[i] 的形状为 (self.nrelation,)，所以通过 rel_w[i].reshape(-1, 1) 将其转换为形状为 (self.nrelation, 1)，这样两个张量可以进行逐元素乘法。
        self.rel_embed = copy.deepcopy(update_rel_embed).requires_grad_()#self.rel_embed为E


class Client(object):
    def __init__(self, args, client_id, data, train_dataloader,
                 valid_dataloader, test_dataloader, ent_embed):
        self.args = args
        self.data = data
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.ent_embed = ent_embed

        self.client_id = client_id

        self.score_local = []
        self.score_global = []

        self.kge_model = KGEModel(args, args.model)
        self.rel_embed = None





    def __len__(self):
        return len(self.train_dataloader.dataset)

    def client_update(self):
        optimizer = optim.Adam([{'params': self.rel_embed},
                                {'params': self.ent_embed}], lr=self.args.lr)

        #当 loss.backward() 被调用时，PyTorch会自动计算梯度，并将梯度信息存储在 self.rel_embed 和 self.ent_embed 这些张量的 .grad 属性中。
        # 优化器在调用 optimizer.step() 时使用这些梯度来更新模型参数。
        losses = []
        #if (self.rel_embed is not None):prox
            #self.fixed_relation_embedding = nn.Parameter(self.rel_embed.clone().cuda(), requires_grad=False)
        if (self.rel_embed is not None):
            self.old_relation_embedding = nn.Parameter(self.rel_embed.clone().cuda(), requires_grad=False)
            self.fixed_relation_embedding = nn.Parameter(self.rel_embed.cuda(), requires_grad=True)



        for i in range(self.args.local_epoch):
            for batch in self.train_dataloader:
                positive_sample, negative_sample, sample_idx = batch

                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)

                negative_score = self.kge_model((positive_sample, negative_sample),
                                                 self.rel_embed, self.ent_embed)

                negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

                positive_score = self.kge_model(positive_sample,
                                                self.rel_embed, self.ent_embed, neg=False)

                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()

                loss = (positive_sample_loss + negative_sample_loss) / 2


                #entity_regularization = 0.5 * self.args.mu * (
                 #       (self.rel_embed - self.fixed_relation_embedding).norm(p=2) ** 2)

                #entity_regularization = entity_regularization.mean()
                #loss=loss+entity_regularization

                #sim = nn.CosineSimilarity(dim=-1)
                #simLocal = sim(self.rel_embed, self.old_relation_embedding).mean() / self.args.mu_temperature
                #simGlobal = sim(self.rel_embed, self.fixed_relation_embedding).mean() / self.args.mu_temperature
                #simLocal = torch.exp(simLocal)
                #simGlobal = torch.exp(simGlobal)
                #contrastive_loss = -self.args.mu_contrastive * torch.log(simGlobal / (simGlobal + simLocal))
                #loss = loss + contrastive_loss

                #dist = nn.PairwiseDistance()
                #simLocal = dist(self.rel_embed, self.old_relation_embedding).mean() / self.args.mu_temperature
                #simGlobal = dist(self.rel_embed, self.fixed_relation_embedding).mean() / self.args.mu_temperature
                #simLocal = torch.exp(simLocal)
                #simGlobal = torch.exp(simGlobal)
                #contrastive_loss = self.args.mu_contrastive * torch.log(simGlobal / (simGlobal + simLocal))
                #loss=loss+contrastive_loss
                triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                contrastive_loss = 0.5 * self.args.mu_contrastive * triplet_loss(self.rel_embed ,self.fixed_relation_embedding, self.old_relation_embedding)
                loss=loss+contrastive_loss




                optimizer.zero_grad()
                loss.backward()#是PyTorch中用于计算梯度的方法。
                optimizer.step()

                losses.append(loss.item())

        return np.mean(losses)

    def client_eval(self, istest=False):
        if istest:
            dataloader = self.test_dataloader
        else:
            dataloader = self.valid_dataloader

        results = ddict(float)
        for batch in dataloader:
            triplets, labels = batch
            triplets, labels = triplets.to(self.args.gpu), labels.to(self.args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = self.kge_model((triplets, None),
                                   self.rel_embed, self.ent_embed)#得分
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            #这行代码创建一个张量 b_range，其元素从 0 开始递增，范围是从 0 到 pred.size()[0] - 1，
            # 其中 pred.size()[0] 是 pred 张量的行数（样本数）。device=self.args.gpu 表示将该张量放置在 GPU 上，
            # 如果设置了 GPU，否则将放置在 CPU 上。
            target_pred = pred[b_range, tail_idx]
            #这行代码将 pred 张量中的目标实体（tail）对应的预测得分提取出来，
            # 并存储在 target_pred 中。b_range 是预测得分张量的行索引，tail_idx 是目标实体的索引，这样通过索引的方式可以获取目标实体的预测得分。
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            #这行代码根据 labels 张量中的标签信息（True/False）来处理 pred 张量中的预测得分。
            # 如果 labels 中某个位置是 True（1），则说明对应样本的目标实体是正例，将该位置对应的预测得分设置为一个较大的负数，即 -10000000。
            # 这样，在后续计算排名时，这些预测得分将排在最后，不会影响对正确排名的计算。
            pred[b_range, tail_idx] = target_pred
            #这行代码将之前提取的目标实体的预测得分 target_pred 赋值回 pred 张量中的相应位置，以恢复原始的预测得分。

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]#由于排名是从 0 开始的，所以最后再加上 1，使排名从 1 开始。

            ranks = ranks.float()
            count = torch.numel(ranks)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 3, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in results.items():
            if k != 'count':
                results[k] /= results['count']

        return results


class FedR(object):
    def __init__(self, args, all_data):
        self.args = args

        train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
            self.rel_freq_mat, ent_embed_list, nrelation = get_all_clients(all_data, args)

        self.args.nrelation = nrelation # question

        # client
        self.num_clients = len(train_dataloader_list)
        self.clients = [
            Client(args, i, all_data[i], train_dataloader_list[i], valid_dataloader_list[i],
                   test_dataloader_list[i], ent_embed_list[i]) for i in range(self.num_clients)
        ]

        self.server = Server(args, nrelation)


        self.total_test_data_size = sum([len(client.test_dataloader.dataset) for client in self.clients])
        self.test_eval_weights = [len(client.test_dataloader.dataset) / self.total_test_data_size for client in self.clients]

        self.total_valid_data_size = sum([len(client.valid_dataloader.dataset) for client in self.clients])
        self.valid_eval_weights = [len(client.valid_dataloader.dataset) / self.total_valid_data_size for client in self.clients]

    def write_training_loss(self, loss, e):
        self.args.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.args.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.args.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.args.writer.add_scalar("evaluation/hits3", results['hits@3'], e)
        self.args.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, e):
        state = {'rel_embed': self.server.rel_embed,
                 'ent_embed': [client.ent_embed for client in self.clients]}
        # delete previous checkpoint
        for filename in os.listdir(self.args.state_dir):
            if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.args.state_dir, filename)):
                os.remove(os.path.join(self.args.state_dir, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_epoch):
        os.rename(os.path.join(self.args.state_dir, self.args.name + '.' + str(best_epoch) + '.ckpt'),
                  os.path.join(self.args.state_dir, self.args.name + '.best'))

    def send_emb(self):
        for k, client in enumerate(self.clients):
            client.rel_embed = self.server.send_emb()

    def train(self):
        best_epoch = 0
        best_mrr = 0
        bad_count = 0

        mrr_plot_result = []
        loss_plot_result = []

        for num_round in range(self.args.max_round):#3
            n_sample = max(round(self.args.fraction * self.num_clients), 1)#4
            sample_set = np.random.choice(self.num_clients, n_sample, replace=False)#5

            self.send_emb()#将服务器relation向量传到客户机中
            round_loss = 0
            for k in iter(sample_set):
                client_loss = self.clients[k].client_update()#7
                round_loss += client_loss
            round_loss /= n_sample
            self.server.aggregation(self.clients, self.rel_freq_mat)#8执行一个函数 aggregation()，该函数可能是用于在服务器端聚合从客户端接收到的更新，以更新全局模型参数。



            logging.info('round: {} | loss: {:.4f}'.format(num_round, np.mean(round_loss)))
            self.write_training_loss(np.mean(round_loss), num_round)

            loss_plot_result.append(np.mean(round_loss))

            if num_round % self.args.check_per_round == 0 and num_round != 0:
                eval_res = self.evaluate()#该函数用于评估当前轮次模型在验证集上的性能，并返回评估结果。
                self.write_evaluation_result(eval_res, num_round)

                if eval_res['mrr'] > best_mrr:#判断当前轮次的 MRR 是否优于最佳 MRR，如果是，则更新最佳 MRR 和最佳轮次 best_mrr 和 best_epoch。
                    best_mrr = eval_res['mrr']
                    best_epoch = num_round
                    logging.info('best model | mrr {:.4f}'.format(best_mrr))
                    self.save_checkpoint(num_round)
                    bad_count = 0
                else:
                    bad_count += 1
                    logging.info('best model is at round {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_epoch, best_mrr, bad_count))

                mrr_plot_result.append(eval_res['mrr'])

            # 检查是否达到早停止条件。self.args.early_stop_patience 是设定的早停止容忍度，如果连续 bad_count 轮模型性能没有提升，则触发早停止。
            if bad_count >= self.args.early_stop_patience:
                logging.info('early stop at round {}'.format(num_round))

                loss_file_name = 'loss/' + self.args.name + '_loss.pkl'
                with open(loss_file_name, 'wb') as fp:
                    pickle.dump(loss_plot_result, fp)

                mrr_file_name = 'loss/' + self.args.name + '_mrr.pkl'
                with open(mrr_file_name, 'wb') as fp:
                    pickle.dump(mrr_plot_result, fp)

                break

        logging.info('finish training')
        logging.info('save best model')
        self.save_model(best_epoch)#保存获得最佳性能的模型参数。
        self.before_test_load()#在最后完成训练后，加载之前保存的最佳模型参数，并在测试集上进行最终的模型评估。
        self.evaluate(istest=True)#评估，该函数用于评估当前轮次模型在验证集上的性能，并返回评估结果。

    def before_test_load(self):#在最后完成训练后，加载之前保存的最佳模型参数，并在测试集上进行最终的模型评估。
        state = torch.load(os.path.join(self.args.state_dir, self.args.name + '.best'), map_location=self.args.gpu)
        self.server.rel_embed = state['rel_embed']
        for idx, client in enumerate(self.clients):
            client.ent_embed = state['ent_embed'][idx]

    def evaluate(self, istest=False):#评估，该函数用于评估当前轮次模型在验证集上的性能，并返回评估结果。
        self.send_emb()
        result = ddict(int)
        if istest:
            weights = self.test_eval_weights
        else:
            weights = self.valid_eval_weights
        for idx, client in enumerate(self.clients):
            client_res = client.client_eval(istest)

            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
                client_res['mrr'], client_res['hits@1'],
                client_res['hits@3'], client_res['hits@10']))

            for k, v in client_res.items():
                result[k] += v * weights[idx]

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
                     result['mrr'], result['hits@1'],
                     result['hits@3'], result['hits@10']))

        return result


