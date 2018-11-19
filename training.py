from DataUtils.Optim import *
import torch.nn as nn
from DataUtils.utilss import *
import time
import random
import sys
from DataUtils.common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    def __init__(self, **kwargs):
        print("The Training Is Starting")
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.config = kwargs["config"]
        # self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        self.loss_function = nn.CrossEntropyLoss(size_average=True)
        print(self.optimizer)
        print(self.loss_function)
        self.best_score = Best_Result()
        self.train_iter_len = len(self.train_iter)

    def _get_model_args(self, batch_features):
        """
        :param batch_features: batch instance
        :return:
        """
        word = batch_features.word_features
        # mask = word > 0
        sentence_length = batch_features.sentence_length
        labels = batch_features.label_features
        batch_size = batch_features.batch_length
        return word, sentence_length, labels, batch_size

    def train(self):
        epochs = self.config.epochs
        for epoch in range(1, epochs + 1):
            print("\n## The {} epoch,All {} epochs ! ##".format(epoch, epochs))
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()
            steps = 1
            backword_count = 0
            self.optimizer.zero_grad()
            for batch_count, batch_features in enumerate(self.train_iter):  # train_iter里边是一个个batch特征()
                # print(batch_count)
                # print(batch_features)
                backword_count += 1
                word, sentence_length, labels, batch_size = self._get_model_args(batch_features)
                # print(word)
                # print(labels)
                logit = self.model(word, sentence_length, train=True)
                #                 # print(logit)
                loss = self.loss_function(logit, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    accuracy = self.getAcc(logit, labels, batch_size)
                    sys.stdout.write("\nbatch_count = [{}], loss is {:.6f}, [accuracy is {:.6f}]".format(batch_count + 1
                                                                                                         , loss.data[0],
                                                                                                         accuracy))
            end_time = time.time()
            print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
            self.eval(model=self.model, epoch=epoch, config=self.config)
            exit()

    def eval(self, model, epoch, config):
        """
        :param model: nn model
        :param epoch:  epoch
        :param config:  config
        :return:
        """
        eval_start_time = time.time()
        self.eval_batch(self.dev_iter, model, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))

        eval_start_time = time.time()
        self.eval_batch(self.test_iter, model, self.best_score, epoch, config,test=True)
        eval_end_time = time.time()
        print("test time {:.3f}".format(eval_end_time - eval_start_time))

    def eval_batch(self, data_iter, model, best_score, epoch, config, test=False):
        """
        :param data_iter:  eval batch data iterator
        :param model:  eval model
        :param best_score:
        :param epoch:
        :param config:
        :param test:  whether to test
        :return:
        """
        model.eval()
        corrects = 0
        size = 0
        loss = 0
        for batch_features in data_iter:
            word, sentence_length, labels, batch_size = self._get_model_args(batch_features)
            logit = self.model(word, sentence_length, train=False)
            loss += self.loss_function(logit, labels)
            size += batch_features.batch_length
            # print(size)
            corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()

        assert size is not 0, print("Error")
        accurcay = float(corrects) / size * 100.0
        average_loss = float(loss) / size

        test_flag = "Test"
        if test is False:
            print()
            test_flag = "Dev"
            best_score.current_dev_score = accurcay
            if accurcay >= best_score.best_dev_score:
                best_score.best_dev_score = accurcay
                best_score.best_epoch = epoch
                best_score.best_test = True
        if test is True and best_score.best_test is True:
            best_score.p = accurcay
        print("{} eval: average_loss = {:.6f},accuracy = {:.6f}".format(test_flag, average_loss, accurcay))
        if test is True:
            print("the current best dev accuracy: {:.6f},locate on {} epoch.".format(best_score.best_dev_score,
                                                                                     best_score.best_epoch))
            print("the current best test accuracy: accuracy = {:.6f}".format(best_score.p))
            best_score.best_test = False

    def getAcc(self, logit, target, batch_size):  # 这个主要对比预测结果和金标值，看预测和金标一样的有多少，算出准确率
        """
        :param logit:  model predict(output)
        :param target:  actual value
        :param batch_size:
        :return:+

        """
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        # print("logit", logit)
        # print(torch.max(logit, 1))
        # print(torch.max(logit, 1)[1].view(target.size()).data)
        # print(target.data)
        # print("aaa")
        # print((torch.max(logit, 1)[1].view(target.size()).data == target.data))
        # print((torch.max(logit, 1)[1].view(target.size()).data == target.data).sum())
        # exit()
        accuracy = float(corrects) / batch_size * 100.0
        return accuracy












