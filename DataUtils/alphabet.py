
import collections
from DataUtils.common import *


class CreateAlphabet:

    """
    Class:      Create_Alphabet
    Function:   Build Alphabet By Alphabet Class
    Notice:     The Class Need To Change So That Complete All Of Tasks
    """
    def __init__(self, min_freq=1, train_data=None, dev_data=None, test_data=None, config=None):

        # minimum vocab size
        self.min_freq = min_freq
        self.config = config
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        # storage word and label
        self.word_state = collections.OrderedDict()
        self.label_state = collections.OrderedDict()

        # unk and pad
        self.word_state[unkkey] = self.min_freq  # 确保在词表过滤的时候“<unk>不会被过滤掉”
        self.word_state[paddingkey] = self.min_freq

        # word and label Alphabet
        self.word_alphabet = Alphabet(min_freq=self.min_freq)
        self.label_alphabet = Alphabet()

        # unk key
        self.word_unkId = 0
        self.label_unkId = 0

        # padding key
        self.word_paddingId = 0
        self.label_paddingId = 0

    def _build_data(self, train_data=None, dev_data=None, test_data=None):
        """

        :param dev_data:
        :param test_data:
        :return:
        """
        assert train_data is not None, "The Train Data Is Not Allow Empty. "
        datasets = []
        datasets.extend(train_data)  # 此时train_data的类型，extend()只能接受一个参数（列表）
        print("the length of train data {}".format(len(datasets)))
        if dev_data is not None:
            print("tne length of dev data {}".format(len(dev_data)))
            datasets.extend(dev_data)
        if test_data is not None:
            print("the length of test data {} ". format(len(test_data)))
            datasets.extend(test_data)
        print("the length of data that create Alphabet {} ".format(len(datasets)))
        return datasets

    def build_vocab(self):
        train_data = self.train_data
        dev_data = self.dev_data
        test_data = self.test_data
        print("Start Build Vocab......")
        datasets = self._build_data(train_data=train_data, dev_data=dev_data, test_data=test_data)

        for index, data in enumerate(datasets):
            # word
            for word in data.words:
                if word not in self.word_state:
                    self.word_state[word] = 1
                else:
                    self.word_state[word] += 1

            # label
            for label in data.labels:
                if label not in self.label_state:
                    self.label_state[label] = 1
                else:
                    self.label_state[label] += 1

        # Create id2words and words2id by the Alphabet Class
        self.word_alphabet.initialWord2idAndId2Word(self.word_state)
        self.label_alphabet.initialWord2idAndId2Word(self.label_state)

        # unkId and paddingId
        self.word_unkId = self.word_alphabet.from_string(unkkey)
        self.word_paddingId = self.word_alphabet.from_string(paddingkey)
        self.label_paddingId = self.label_alphabet.from_string(paddingkey)

        # fix the vocab
        self.word_alphabet.set_fixed_flag(True)
        self.label_alphabet.set_fixed_flag(True)


class Alphabet:

    def __init__(self, min_freq=1):
        self.id2words = []
        self.words2id = collections.OrderedDict()
        self.vocab_size = 0
        self.min_freq = min_freq
        self.max_cap = 1e8
        self.fixed_vocab = False

    def initialWord2idAndId2Word(self, data):
        """
        :param data:
        :return:
        """
        for key in data:  # data {"I": 12, "want": 14, "go": 33}
            if data[key] >= self.min_freq:
                self.from_string(key)
        self.set_fixed_flag(True)

    def set_fixed_flag(self, bfixed):
        """

        :param self:
        :param bfixed:
        :return:
        """
        self.fixed_vocab = bfixed
        if (not self.fixed_vocab) and (self.vocab_size >= self.max_cap):
            self.fixed_vocab = True

    def from_string(self, string):
        """

        :param self:
        :param string:
        :return:
        """
        if string in self.words2id:
            return self.words2id[string]
        else:
            if not self.fixed_vocab:
                newid = self.vocab_size
                self.id2words.append(string)
                self.words2id[string] = newid
                self.vocab_size += 1
                if self.vocab_size >= self.max_cap:
                    self.fixed_vocab = True
                return newid
            else:
                return -1

    def from_id(self, qid, defineStr= ""):
        """

        :param self:
        :param qid:
        :param defineStr:
        :return:
        """
        if int(qid) < 0 or self.vocab_size <= qid:
            return defineStr
        else:
            return self.id2words[qid]








