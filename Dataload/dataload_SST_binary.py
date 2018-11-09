import random
import sys
from Dataload.Instance import Instance
class DataLoader():
    def __init__(self, path, shuffle, config):
        """
        :param path:
        :param shuffle:
        :param config:
        :return:
        """
        print("Loading data:......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle

    def dataload(self):
        """

        :return:
        """
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path,list,), "path must be in list"#instanec()指定对象是某种特定的形式
        print('data path{}'.format(path))
        for data_id in range(len(path)):
            print("load data form{}".format(path[data_id]))
            insts = self._Load_Each_Data(path=path[data_id], shuffle=shuffle)
            if shuffle is True and data_id == 0:
                print("shuffle train data......")
                random.shuffle(insts)
            self.data_list.append(insts)#把三个文件的标签列表和单词列表都放在里边
            # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]



    def _Load_Each_Data(self,path=None,shuffle=False):
        """

        :param path:
        :param shuffle:
        :return:
        """
        assert path is not None, "the data is not allow empty"
        insts=[]
        now_lines = 0

        with open(path,encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split()
                inst = Instance()
                now_lines += 1
                if now_lines % 20 == 0:
                    # print("just only handle {} datas".format(now_lines))
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))

                label=line[0]
                word=line[1:]
                if label not in ["0", "1"]:
                    print("Error line:", "".join(line))
                    continue
                inst.words=word
                inst.labels.append(label)
                inst.words_size = len(inst.words)
                insts.append(inst)

                if len(insts) == self.max_count:
                    break


        return insts
