import time
import os
import shutil
from Dataload.dataload_SST_binary import *
from DataUtils.alphabet import CreateAlphabet
import torch
from DataUtils.batch_iterator import *
from DataUtils.Embed import Embed
from DataUtils.common import paddingkey
from model.Text_Classification import *


def get_learning_algorithm(config):
    """
    :param config:  config
    :return:   optimizer algorithm
    """
    algorithm = None
    if config.adam is True:
        algorithm = "Adam"
    elif config.sgd is True:
        algorithm = "SGD"
    print("the learning algorithm is {}.".format(algorithm))
    return algorithm


def get_params(config, alphabet):
    """
    :param config: config
    :param alphabet: alphabet dict
    :return:
    """
    # get algorithm
    config.learning_algorithm = get_learning_algorithm(config)

    # save best model path
    config.save_best_model_path = config.save_best_model_dir
    if config.test is False:
        if os.path.exists(config.save_best_model_path):
            shutil.rmtree(config.save_best_model_path)
    # get params
    config.embed_num = alphabet.word_alphabet.vocab_size  # word number
    config.label_num = alphabet.label_alphabet.vocab_size  # label number
    config.paddingId = alphabet.word_paddingId
    config.alphabet = alphabet
    print("embed_num : {},class_num : {}".format(config.embed_num,config.label_num))
    print("PaddingID {}".format(config.paddingId))


def save_dict2file(dict,path):
    """

    :param dict:  dict
    :param path:  path to sasve dict
    :return:
    """
    print("Saving dictionary.........")
    if os.path.exists(path):
        print("path {} is exist,deleted.".format(path))
    file = open(path,encoding="utf-8",mode="w")  #'w'是以文件写入的方式打开文件
    for word, index in dict.items():
        file.write(str(word) + "\t" + str(index) + "\n")
    file.close()
    print("Save dictionary has been finished.........")


def save_dictionary(config):
    """
    :param config: config
    :return:
    """
    if config.save_dict is True:
        if os.path.exists(config.dict_directory):
            shutil.rmtree(config.dict_directory)
        if not os.path.isdir(config.dict_directory):
            os.makedirs(config.dict_directory)

        config.word_dict_path = "/".join([config.dict_directory, config.word_dict])
        config.label_dict_path = "/".join([config.dict_directory, config.label_dict])
        print("word_dict_directory ：{}".format(config.word_dict_path))
        print("label_dict_directory : {} ".format(config.label_dict_path))
        save_dict2file(config.alphabet.word_alphabet.words2id, config.word_dict_path)
        save_dict2file(config.alphabet.label_alphabet.words2id, config.label_dict_path)
        # copy to mu lu
        print("copy dictionaconry to {}".format(config.save_dir))
        shutil.copytree(config.dict_directory, "/".join([config.save_dir, config.dict_directory]))






def preprocessing(config):
    """
    :param config:
     :return:
     """
    print("processing data............")
    # read file
    data_loader = DataLoader(path=[config.train_file, config.dev_file, config.test_file], shuffle=True, config=config)
    train_data, dev_data, test_data = data_loader.dataload()
    print("train sentence {},dev sentence {},test sentence {}.".format(len(train_data), len(dev_data), len(test_data)))
    data_dict = {"train_data": train_data, "dev_data": dev_data, "test_data": test_data}

    if config.save_pkl:
        torch.save(obj=data_dict, f=os.path.join(config.pkl_directory, config.pkl_data))

    # create the alphabet
    alphabet = None
    if config.embed_finetune is False:
        alphabet = CreateAlphabet(min_freq=config.min_freq, train_data=train_data, dev_data=dev_data, test_data=test_data, config=config)
        alphabet.build_vocab()
    if config.embed_finetune is True:
        alphabet = CreateAlphabet(min_freq=config.min_freq, train_data=train_data, config=config)
        alphabet.build_vocab()
    alphabet_dict = {"alphabet": alphabet}
    if config.save_pkl:
        torch.save(obj=alphabet_dict, f=os.path.join(config.pkl_directory, config.pkl_alphabet))

    # create iterator
    create_iter = Iterators(batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                            data=[train_data, dev_data, test_data], operator=alphabet, config=config)
    train_iter, dev_iter, test_iter = create_iter.createIterator()
    iter_dict = {"train_iter": train_iter, "dev_iter": dev_iter, "test_iter": test_iter}
    if config.save_pkl:
        torch.save(obj=iter_dict, f=os.path.join(config.pkl_directory, config.pkl_iter))
    return train_iter, dev_iter, test_iter, alphabet


def pre_embed(config, alphabet):
    """
    :param config:
    :param alphabet:
    :return:
    """
    print("............................")
    pretrain_embed = None
    embed_types = ""
    if config.pretrained_embed and config.zeros:
        embed_types = "zeros"
    elif config.pretrained_embed and config.avg:
        embed_types = "avg"
    elif config.pretrained_embed and config.uniform:
        embed_types = "uniform"
    elif config.pretrained_embed and config.nnembed:
        embed_types = "nn"
    if config.pretrained_embed is True:
        p = Embed(path=config.pretrained_embed_file, words_dict=alphabet.word_alphabet.id2words, embed_type=embed_types,
                  pad=paddingkey)
        pretrain_embed = p.get_embed()

        embed_dict = {"pretrain_embed": pretrain_embed}
        # pcl.save(obj=embed_dict, path=os.path.join(config.pkl_directory, config.pkl_embed))
        torch.save(obj=embed_dict, f=os.path.join(config.pkl_directory, config.pkl_embed))

    return pretrain_embed


def load_model(config):
    """

    :param config: config
    :return: nn model
    """
    print("********************************************************")
    model = Text_Classification(config)
    if config.use_cuda is True:
       model = model.cuda()
    return model

def load_data(config):
    """
    :param config: config
    :return: batch data iterator  and alphabet
    """
    print("load data for process or pkl data")
    alphabet=None
    start_time = time.time()
    if(config.train is True)and(config.process is True):
        print('PROCESS DATA:')
    if os.path.exists(config.pkl_directory): shutil.rmtree(config.pkl_directory)
    if not os.path.isdir(config.pkl_directory):os.makedirs(config.pkl_directory)
    train_iter, dev_iter, test_iter, alphabet = preprocessing(config)
    config.pretrained_weight = pre_embed(config=config, alphabet=alphabet)
    end_time = time.time()
    print("All Data/Alphabet/Iterator Use Time {:.4}".format(end_time - start_time))
    print("***************************************")
    return train_iter, dev_iter, test_iter, alphabet



