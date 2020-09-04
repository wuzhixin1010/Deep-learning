#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21

    ### YOUR CODE HERE for part 1f
    ### TODO:
    ###     Perform necessary padding to the sentences in the batch similar to the pad_sents()
    ###     method below using the padding character from the arguments. You should ensure all
    ###     sentences have the same number of words and each word has the same number of
    ###     characters.
    ###     Set padding words to a `max_word_length` sized vector of padding characters.
    ###
    ###     You should NOT use the method `pad_sents()` below because of the way it handles
    ###     padding and unknown words.
    #先找出句子的最大长度，并记录每个句子的长度以及顺便填充单词空缺处
    sent_len_list = []
    word_len_list = []
    max_sent_len = 0
    sents_padded = []
    for sentence in sents:
        cur_sentence_len = len(sentence)
        sent_len_list.append(cur_sentence_len)
        if max_sent_len < cur_sentence_len:
            max_sent_len = cur_sentence_len
        #将已有的单词填充到最大长度
        for word in sentence:
            cur_word_len = len(word)

            if cur_word_len < max_word_length:
                extend_cha = [char_pad_token] * (max_word_length - cur_word_len)
                word.extend(extend_cha)
    ###对于句子长度不足的情况，
    ###我们先存下每个完整单词的表示形式即insert_char_list,
    ###然后视需要填充的单词个数进行填充。
    ###extent会把列表中的每个对象(单词列表—insert_char_list)分别加到目标列表中
    for index, sentence in enumerate(sents):
        insert_char_list = [char_pad_token] * max_word_length
        extend_word_list = [insert_char_list] * (max_sent_len-sent_len_list[index])
        sentence.extend(extend_word_list)
        sents_padded.append(sentence)



    ### END YOUR CODE

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []
    max_len = 0
    len_list = []

    ### COPY OVER YOUR CODE FROM ASSIGNMENT 4

    for sentence in sents:
        cur_sent_len = len(sentence)
        if cur_sent_len > max_len:
            max_len = cur_sent_len
        len_list.append(cur_sent_len)
    for sent_id, sentence in enumerate(sents):
       # sentence.append(pad_token * (max_len - len_list[sent_id]))
        sentence.extend([pad_token] * (max_len - len_list[sent_id]))
        sents_padded.append(sentence)


    ### END YOUR CODE FROM ASSIGNMENT 4

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
