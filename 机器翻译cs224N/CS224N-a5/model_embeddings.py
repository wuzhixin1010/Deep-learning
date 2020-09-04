#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn
from cnn import CNN
from vocab import Vocab,VocabEntry
from highway import Highway


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
#

# End "do not change"
DROPOUT_RATE = 0.3
e_word = 256
kernel_size = 5
embed_size = 50
m_word = 21   #每个单词的字符个数

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
                                    (batch_size, max_sentence_length, max_word_length)
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.pad_token_idx = vocab.char2id['<pad>']
        #tgt_pad_token_idx = vocab.tgt.char2id['<pad>']
        self.embed_size = embed_size
        self.embedding = nn.Embedding((len(vocab.char_list) + 4),self.embed_size, padding_idx=self.pad_token_idx )
        #self.tgt_embedding = nn.Embedding(len(vocab.tgt.char_list), self.embed_size, padding_idx=tgt_pad_token_idx)
        ### (batch, m_word=max_word_length=len(char_list), embed_size=e_char )
        self.Drop_layer = nn.Dropout(DROPOUT_RATE)


        ### END YOUR CODE

    def forward(self, padded_sents):#( batch_size, sentence_length, max_word_length)
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape  where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (batch_size, sentence_length, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        ### YOUR CODE HERE for part 1j


        ### 通过embedding将padded_sents 转化为 x_reshape
        ### 注意，输入的padded_sents 是句子，我们要对每个词进行embedding

        ### 将 x_reshape(batch, input_channel=e_char, m_word) 放入cnn，

        # 再放入high_way,然后dropout得到x_embedding

        ###以下
        x_word = torch.zeros((len(padded_sents[:,0,0]), len(padded_sents[0,:,0]), e_word))
        #(batch_size, sentence_length, embed_size)

        for n_word in range(len(padded_sents[0,:,0])):
            padded_word = padded_sents[:,n_word,:]
            x_reshape = self.embedding(padded_word).permute((0,2,1))  #(batch, embed_size=e_char,m_word=max_word_length=len(char_list) )

            cnn_net = CNN(self.embed_size, e_word, k=kernel_size, m_word=m_word) ###(batch,output_channel=e_word)
            x_conv_out = cnn_net(x_reshape)
            x_conv_out = torch.squeeze(x_conv_out, dim=2)  # #(batch,output_channel=e_word)

            x_highway_layer = Highway(x_conv_out)
            x_highway = x_highway_layer(x_conv_out)# (batch,e_word)

            x_word[:, n_word, :] = self.Drop_layer(x_highway)  #(batch_size, sentence_length, embed_size)
        return x_word                        # #(batch_size, sentence_length, embed_size)







        ### END YOUR CODE

