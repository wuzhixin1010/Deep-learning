#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size,batch_first=True)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.paddingidx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(num_embeddings=len(target_vocab.char2id), embedding_dim=char_embedding_size, padding_idx=self.paddingidx)
        self.target_vocab = target_vocab

        

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch) 传入的应该是一个词，length是词长，然后对每个字母进行词嵌入
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        #先找出每个字母的 词嵌入
        #然后丢进lstm，得到h_n, c_n
        #对于每一个timestep，我们通过output_proj计算这个时间的score,(v_char)
        #把score丢进softmax，得到下一个字母的概率分布。
        x_emb = self.decoderCharEmb(input)  #(length, batch, embedding_size)
        x_emb = x_emb.permute((1,0,2))

        x_lstm_out, dec_hidden = self.charDecoder(x_emb, hx=dec_hidden)#(batch, length, hidden_size)

        s_t = self.char_output_projection(x_lstm_out)#scores:permute前(batch, length, v_char)



        return s_t, dec_hidden






        
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        s_t, dec_hidden = self.forward(char_sequence[:-1], dec_hidden) #scores:permute前(batch, length, v_char)


        ###以下记录一下crossentropy：
        ###crossentropy: 其实包含了softmax+负log, 所以input需要包含class（放在倒二个维度）,他会将每一个class的分数丢到softmax里面重新得到一个分数。
        ###              然后我们需要对每个字母都进行loss计算，也就是这里单词的长度length，放在input的最后一个维度
        ### 我们从forward得到的分数是每个batch（word）每个位置的单词 属于词汇表任意一个词的分数，维度是(batch, length, v_char)，
        ### 要作为crossentropy的input，我们要丢进去的是每个batch（word）的那么多个class（词汇表的总字符数）里面，该字符对应于每个class的分数，这里每个单词有k个字符，input（batch,v_char,length)
        ###target:每个batch（单词）,每个位置对应的正确字符只有一个，所以target（batch, length) 在答案出score为1，其他都为0，所以不必有一个class的维度来记录分数了
        ###output：每个batch的总的loss（或者平均，用参数reduction来设置。（batch)
        loss_func = nn.CrossEntropyLoss(ignore_index=self.paddingidx, reduction='sum')
        input = s_t.permute((0,2,1)) #(batch,vocab,length-1)
        target = char_sequence[1:] #(batch,length-1)
        loss = loss_func(input=input,target=target.transpose(1,0))#(把每个batch的loss加起来了好像，标量)？？？？？？？？

        return loss








        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        
        ### END YOUR CODE
        #将初始状态丢进lstm里面，得到结果集（batch, length, hidden_size)
        #proj计算output的分数
        #softmax计算可能性
        #判断是否是end，将结果存入

        output_char = []
        batch_word = []


        batch_size = len(initialStates[1][0,:,0])
        input_list = [self.target_vocab.start_of_word]*batch_size
        input = torch.tensor(input_list, device=device).unsqueeze(1)

        input_emb = self.decoderCharEmb(input)#(batch,length=1,embedding_size)

        for char in range(max_length):
            output,decoder_state = self.charDecoder(input_emb, hx=initialStates)#(batch,length=1,hidden)
            s_t = self.char_output_projection(output)#(batch, length=1,v_char)
            softmax_score = torch.softmax(s_t,dim=2)#(batch, length=1,v_char)
            predict_char = torch.max(softmax_score, dim=2)[1].squeeze(1)#squeeze前(batch,1)
            output_char.append(predict_char)#(length)
            input = predict_char.unsqueeze(1)
            input_emb = self.decoderCharEmb(input)
            initialStates = decoder_state

        #将每一个位置的字母，一个batch一个batch分出来，判断是否是end，然后转化为字符再存入
        for i_char in range(max_length):
            #对于每个字符
            char_indx = output_char[i_char]
            for i_batch in range(len(char_indx)):
                #对于每个batch的该字符
                i_batch_char_indx = char_indx[i_batch]
                #判断是否是end token
                if i_batch_char_indx != self.target_vocab.end_of_word:
                    tensor_to_int = int(i_batch_char_indx)
                    trans_to_char = self.target_vocab.id2char[tensor_to_int]
                    if len(batch_word) == batch_size:
                        batch_word[i_batch] += trans_to_char
                    else:
                        batch_word.append(trans_to_char)
        print(batch_word)

        return batch_word


