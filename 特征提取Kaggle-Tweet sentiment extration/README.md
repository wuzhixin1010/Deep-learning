# My first kaggle competition--Tweet sentiment extraction(Top 6%)
## Background
在这个比赛中，我们得到一个包含有特定情绪的tweet的数据集，我们将挑选出tweet中反映该情绪的部分(单词或短语)。因此，我们的训练集包含三个列，包括tweet、sentiment和selected_text(word或phrase)。在测试集中，我们将预测selected_text。
## Dataloader
处理数据的时候就是要把文本转化成token，然后把情感和推特文本拼接成一个句子，还要找出训练集中selected_text对应token在推特文本对应token中的开始和结束位置，然后要把每个句子padding成最大长度。
## Model
整个模型包含三个模型的融合，三个模型都是Roberta_base，Roberta模型的输入包含tokenize后的句子，判断padding部分的mask和判断两个句子分隔的token_type_ids。
+ 模型一：使用roberta模型的最后两层输出，合并大小为(batch_size,MAX_LEN,2*hidden_state)。将模型的输出进行dropout(rate=0.3)之后连接一个全连接层，全连接层的大小为(2*hidden_state, 2),得到每个位置的作为起始位置和结束位置的两个分数，大小皆为(batch_size,MAX_LEN).
+ 模型二：将roberta模型的倒数四层输出做平均处理，大小为(batch_size,MAX_LEN,hidden_size)，将模型的输出进行dropout(rate=0.5)之后连接一个全连接层，大小为(hidden_size, 2)，输出的两个元素分别代表答案在推特文本中的起始位置和结束位置，大小皆为(batch_size,MAX_LEN).
+ 模型三：使用Roberta最后一层学到的hidden_state，大小为（batch_size,sequence_length,768),Dropout(rate=0.3),然后连接kernel_size为128的1D卷积层，对768维的hidden_state进行卷积，stride为1，目的是提取出能代表文本起始与结束位置的有效信息。1D-cnn输入的channel和输出的channel大小皆为句子长度，意味着对每个位置的信息都进行筛选。将cnn的输出进行Maxpool之后连接一个全连接层，全连接层输出的大小为(batch_size,sequence_length,2),这样每个位置都得到一个代表起始位置和结束位置的分数。
## Loss and optimizer
+ Loss:模型的输出为每个位置作为起始位置和结束位置的分数，将其softmax得到的分数，训练的时候与已知的target一起，用CrossEntropy作为损失函数，总的Loss为起始位置与结束位置的Loss总和。
+ optimizer:AdamW,linear_schedule_with_warmup
## Evaluation
使用Jaccard similarity作为评估函数
```python
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

```
## 附：
+ [我的kaggle](https://www.kaggle.com/wuzhixin)
+ [这个比赛Sota](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477)
