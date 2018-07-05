from os import listdir
from os.path import isfile, join
import jieba
import codecs
from langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import pickle
import random
import numpy as np 

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.preprocessing.text import Tokenizer
# from keras.layers.core import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout

# pickle and load stuff
def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()

def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff

fileFolder = './data/ChnSentiCorp_htl_ba_6000'
# get positive and negative files
dataBaseDirPos = join(fileFolder, 'pos/')
dataBaseDirNeg = join(fileFolder, 'neg/')
positiveFiles = [dataBaseDirPos + f for f in listdir(dataBaseDirPos) if isfile(join(dataBaseDirPos, f))]
negativeFiles = [dataBaseDirNeg + f for f in listdir(dataBaseDirNeg) if isfile(join(dataBaseDirNeg, f))]


# decode the sentence with 'GB2312', and convert the traditional chinese to simple chinese
# 将句子用‘GB2312’ 编码，并将繁体字转换为简体字
documents = []
for filename in positiveFiles:
    text = ""
    with codecs.open(filename, "rb") as doc_file:
        for line in doc_file:
            try:
                line = line.decode("GB2312")
            except:
                continue
            text+=Converter('zh-hans').convert(line)# Convert from traditional to simplified Chinese

            text = text.replace("\n", "")
            text = text.replace("\r", "")
    documents.append((text, "pos"))

for filename in negativeFiles:
    text = ""
    with codecs.open(filename, "rb") as doc_file:
        for line in doc_file:
            try:
                line = line.decode("GB2312")
            except:
                continue
            text+=Converter('zh-hans').convert(line)# Convert from traditional to simplified Chinese

            text = text.replace("\n", "")
            text = text.replace("\r", "")
    documents.append((text, "neg"))

# Uncomment those two lines to save/load the documents for later use since the step above takes a while
# 对数据进行保存或加载
# __pickleStuff("./data/chinese_sentiment_corpus.p", documents)
# documents = __loadStuff("./data/chinese_sentiment_corpus.p")
print(len(documents))
print(documents[4000])

# shuffle the data
# 洗牌
random.shuffle(documents)

# Tokenize only
# 准备数据和label，样本中包含stop word
# totalX = []
# totalY = [str(doc[1]) for doc in documents]
# for doc in documents:
#     seg_list = jieba.cut(doc[0], cut_all=False)
#     seg_list = list(seg_list)
#     totalX.append(seg_list)


# Switch to below code to experiment with removing stop words
# Tokenize and remove stop words  样本中删除stop word
totalX = []
totalY = [str(doc[1]) for doc in documents]
stopwords = [ line.rstrip() for line in codecs.open('./data/chinese_stop_words.txt',"r", encoding="utf-8") ]
for doc in documents:
    seg_list = jieba.cut(doc[0], cut_all=False)
    seg_list = list(seg_list)
    # Uncomment below code to experiment with removing stop words
    final =[]
    for seg in seg_list:
        if seg not in stopwords:
            final.append(seg)
    totalX.append(final)

h = sorted([len(sentence) for sentence in totalX])
maxLength = h[int(len(h) * 0.60)]
print("input maxlength:", maxLength)

# Words to number tokens, padding
# 将文本转为数字，将短的句子补充为长句子
# Keras Tokenizer expect the words tokens to be seperated by space 
totalX = [" ".join(wordslist) for wordslist in totalX]  
input_tokenizer = Tokenizer(30000) # Initial vocab size
input_tokenizer.fit_on_texts(totalX)
vocab_size = len(input_tokenizer.word_index) + 1
print("input vocab_size:",vocab_size)
totalX = np.array(pad_sequences(input_tokenizer.texts_to_sequences(totalX), maxlen=maxLength))
__pickleStuff("./data/input_tokenizer_chinese.p", input_tokenizer)

# output array
target_tokenizer = Tokenizer(3)
target_tokenizer.fit_on_texts(totalY)
print("output vocab_size:",len(target_tokenizer.word_index) + 1)
totalY = np.array(target_tokenizer.texts_to_sequences(totalY)) -1
totalY = totalY.reshape(totalY.shape[0])

# turn 0 and 1 to categories(one-hot vectors)
# label用One-hot编码
totalY = to_categorical(totalY, nb_classes=2)
output_dimen = totalY.shape[1]

# save data for later predition
target_reverse_word_index = {v: k for k, v in list(target_tokenizer.word_index.items())}
sentiment_tag = [target_reverse_word_index[1],target_reverse_word_index[2]] 
metaData = {"maxLength":maxLength,"vocab_size":vocab_size,"output_dimen":output_dimen,"sentiment_tag":sentiment_tag}
__pickleStuff("./data/meta_sentiment_chinese.p", metaData)


# build the model, train and save it
#The training data is logged to Tensorboard, we can look at it by cd into directory 

embedding_dim = 256

model = Sequential()

# vocab_size: 词汇表中单词的总数
# embedding_dim: 每个单词编码后输出的维度
# input_length: 一个句子中包含的单词的最大个数
model.add(Embedding(vocab_size, embedding_dim, input_length = maxLength))
# Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
# All the intermediate outputs are collected and then passed on to the second GRU layer.
# model.add(GRU(256, dropout=0.9, return_sequences=True))
model.add(GRU(256, return_sequences=True))
model.add(Dropout(0.9))
# Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
model.add(GRU(256))
model.add(Dropout(0.9))
# The output is then sent to a fully connected layer that would give us our final output_dim classes
model.add(Dense(output_dimen, activation='softmax'))
# We use the adam optimizer instead of standard SGD since it converges much faster
tbCallBack = TensorBoard(log_dir='./Graph/sentiment_chinese', histogram_freq=0,
                            write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(totalX, totalY, validation_split=0.1, batch_size=32, nb_epoch=10, verbose=1, callbacks=[tbCallBack])
model.save('./data/sentiment_chinese_model.HDF5')

print("Saved model!")