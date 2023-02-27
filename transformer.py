import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os,sys,time,re
#import unicodedata,jieba
# 1. loads data
# 2. preprocesses data -> dataset
# 3. tools
# 3.1 generates position embedding
# 3.2 create mask. (a. padding, b. decoder)
# 3.3 scaled_dot_product_attention
# 4. builds model
# 4.1 MultiheadAttention
# 4.2 EncoderLayer
# 4.3 DecoderLayer
# 4.4 EncoderModel
# 4.5 DecoderModel
# 4.6 Transformer
# 5. optimizer & loss
# 6. train step -> train
# 7. Evaluate and Visualize

filepath='D:\\codes\\deeplearning\\transformer\\yue-eng\\can.txt'

def parse_data(filename):
    lines=open(filename,encoding='UTF-8').read().strip().split('\n')#逐行读取
    setence_pairs=[line.split('\t') for line in lines]
    preprocessed_setence_pairs=[
        (en,other) for en,other in setence_pairs]
    return zip(*preprocessed_setence_pairs)
en_dataset,other_dataset=parse_data(filepath)
en_dataset=list(en_dataset)
other_dataset=list(other_dataset)
print(other_dataset)
print(en_dataset)
en_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((' '.join(e) for e in en_dataset), target_vocab_size=2**13)
pt_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(( o for o in other_dataset), target_vocab_size=2**13)
#pt_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((' '.join(e) for e in other_dataset), target_vocab_size=2**13)

sample_string = "Transformer is awesome."

tokenized_string = en_tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

origin_string = en_tokenizer.decode(tokenized_string)
print('The original string is {}'.format(origin_string))

assert origin_string == sample_string

for token in tokenized_string:
    print('{} --> "{}"'.format(token, en_tokenizer.decode([token])))


buffer_size = 20000
batch_size = 50
max_length = 30

def encode_to_subword(pt_sentence, en_sentence):
    pt_sequence = [pt_tokenizer.vocab_size] \
    + pt_tokenizer.encode(pt_sentence.numpy()) \
    + [pt_tokenizer.vocab_size + 1]
    en_sequence = [en_tokenizer.vocab_size] \
    + en_tokenizer.encode(en_sentence.numpy()) \
    + [en_tokenizer.vocab_size + 1]
    return pt_sequence, en_sequence

def filter_by_max_length(pt, en):
    return tf.logical_and(tf.size(pt) <= max_length,
                          tf.size(en) <= max_length)

def tf_encode_to_subword(pt_sentence, en_sentence):
    return tf.py_function(encode_to_subword,
                          [pt_sentence, en_sentence],
                          [tf.int64, tf.int64])


#cn = [pt_tokenizer.encode(line) for line in other_dataset]

cn= [pt_tokenizer.encode(' '.join(line))for line in other_dataset]
en= [en_tokenizer.encode(' '.join(line))for line in en_dataset]

en_ = []
for i in en :
    en_.append([en_tokenizer.vocab_size]+list(i)+[en_tokenizer.vocab_size+1])
cn_ = []
for i in cn :
    cn_.append([pt_tokenizer.vocab_size]+list(i)+[pt_tokenizer.vocab_size+1])

en_text=tf.keras.preprocessing.sequence.pad_sequences(
    en_, maxlen=max_length, dtype='int64', padding='post',
    value=0.0)
cn_text=tf.keras.preprocessing.sequence.pad_sequences(
    cn_, maxlen=max_length, dtype='int64', padding='post',
    value=0.0)

en_batch = tf.data.Dataset.from_tensor_slices(en_text)
cn_batch = tf.data.Dataset.from_tensor_slices(cn_text)

en_batch  = en_batch.padded_batch(30)
cn_batch = cn_batch.padded_batch(30)

input_train,input_test,output_train,output_test=train_test_split(en_text,cn_text,test_size=0.2)
epochs=60
def make_dataset(input_tensor,output_tensor,batch_size,epochs,shuffle):
    dataset=tf.data.Dataset.from_tensor_slices(
        (input_tensor,output_tensor))
    if shuffle:
        dataset=dataset.shuffle(30000)
    dataset=dataset.repeat(epochs).batch(batch_size,drop_remainder=True)
    return dataset

batch_size=140

train_dataset=make_dataset(input_train,output_train,batch_size,epochs,True)
test_dataset=make_dataset(input_test,output_test,batch_size,epochs,False)

# en_spa_filepath='D:\\codes\\deeplearning\\transformer\\yue-eng\\can.txt'
#
#
# #solve the spanish
# def unicode_to_ascii(s):#NFD normalize：把重音符号分开；过滤掉重音
#     return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
#
#
#
#
# #把标点符号和词语分开，去掉多余空格
# def preprocess_sentnece(s):
#     s=unicode_to_ascii(s.lower().strip())
#     s=re.sub(r"([?.!,¿])",r" \1 ",s)#匹配到标点后前后各加一个空格
#     s=re.sub(r'[" "]+'," ",s)#多个空格的话只保留一个
#     s=re.sub(r'[^a-zA-Z?.!,¿]'," ",s)#所有除了字母和标点外的符号都替换成空格
#     s=s.rstrip().strip()#去掉前后空格
#     s='<start> '+s+' <end>'#增加特殊字符
#     return s
#
# def preprocess_sentence_chinese(s):
#     s=s.strip()
#     s=re.sub(r"([！。，？「」——])",r" \1 ",s)
#     s = re.sub(r'[" "]+', " ", s)  # 多个空格的话只保留一个
#     s = s.rstrip().strip()  # 去掉前后空格
#     seg_list=jieba.cut(s)  # 搜索引擎模式,利用jieba分词
#     s=" ".join(list(seg_list))#利用空格将分好的词拼接起来
#     s = '<start> ' + s + ' <end>'  # 增加特殊字符
#     return s
#
# def parse_data(filename):
#     lines=open(filename,encoding='UTF-8').read().strip().split('\n')#逐行读取
#     setence_pairs=[line.split('\t') for line in lines]
#     preprocessed_setence_pairs=[
#         (preprocess_sentnece(en),preprocess_sentence_chinese(sp)) for en,sp in setence_pairs]
#     return zip(*preprocessed_setence_pairs)
# en_dataset,sp_dataset=parse_data(en_spa_filepath)
# print(sp_dataset[-1])
#
#
# #词转成id词库
# def tokenizer(lang):#词表没有限制，没有黑名单，用空格分隔
#     lang_tokenizer=keras.preprocessing.text.Tokenizer(num_words=None,filters='',split=" ")
#     lang_tokenizer.fit_on_texts(lang)#训练
#     tensor=lang_tokenizer.texts_to_sequences(lang)#转成id
#     tensor=keras.preprocessing.sequence.pad_sequences(tensor,padding='post')#长度不足的句子以零填充
#     return tensor,lang_tokenizer
#
# input_tensor,input_tokenizer=tokenizer(en_dataset)
# output_tensor,output_tokenizer=tokenizer(sp_dataset)
#
#
# def max_length(tensor):
#     return max(len(t) for t in tensor)
#
# max_length_input=max_length(input_tensor)
# max_length_output=max_length(output_tensor)
# print('sentence length max:',max_length_input,max_length_output)
# input_train,input_test,output_train,output_test=train_test_split(input_tensor,output_tensor,test_size=0.2)
#
# def make_dataset(input_tensor,output_tensor,batch_size,epochs,shuffle):
#     dataset=tf.data.Dataset.from_tensor_slices(
#         (input_tensor,output_tensor))
#     if shuffle:
#         dataset=dataset.shuffle(30000)
#     dataset=dataset.repeat(epochs).batch(batch_size,drop_remainder=True)
#     return dataset
#
# batch_size=70
# epochs=40
# train_dataset=make_dataset(input_train,output_train,batch_size,epochs,True)
# test_dataset=make_dataset(input_test,output_test,batch_size,epochs,False)





























# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# pos.shape: [sentence_length, 1]
# i.shape  : [1, d_model]
# result.shape: [sentence_length, d_model]
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000,
                               (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def get_position_embedding(sentence_length, d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # sines.shape: [sentence_length, d_model / 2]
    # cosines.shape: [sentence_length, d_model / 2]
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    # position_embedding.shape: [sentence_length, d_model]
    position_embedding = np.concatenate([sines, cosines], axis=-1)
    # position_embedding.shape: [1, sentence_length, d_model]
    position_embedding = position_embedding[np.newaxis, ...]

    return tf.cast(position_embedding, dtype=tf.float32)


position_embedding = get_position_embedding(50, 512)
print(position_embedding.shape)


def plot_position_embedding(position_embedding):
    plt.pcolormesh(position_embedding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


#plot_position_embedding(position_embedding)

# 1. padding mask, 2. look ahead

# batch_data.shape: [batch_size, seq_len]
def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)


# attention_weights.shape: [3,3]
# [[1, 0, 0],
#  [4, 5, 0],
#  [7, 8, 9]]
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)

create_look_ahead_mask(3)


def scaled_dot_product_attention(q, k, v, mask):
    """
    Args:
    - q: shape == (..., seq_len_q, depth)
    - k: shape == (..., seq_len_k, depth)
    - v: shape == (..., seq_len_v, depth_v)
    - seq_len_k == seq_len_v
    - mask: shape == (..., seq_len_q, seq_len_k)
    Returns:
    - output: weighted sum
    - attention_weights: weights of attention
    """

    # matmul_qk.shape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        # 使得在softmax后值趋近于0
        scaled_attention_logits += (mask * -1e9)

    # attention_weights.shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)

    # output.shape: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def print_scaled_dot_product_attention(q, k, v):
    temp_out, temp_att = scaled_dot_product_attention(q, k, v, None)
    print("Attention weights are:")
    print(temp_att)
    print("Output is:")
    print(temp_out)


class MultiHeadAttention(keras.layers.Layer):
    """
    理论上:
    x -> Wq0 -> q0
    x -> Wk0 -> k0
    x -> Wv0 -> v0

    实战中:
    q -> Wq0 -> q0
    k -> Wk0 -> k0
    v -> Wv0 -> v0

    实战中技巧：
    q -> Wq -> Q -> split -> q0, q1, q2...
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)

        self.dense = keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # x -> (batch_size, num_heads, seq_len, depth)

        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)  # q.shape: (batch_size, seq_len_q, d_model)
        k = self.WK(k)  # k.shape: (batch_size, seq_len_k, d_model)
        v = self.WV(v)  # v.shape: (batch_size, seq_len_v, d_model)

        # q.shape: (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # k.shape: (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # v.shape: (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_outputs, attention_weights = \
            scaled_dot_product_attention(q, k, v, mask)

        # scaled_attention_outputs.shape: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm=[0, 2, 1, 3])
        # concat_attention.shape: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size, -1, self.d_model))

        # output.shape : (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 256))  # (batch_size, seq_len_q, dim)
output, attn = temp_mha(y, y, y, mask=None)
print(output.shape)
print(attn.shape)

def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])

sample_ffn = feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape


class EncoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout
      -> feed_forward -> add & normalize & dropout
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, training, encoder_padding_mask):
        # x.shape          : (batch_size, seq_len, dim=d_model)
        # attn_output.shape: (batch_size, seq_len, d_model)
        # out1.shape       : (batch_size, seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, encoder_padding_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        # ffn_output.shape: (batch_size, seq_len, d_model)
        # out2.shape      : (batch_size, seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2


sample_encoder_layer = EncoderLayer(512, 8, 2048)
sample_input = tf.random.uniform((64, 50, 512))
sample_output = sample_encoder_layer(sample_input, False, None)
print(sample_output.shape)


class DecoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout -> out1
    out1 , encoding_outputs -> attention -> add & normalize & dropout -> out2
    out2 -> ffn -> add & normalize & dropout -> out3
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_norm3 = keras.layers.LayerNormalization(
            epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, encoding_outputs, training,
             decoder_mask, encoder_decoder_padding_mask):
        # decoder_mask: 由look_ahead_mask和decoder_padding_mask合并而来

        # x.shape: (batch_size, target_seq_len, d_model)
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)

        # attn1, out1.shape : (batch_size, target_seq_len, d_model)
        attn1, attn_weights1 = self.mha1(x, x, x, decoder_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layer_norm1(attn1 + x)

        # attn2, out2.shape : (batch_size, target_seq_len, d_model)
        attn2, attn_weights2 = self.mha2(
            out1, encoding_outputs, encoding_outputs,
            encoder_decoder_padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layer_norm2(attn2 + out1)

        # ffn_output, out3.shape: (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(ffn_output + out2)

        return out3, attn_weights1, attn_weights2


sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_input = tf.random.uniform((64, 60, 512))
sample_decoder_output, sample_decoder_attn_weights1, sample_decoder_attn_weights2 = sample_decoder_layer(sample_decoder_input, sample_output, False, None, None)

print(sample_decoder_output.shape)
print(sample_decoder_attn_weights1.shape)
print(sample_decoder_attn_weights2.shape)


class EncoderModel(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = keras.layers.Embedding(input_vocab_size,
                                                self.d_model)
        # position_embedding.shape: (1, max_length, d_model)
        self.position_embedding = get_position_embedding(max_length,self.d_model)

        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, training, encoder_padding_mask):
        # x.shape: (batch_size, input_seq_len)
        input_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            input_seq_len, self.max_length,
            "input_seq_len should be less or equal to self.max_length")

        # x.shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :input_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training,encoder_padding_mask)

        # x.shape: (batch_size, input_seq_len, d_model)
        return x





class DecoderModel(keras.layers.Layer):
    def __init__(self, num_layers, target_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(DecoderModel, self).__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        self.embedding = keras.layers.Embedding(target_vocab_size,
                                                d_model)
        self.position_embedding = get_position_embedding(max_length,
                                                         d_model)

        self.dropout = keras.layers.Dropout(rate)
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, encoding_outputs, training,
             decoder_mask, encoder_decoder_padding_mask):
        # x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            output_seq_len, self.max_length,
            "output_seq_len should be less or equal to self.max_length")

        attention_weights = {}

        # x.shape: (batch_size, output_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :output_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn1, attn2 = self.decoder_layers[i](
                x, encoding_outputs, training,
                decoder_mask, encoder_decoder_padding_mask)
            attention_weights[
                'decoder_layer{}_att1'.format(i + 1)] = attn1
            attention_weights[
                'decoder_layer{}_att2'.format(i + 1)] = attn2
        # x.shape: (batch_size, output_seq_len, d_model)
        return x, attention_weights





class Transformer(keras.Model):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size,
                 max_length, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_model = EncoderModel(
            num_layers, input_vocab_size, max_length,
            d_model, num_heads, dff, rate)

        self.decoder_model = DecoderModel(
            num_layers, target_vocab_size, max_length,
            d_model, num_heads, dff, rate)

        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, encoder_padding_mask,
             decoder_mask, encoder_decoder_padding_mask):
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(
            inp, training, encoder_padding_mask)

        # decoding_outputs.shape: (batch_size, output_seq_len, d_model)
        decoding_outputs, attention_weights = self.decoder_model(
            tar, encoding_outputs, training,
            decoder_mask, encoder_decoder_padding_mask)

        # predictions.shape: (batch_size, output_seq_len, target_vocab_size)
        predictions = self.final_layer(decoding_outputs)

        return predictions, attention_weights





# 1. initializes model.
# 2. define loss, optimizer, learning_rate schedule
# 3. train_step
# 4. train process

num_layers = 3
d_model = 128
dff = 512
num_heads = 8


input_vocab_size = en_tokenizer.vocab_size + 2
target_vocab_size = pt_tokenizer.vocab_size + 2
# input_vocab_size=len(input_tokenizer.word_index)+1
# target_vocab_size=len(output_tokenizer.word_index)+1
#print('vocab_size:',input_vocab_size,output_vocab_size)
dropout_rate = 0.05

transformer = Transformer(num_layers,
                          input_vocab_size,
                          target_vocab_size,
                          max_length,
                          d_model, num_heads, dff, dropout_rate)


class CustomizedSchedule(
    keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomizedSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))

        arg3 = tf.math.rsqrt(self.d_model)

        return arg3 * tf.math.minimum(arg1, arg2)


learning_rate = CustomizedSchedule(d_model)
optimizer = keras.optimizers.Adam(learning_rate,
                                  beta_1=0.9,
                                  beta_2=0.98,
                                  epsilon=1e-9)

temp_learning_rate_schedule = CustomizedSchedule(d_model)

loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def create_masks(inp, tar):
    """
    Encoder:
      - encoder_padding_mask (self attention of EncoderLayer)
    Decoder:
      - look_ahead_mask (self attention of DecoderLayer)
      - encoder_decoder_padding_mask (encoder-decoder attention of DecoderLayer)
      - decoder_padding_mask (self attention of DecoderLayer)
    """
    encoder_padding_mask = create_padding_mask(inp)
    encoder_decoder_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)
    decoder_mask = tf.maximum(decoder_padding_mask,
                              look_ahead_mask)

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask

temp_inp=next(iter(en_batch))
temp_tar=next(iter(cn_batch))
print('1:',temp_inp.shape)
print("2:",temp_tar.shape)
create_masks(temp_inp, temp_tar)

train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask \
        = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True,encoder_padding_mask, decoder_mask,encoder_decoder_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


epochs = 20

for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(),
                train_accuracy.result()))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time take for 1 epoch: {} secs\n'.format(
        time.time() - start))

"""
eg: A B C D -> E F G H.
Train: A B C D, E F G -> F G H
Eval:  A B C D -> E
       A B C D, E -> F
       A B C D, E F -> G
       A B C D, E F G -> H
"""


def evaluate(inp_sentence):
    input_id_sentence = [en_tokenizer.vocab_size] \
                        + en_tokenizer.encode(inp_sentence) + [en_tokenizer.vocab_size + 1]
    # encoder_input.shape: (1, input_sentence_length)
    encoder_input = tf.expand_dims(input_id_sentence, 0)

    # decoder_input.shape: (1, 1)
    decoder_input = tf.expand_dims([pt_tokenizer.vocab_size], 0)

    for i in range(max_length):
        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(encoder_input, decoder_input)
        # predictions.shape: (batch_size, output_target_len, target_vocab_size)
        predictions, attention_weights = transformer(encoder_input,decoder_input,False,encoder_padding_mask,decoder_mask, encoder_decoder_padding_mask)
        # predictions.shape: (batch_size, target_vocab_size)
        predictions = predictions[:, -1, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1),
                               tf.int32)

        if tf.equal(predicted_id, pt_tokenizer.vocab_size + 1):
            return tf.squeeze(decoder_input, axis=0), attention_weights

        decoder_input = tf.concat([decoder_input, [predicted_id]],
                                  axis=-1)
    return tf.squeeze(decoder_input, axis=0), attention_weights


def plot_encoder_decoder_attention(attention, input_sentence,
                                   result, layer_name):
    fig = plt.figure(figsize=(16, 8))

    input_id_sentence = en_tokenizer.encode(input_sentence)

    # attention.shape: (num_heads, tar_len, input_len)
    attention = tf.squeeze(attention[layer_name], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        ax.matshow(attention[head][:-1, :])

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(input_id_sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [en_tokenizer.decode([i]) for i in input_id_sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)
        ax.set_yticklabels(
            [pt_tokenizer.decode([i]) for i in result if i < pt_tokenizer.vocab_size],
            fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()


def translate(input_sentence, layer_name=''):
    result, attention_weights = evaluate(input_sentence)

    predicted_sentence = pt_tokenizer.decode(
        [i for i in result if i < pt_tokenizer.vocab_size])

    print("Input: {}".format(input_sentence))
    print("Predicted translation: {}".format(predicted_sentence))

    if layer_name:
        plot_encoder_decoder_attention(attention_weights, input_sentence,
                                       result, layer_name)

translate('i love you')
translate('I like you')