import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os,sys,time,re
import unicodedata
from sklearn.model_selection import train_test_split
"""
1.preprocessing data
2.build model
2.1 encoder
2.2 attention
2.3 decoder
2.4 loss optimizer,train
3.evaluation:given sentence,return translation
"""

en_spa_filepath='D:\\codes\\deeplearning\\transformer\\fra-eng\\fra1.txt'


#solve the spanish
def unicode_to_ascii(s):#NFD normalize：把重音符号分开；过滤掉重音
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')




#把标点符号和词语分开，去掉多余空格
def preprocess_sentnece(s):
    s=unicode_to_ascii(s.lower().strip())
    s=re.sub(r"([?.!,¿])",r" \1 ",s)#匹配到标点后前后各加一个空格
    s=re.sub(r'[" "]+'," ",s)#多个空格的话只保留一个
    s=re.sub(r'[^a-zA-Z?.!,¿]'," ",s)#所有除了字母和标点外的符号都替换成空格
    s=s.rstrip().strip()#去掉前后空格
    s='<start> '+s+' <end>'#增加特殊字符
    return s


def parse_data(filename):
    lines=open(filename,encoding='UTF-8').read().strip().split('\n')#逐行读取
    setence_pairs=[line.split('\t') for line in lines]
    preprocessed_setence_pairs=[
        (preprocess_sentnece(en),preprocess_sentnece(sp)) for en,sp in setence_pairs]
    return zip(*preprocessed_setence_pairs)

en_dataset,sp_dataset=parse_data(en_spa_filepath)


#词转成id词库
def tokenizer(lang):#词表没有限制，没有黑名单，用空格分隔
    lang_tokenizer=keras.preprocessing.text.Tokenizer(num_words=None,filters='',split=" ")
    lang_tokenizer.fit_on_texts(lang)#训练
    tensor=lang_tokenizer.texts_to_sequences(lang)#转成id
    tensor=keras.preprocessing.sequence.pad_sequences(tensor,padding='post')#长度不足的句子以零填充

    return tensor,lang_tokenizer

input_tensor,input_tokenizer=tokenizer(sp_dataset[:30000])
output_tensor,output_tokenizer=tokenizer(en_dataset[:30000])


def max_length(tensor):
    return max(len(t) for t in tensor)

max_length_input=max_length(input_tensor)
max_length_output=max_length(output_tensor)
print('sentence length max:',max_length_input,max_length_output)
input_train,input_test,output_train,output_test=train_test_split(input_tensor,output_tensor,test_size=0.2)

def make_dataset(input_tensor,output_tensor,batch_size,epochs,shuffle):
    dataset=tf.data.Dataset.from_tensor_slices(
        (input_tensor,output_tensor))
    if shuffle:
        dataset=dataset.shuffle(30000)
    dataset=dataset.repeat(epochs).batch(batch_size,drop_remainder=True)
    return dataset

batch_size=70
epochs=30

train_dataset=make_dataset(input_train,output_train,batch_size,epochs,True)
test_dataset=make_dataset(input_test,output_test,batch_size,epochs,False)

embedding_units=256
units=1024

#padding 用0 补全，故真实词表要加1为词表中的所有词数
input_vocab_size=len(input_tokenizer.word_index)+1
output_vocab_size=len(output_tokenizer.word_index)+1
print('vocab_size:',input_vocab_size,output_vocab_size)

class Encoder(keras.Model):
    def __init__(self,vocab_size,embedding_units,encoding_units,batch_size):
        super(Encoder, self).__init__()
        self.batch_size=batch_size
        self.encoding_units=encoding_units
        self.embeding=keras.layers.Embedding(vocab_size,embedding_units)
        self.gru=keras.layers.GRU(self.encoding_units,
                                  return_sequences=True,#encoder 输出
                                  return_state=True, #传递的中间序列h_i
                                  recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):#让类实例化的对象可以像函数一样调用
        x=self.embeding(x)
        output,state=self.gru(x,initial_state=hidden)

        return output,state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size,self.encoding_units))

for x,_ in train_dataset.take(1):
    pass

encoder=Encoder(input_vocab_size,embedding_units,units,batch_size)
sample_hidden=encoder.initialize_hidden_state()
sample_output,sample_hidden=encoder.call(x,sample_hidden)
print('encoder output:',sample_output.shape)
print('decoder hidden:',sample_hidden.shape)

class BahdanauAttention(keras.Model):
    def __init__(self,nn_units):
        super(BahdanauAttention, self).__init__()
        self.w1=keras.layers.Dense(nn_units)
        self.w2=keras.layers.Dense(nn_units)
        self.v=keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_output):
        #encoder output:(batch_size,length,units)
        #decoder hidden: (batch_size,units)
        decoder_hidden=tf.expand_dims(decoder_hidden,1)
        #before v:(batch_size,length,units)
        #after v:(batch_size,length,1)
        mid=tf.nn.tanh(self.w1(encoder_output)+self.w2(decoder_hidden))
        score=self.v(mid)
        # after:(batch_size,length,1)
        attention_weights=tf.nn.softmax(score,1)
        #(batch_size,length,units)
        context_vector=attention_weights*encoder_output
        #(batch_size,units)
        context_vector=tf.reduce_sum(context_vector,axis=1)

        return context_vector,attention_weights

attention_model=BahdanauAttention(nn_units=10)

context_vector,attention_weights=attention_model.call(sample_hidden,sample_output)

print(context_vector.shape)

class Decoder(keras.Model):
    def __init__(self,vocab_size,embedding_units,decoding_units,batch_size):
        super(Decoder, self).__init__()
        self.batch_size=batch_size
        self.decoding_units=decoding_units
        self.embedding=keras.layers.Embedding(vocab_size,embedding_units)
        self.gru=keras.layers.GRU(self.decoding_units,
                                  return_sequences=True,  # decoder 输出
                                  return_state=True,  # 传递的中间序列h_i
                                  recurrent_initializer='glorot_uniform')
        self.fc=keras.layers.Dense(vocab_size)
        self.attention=BahdanauAttention(self.decoding_units)

    def call(self, x,hidden,encoding_outputs):#单部预测
        #context_vector:(batch_size,units)
        context_vector, attention_weights=self.attention(hidden,encoding_outputs)
        # before embedding ：x:(batch_size,1)
        # after embeding:x (batch_size,1,embedding_units)
        x=self.embedding(x)
        #(batch_size,1,units)
        expand_context_vec=tf.expand_dims(context_vector,1)
        #(bs,1,embedding_units+units)
        combined_x=tf.concat([expand_context_vec,x],axis=-1)

        #output:(batchsize,1,decoding_units),state:(batch_size,decoding_units)
        output ,state=self.gru(combined_x)
        # output:(batchsize,decoding_units)
        output=tf.reshape(output,(-1,output.shape[2]))
        # output:(batchsize,vocab_size)
        output=self.fc(output)

        return output,state,attention_weights

decoder=Decoder(output_vocab_size,embedding_units,units,batch_size)




optimizer=keras.optimizers.Adam()
loss_object=keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')

#单步损失函数
def loss_func(real,pred):
    #padding 不参与损失函数计算
    mask=tf.math.logical_not(tf.math.equal(real,0))#padding 返回 0，否则返回1
    loss_=loss_object(real,pred)
    mask=tf.cast(mask,dtype=loss_.dtype)
    loss_*=mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(input,target,encoding_hidden):
    loss=0
    with tf.GradientTape() as tape:
        encoding_outputs,encoding_hidden=encoder(input,encoding_hidden)
        decoding_hidden=encoding_hidden

        #eg:<start> i am here <end>:进行4次循环，最后一个词不计算损失
        #1.<start>->i  2.(<start>) i->am 3.(<start> i) am -> here 4.(<start> i am) here -><end>

        for t in range(0,target.shape[1]-1):
            decoding_input=tf.expand_dims(target[:,t],1)
            pre,decoding_hidden,_=decoder(decoding_input,decoding_hidden,encoding_outputs)
            loss+=loss_func(target[:,t+1],pred=pre)
        batch_loss=loss/int(target.shape[0])
        variables=encoder.trainable_variables+decoder.trainable_variables
        grad=tape.gradient(loss,variables)
        optimizer.apply_gradients(zip(grad,variables))
        return batch_loss

epochs=10
steps_per_epoch=len(input_tensor)//batch_size

for epoch in range(epochs):
    start=time.time()

    embedding_hidden=encoder.initialize_hidden_state()
    total_loss=0

    for (batch,(input,target)) in enumerate(train_dataset.take(steps_per_epoch)):
        batch_loss=train_step(input,target,encoding_hidden=embedding_hidden)
        total_loss+=batch_loss

        if batch %100==0:
            print("epoch {} batch {} loss{:.3f}".format(epoch,batch,batch_loss.numpy()))
    print('time for 1 epoch{}'.format(time.time()-start))
    print('epoch {} loss{:.3f}'.format(epoch,total_loss/steps_per_epoch))


def evaluate(input_sentence):
    attention_matrix=np.zeros((max_length_input,max_length_output))
    input_sentence=preprocess_sentnece(input_sentence)

    inputs=[input_tokenizer.word_index[token] for token in input_sentence.split(" ")]
    inputs=keras.preprocessing.sequence.pad_sequences(
        [inputs],maxlen=max_length_input,padding='post')
    inputs=tf.convert_to_tensor(inputs)

    results=''#存放预测结果的空字符串
    encoding_hidden=tf.zeros((1,units))
    encoding_outputs,encoding_hidden=encoder(inputs,encoding_hidden)
    decoding_hidden=encoding_hidden

    #eg:<start>->A;A->B->C->D

    #decoding_input(1,1)
    decoding_input=tf.expand_dims([output_tokenizer.word_index['<start>']],0)
    for t in range(max_length_output):
        pred, decoding_hidden, attention_weights=decoder(decoding_input,decoding_hidden,encoding_outputs)

        #attention weight:(bs,input_length,1)->(1,length,1)
        attention_weights=tf.reshape(attention_weights,(-1,))
        #attention_matrix[t]=attention_weights.numpy()
        #pred:(batchsize=1,vocab_size)
        pred_id=tf.argmax(pred[0]).numpy()
        results+=output_tokenizer.index_word[pred_id]+' '
        if output_tokenizer.index_word[pred_id]=='<end>':
            return results,input_sentence #attention_matrix
        decoding_input=tf.expand_dims([pred_id],0)
    return results, input_sentence #attention_matrix


def translate(input_sentence):
    results, input_sentence=evaluate(input_sentence)
    print("input: %s"%(input_sentence))
    print("pre:%s"%(results))

# translate(u'hace mucho frío aquí')
# translate(u'esta es mi vida')
# translate(u'sigues en casa')

