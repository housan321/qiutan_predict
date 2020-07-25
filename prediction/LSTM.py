#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
#displayd data
from IPython.display import display

graph = tf.Graph()

leagueId = {'英超': '36', '西甲': '31', '意甲': '34', '德甲': '8', '法甲': '11',
            '英冠': '37', '苏超': '29', '葡超': '23', '挪超': '22', '瑞典超': '26','俄超':'10',
            '中超': '60', '日职联': '25', '日职乙': '284', '韩K联': '15',
            '巴西甲': '4',
            '混合': 'max'}

league = '英超'
# In[2]:
modelname = r'./model/xgboost_joblib({}).dat'.format(league)
# Read data and drop redundant column.
data = pd.read_csv(r'./datasets/final_dataset/final_dataset({}).csv'.format(leagueId[league]), encoding="gbk")

data.dropna(inplace=True)
print(list(data.columns))

# select_features = ['FTR', 'FTRR','az_value9', 'Diff_AZ_Value', 'oz_odds_value9', 'Diff_OZ_Value', 'coff_home', 'coff_away']
select_features = ['FTR',
                    'HTGD', 'ATGD', 'HTP','ATP', 'HHTGD','HHTP','AATGD', 'AATP',
                    'oz_home9_std', 'oz_draw9_std', 'oz_away9_std',
                    # 'az_value0', 'Diff_AZ_Value','oz_odds_value0', 'Diff_OZ_Value',
                    'h_win_rate', 'a_win_rate','VTFormPts', 'DiffPts', 'DiffFormPts', 'Diff_HA_Pts',
                    '3general_coeff_h', '3general_coeff_a','7general_coeff_h', '7general_coeff_a',
                   ]
data = data[select_features]

# Preview data.
display(data.head())

# what is the win rate for the home team?
# Total number of matches.
n_matches = data.shape[0]
# Calculate number of features. -1 because we are saving one as the target variable (win/lose/draw)
n_features = data.shape[1] - 1
# Calculate matches won by home team.
n_homewins = len(data[data.FTR == 'H'])
# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("Total number of matches: {}".format(n_matches))
print("Number of features: {}".format(n_features))
print("Number of matches won by home team: {}".format(n_homewins))
print("Win rate of home team: {:.2f}%".format(win_rate))

data.loc[data["FTR"] == "H", "FTR"] = 1
data.loc[data["FTR"] == "NH", "FTR"] = 0

data['final1'] = 1 - data['FTR']
data['final2'] = data['FTR']

dataT = data[2300:]
data = data[:2300]







hm_epochs = 12
n_classes = 2
batch_size = 1
chunk_size = 21
n_chunks = 1
rnn_size = 64

with graph.as_default():
    x = tf.placeholder('float', [None, n_chunks,chunk_size])
    y = tf.placeholder('float')


# In[5]:


def recurrent_neural_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    x=tf.transpose(x,[1,0,2])
    print("transpose",x)
    x=tf.reshape(x,[-1,chunk_size])
    print("reshape",x)
    x=tf.split(x,n_chunks)
    print("split",x)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    


    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


# In[6]:


def train_neural_network(x):
    prediction = recurrent_neural_model(x)
    cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    y_pred = tf.nn.softmax(logits=prediction)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # hm_epochs=15
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(0,data.shape[0],batch_size):
                epoch_x, epoch_y = data.iloc[i:i+batch_size,1:chunk_size+1].values,data.iloc[i:i+batch_size,chunk_size+1:].values
                epoch_x=epoch_x.reshape((batch_size,n_chunks,chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy Train:',accuracy.eval({x:data.iloc[:,1:chunk_size+1].values.reshape((-1,n_chunks,chunk_size)),
                                               y:data.iloc[:,chunk_size+1:].values}))
        print('Accuracy Test:',accuracy.eval({x:dataT.iloc[:,1:chunk_size+1].values.reshape((-1,n_chunks,chunk_size)),
                                              y:dataT.iloc[:,chunk_size+1:].values}))
        saver.save(sess, "./model/LSTM.ckpt")


# In[7]:

with graph.as_default():
    train_neural_network(x)


# In[ ]:





