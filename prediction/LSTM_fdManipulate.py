#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from IPython.display import display

from tensorflow.contrib import rnn
import tensorflow as tf
graph = tf.Graph()
# In[2]:

# Read data and drop redundant column.
data = pd.read_csv('./datasets/final_dataset/final_dataset(31).csv', encoding = "gbk")

data.drop(['HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'VTWinStreak3', 'VTWinStreak5', 'VTLossStreak3', 'VTLossStreak5',
           'oz_home9_mean', 'oz_draw9_mean', 'oz_away9_mean',
            'HM1','HM2','HM3','AM1','AM2','AM3','DiffLP',
           ], 1, inplace=True)

# ## Data Exploration
#what is the win rate for the home team?
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


data.FTR[data.FTR=='H'] = 1
data.FTR[data.FTR=='NH'] = 0
onehotencoder=OneHotEncoder()
onehotencoder.fit(data.FTR.reshape(-1,1))
final = onehotencoder.transform([[each] for each in data.FTR]).toarray()


X_all = data.drop(['FTR','FTRR'],1)


# we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)

        # Collect the revised columns
        output = output.join(col_data)

    return output


# X_part = X_all[['HM1','HM2','HM3','AM1','AM2','AM3','VM1','VM2','VM3','VM4','VM5',]]      # ,'VM1','VM2','VM3','VM4','VM5'
X_part = X_all[['VM1', 'VM2', 'VM3', 'VM4', 'VM5']]  # ,'VM1','VM2','VM3','VM4','VM5'
X_part = preprocess_features(X_part)

# X_all = X_all.drop(['HM1','HM2','HM3','AM1','AM2','AM3','VM1','VM2','VM3','VM4','VM5',],1)    # ,'VM1','VM2','VM3','VM4','VM5'
X_all = X_all.drop(['VM1', 'VM2', 'VM3', 'VM4', 'VM5'], 1)
X_all = pd.concat([X_all, X_part], axis=1)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
# Show the feature information by printing the first five rows
print("\nFeature values:")
display(X_all.head())


X_all.loc[:,"final1"]=final[:,0]
X_all.loc[:,"final2"]=final[:,1]


data = X_all[:2200]
dataT = X_all[2200:]

data.head()
dataT.head()


hm_epochs = 8
n_classes = 2
batch_size = 10
chunk_size = data.shape[1]-2
n_chunks = 1
rnn_size = 128

with graph.as_default():
    x = tf.placeholder('float', [None, n_chunks, chunk_size])
    y = tf.placeholder('float')


# In[5]:


def recurrent_neural_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    print("transpose", x)
    x = tf.reshape(x, [-1, chunk_size])
    print("reshape", x)
    x = tf.split(x, n_chunks)
    print("split", x)

    # lstm_cell = rnn.BasicLSTMCell(rnn_size)
    # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    lstm_cell_1 = rnn.BasicLSTMCell(rnn_size)
    lstm_cell_2 = rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2])
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)



    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


# In[6]:


def train_neural_network(x):
    prediction = recurrent_neural_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    y_pred = tf.nn.softmax(logits=prediction)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # hm_epochs=15
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(0, data.shape[0], batch_size):
                epoch_x, epoch_y = data.iloc[i:i + batch_size, 0:chunk_size].values, data.iloc[i:i + batch_size,
                                                                                         chunk_size:].values
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy Train:',
              accuracy.eval({x: data.iloc[:, 0:chunk_size].values.reshape((-1, n_chunks, chunk_size)),
                             y: data.iloc[:, chunk_size:].values}))
        print('Accuracy Test:',
              accuracy.eval({x: dataT.iloc[:, 0:chunk_size].values.reshape((-1, n_chunks, chunk_size)),
                             y: dataT.iloc[:, chunk_size:].values}))
        saver.save(sess, "./model/model.ckpt")


# In[7]:

with graph.as_default():
    train_neural_network(x)




