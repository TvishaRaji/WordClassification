#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf 


# In[3]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import plotly.graph_objs as go


# In[43]:


from keras.callbacks import EarlyStopping


# In[44]:


from IPython.core.interactiveshell import InteractiveShell


# In[45]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
#from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import plotly.graph_objs as go
#import plotly.plotly as py
#import cufflinks
from IPython.core.interactiveshell import InteractiveShell
#import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
#cufflinks.go_offline()
#cufflinks.set_config_file(world_readable=True, theme='pearl')
# Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[91]:


data = pd.read_csv('complaints_processed.csv')
data.info()


# In[92]:


data = pd.read_csv('complaints_processed.csv')
data.info()


# In[93]:


data.head()


# In[94]:


data.pop('narrative.1')


# In[95]:


data.head()


# In[96]:


data.pop('narrative.2')


# In[97]:


data.pop('narrative.3')&&data.pop('narrative.4')


# In[98]:


data.pop('narrative.3')


# In[99]:


data.pop('narrative.4')


# In[100]:


data.head()


# In[107]:


data_n = data.drop(['Unnamed: 0'], axis= 1)


# In[108]:


data_n=data_n.groupby(['product']).count()
data_n.head()


# In[119]:


import plotly.express as px

fig = px.pie(data_n,values='narrative',
                title="Count Complaint in Each Product")
fig.show()


# In[59]:


data_n.head()


# In[60]:


def print_plot(index):
    example = data_n[data_n.index == index][['narrative', 'product']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Product:', example[1])
print_plot(10)


# In[61]:


print_plot(70)


# In[62]:


data_n.info()


# In[63]:


data_n = data_n.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = str(text)
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    # replace REPLACE_BY_SPACE_RE symbols by space in text. 
    # substitute the matched string in REPLACE_BY_SPACE_RE with space.
    
    text = BAD_SYMBOLS_RE.sub('', text) 
    # remove symbols which are in BAD_SYMBOLS_RE from text. 
    # substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    #text = text.replace('x', '')
    #text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    # remove stopwors from text
    return text
data_n['narrative'] = data_n['narrative'].apply(clean_text)
data_n['narrative'] = data_n['narrative'].str.replace('\d+', '')


# In[64]:


print_plot(30)


# In[65]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, 
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', 
                      lower=True)
tokenizer.fit_on_texts(data_n['narrative'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[66]:


X = tokenizer.texts_to_sequences(data_n['narrative'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[67]:


Y = pd.get_dummies(data_n['product']).values
print('Shape of label tensor:', Y.shape)


# In[73]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[69]:


## Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                              input_length=X.shape[1]),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, 
                                                       return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    #tf.keras.layers.Dense(128, activation='softmax'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()


# In[70]:


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              optimizer='Nadam', metrics=["CategoricalAccuracy"])


# In[71]:


num_epochs = 5
batch_size = 128
## For early stopping to ensure it doesnt overfit
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
history = model.fit(X_train, Y_train, 
                    epochs=num_epochs, batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss',
                                             patience=3,
                                             min_delta=0.0001)])


# In[2]:


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[75]:


history.history


# In[76]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# In[77]:


plt.title('Accuracy')
plt.plot(history.history['categorical_accuracy'], label='train')
plt.plot(history.history['val_categorical_accuracy'], label='test')
plt.legend()
plt.show();


# In[120]:


new_complaint = ['I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX.']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['credit_card',
'credit_reporting',
'debt_collection',
'mortgages_and_loans',
'retail_banking']
print(pred, labels[np.argmax(pred)])


# In[1]:


new_complaint = [' I would like to enquire about my loan application status with reference number 210420115007 and like to know about my previous exsisting mortage and intrests on them  ']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['credit_card',
'credit_reporting',
'debt_collection',
'mortgages_and_loans',
'retail_banking']
print(pred, labels[np.argmax(pred)])


# In[ ]:




