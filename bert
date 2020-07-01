# -*- coding: utf-8 -*-


!pip install transformers

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

"""  Importing the dataset
"""

df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)

"""we'll only use 2,000 sentences from the dataset"""

batch_1 = df[:2000]

"""## Loading the Pre-trained BERT model
Let's now load a pre-trained BERT model.
"""

# For DistilBERT:
#model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## for BERT instead of distilBERT
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

"""Right now, the variable `model` holds a pretrained distilBERT model -- a version of BERT that is smaller, but much faster and requiring a lot less memory.

### Tokenization
Our first step is to tokenize the sentences -- break them up into word and subwords in the format BERT is comfortable with.
"""

tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

"""

### Padding
After tokenization, `tokenized` is a list of sentences -- each sentences is represented as a list of tokens. We want BERT to process our examples all at once (as one batch). It's just faster that way. For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array, rather than a list of lists (of different lengths).
"""

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

"""Our dataset is now in the `padded` variable, we can view its dimensions below:"""

np.array(padded).shape

"""### Masking
If we directly send `padded` to BERT, that would slightly confuse it. We need to create another variable to tell it to ignore (mask) the padding we've added when it's processing its input. That's what attention_mask is:
"""

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

"""## Model 
Now that we have our model and inputs ready, let's run our model!

The `model()` function runs our sentences through BERT. The results of the processing will be returned into `last_hidden_states`.
"""

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

"""Let's slice only the part of the output that we need. That is the output corresponding the first token of each sentence. The way BERT does sentence classification, is that it adds a token called `[CLS]` (for classification) at the beginning of every sentence. The output corresponding to that token can be thought of as an embedding for the entire sentence.



We'll save those in the `features` variable, as they'll serve as the features to our logitics regression model.
"""

features = last_hidden_states[0][:,0,:].numpy()

"""The labels indicating which sentence is positive and negative now go into the `labels` variable"""

labels = batch_1[1]

"""## Model #2: Train/Test Split
Let's now split our datset into a training set and testing set (even though we're using 2,000 sentences from the SST2 training set).
"""

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

""" We now train the LogisticRegression model. If you've chosen to do the gridsearch, you can plug the value of C into the model declaration (e.g. `LogisticRegression(C=5.2)`)."""

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

"""

## Evaluating Model #2
So how well does our model do in classifying sentences? One way is to check the accuracy against the testing dataset:
"""

lr_clf.score(test_features, test_labels)



"""So our model clearly does better than a dummy classifier. But how does it compare against the best models?

"""

