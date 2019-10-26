#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


# In[ ]:


####SENTENCE TOKENIZATION
text = "this’s a sent tokenize test. this is sent two. is this sent three? sent 4 is cool! Now it’s your turn."
sent_tokenize_list = sent_tokenize(text)
len(sent_tokenize_list)


# In[ ]:


sent_tokenize_list


# In[ ]:


sent_tokenize_list[1]


# In[ ]:


####WORD TOKENIZATION
word_tokenize_list = word_tokenize(text)
word_tokenize_list


# In[ ]:


####POS TAGGING
nltk.download('averaged_perceptron_tagger')
text1 = "Dive into NLTK: Part-of-speech tagging and POS Tagger"
word_tokenize_list1 = word_tokenize(text1)
pos_tag_list = nltk.pos_tag(word_tokenize_list1)
pos_tag_list


# In[ ]:


nltk.download('tagsets')
nltk.help.upenn_tagset('NN')


# In[ ]:


import nltk
text = "learn php from guru99"
tokens = nltk.word_tokenize(text)
print(tokens)
tag = nltk.pos_tag(tokens)
print(tag)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp  =nltk.RegexpParser(grammar)
result = cp.parse(tag)
print(result)
result.draw()  

