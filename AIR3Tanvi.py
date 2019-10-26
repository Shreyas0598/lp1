#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nltk.chat.util import Chat,reflections


# In[3]:


my_dummy_reflections = {
    "Hey" : "Hi",
    "abcd": "xyz"
}


# In[4]:


pairs = [
    [
        r"what is your name?",
        ["My name is Chatty!!!"]
    ],
    [
        r"how is the weather in (.*)",
        ["The weather in %1 is quite pleasant"]
    ]
]


# In[ ]:


def chatty():
    print("Hi!!! Please enter queries in lower case")
    chat = Chat(pairs, my_dummy_reflections)
    chat.converse()
if __name__ == "__main__":
    chatty()

