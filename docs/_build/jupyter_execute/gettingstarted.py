#!/usr/bin/env python
# coding: utf-8

# In[1]:


from crispy import CRISP


# In[2]:


c_ex = CRISP("../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits")


# In[3]:


print(c_ex)


# In[4]:


c_sub = c_ex[3] # remember Python indexing starts at 0!


# In[5]:


c_sub.intensity_map()

