#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load dataset

# In[10]:


x = pd.ExcelFile('leaf.xlsx')
leaf = x.parse(skiprows=1)
leaf.head()


# In[12]:


leaf.columns = ['Class', 'Specimen', 'Eccentricity', ' Aspect Ratio', 'Elongation',
               'Solidity', 'Stochastic Convexity', 'Isoperimetric Factor', 'Indentation Depth',
               'Lobedness', 'Intensity', 'Contrast', 'Smoothness', 'Third moment', 'Uniformity', 'Entropy']


# In[13]:


leaf.head()


# In[14]:


leaf = leaf[['Eccentricity', ' Aspect Ratio', 'Elongation', 'Solidity', 'Stochastic Convexity', 
             'Isoperimetric Factor', 'Indentation Depth', 'Lobedness', 'Intensity', 'Contrast',
             'Smoothness', 'Third moment', 'Uniformity', 'Entropy']]
leaf.head()


# ## Convert the data into a numpy array

# In[15]:


x = leaf.values
x = scale(x);x


# ## Create a covariance matrix

# In[19]:


covar_matrix = PCA(n_components = 14) # we have 14 features


# ## Calculate Eigenvalues

# In[20]:


covar_matrix.fit(x)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var = np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features


# ### In the above array we see that the first feature explains roughly 41% of the variance within our dataset, the 1st two explain 71 %,  the 1st three explain 86 % and so on. 

# In[24]:


sb.set(font_scale=1.2,style="whitegrid") # set styling preferences
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(40,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)


# ### Based on the plot above it's clear we should pick 7 features.
