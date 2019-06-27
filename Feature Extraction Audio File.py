#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


import os
os.chdir('C:\\Analytics\\Deep Learning\\audio files')


# In[7]:


import librosa
audio_path = "bird.wav"
x,sr = librosa.load(audio_path)


# In[8]:


import IPython.display as ipd
ipd.Audio(audio_path)


# # Visualizing Audio

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(20,5))
librosa.display.waveplot(x,sr=sr)


# Spectrogram

# A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. We can also called it as voice prints

# In[13]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(20,5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time',y_axis='hz')
plt.colorbar()


# Log frequency axis

# In[14]:


librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='log')
plt.colorbar()


# Creating an audio signal

# Let us now create an audio signal at 220Hz. We know an audio signal is a numpy array, so we shall create one and pass it on to the audio function.

# In[15]:


import numpy as np
sr = 22050 # sample rate
T = 5.0  # seconds
t = np.linspace(0,T, int(T*sr), endpoint=False) # which is a time variable
x = 0.5*np.sin(2*np.pi*220*t)


# Playing the sound

# In[16]:


ipd.Audio(x, rate=sr)


# Lets save the signal

# In[17]:


librosa.output.write_wav('generate.wav',x,sr)


# In[18]:


x,sr = librosa.load('horse_gallop.wav')
ipd.Audio(x,rate=sr)


# In[19]:


# plot the signal
plt.figure(figsize=(20,5))
librosa.display.waveplot(x,sr=sr)


# # 1. Zero crossing Rate

# The zero-crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to zero to negative or from negative to zero to positive. This feature has been used heavily in both speech recognition and music information retrieval, being a key feature to classify percussive sounds.

# In[20]:


# Zoom In
n0 = 9000
n1 = 9100
plt.figure(figsize=(20,5))
plt.plot(x[n0:n1])
plt.grid()


# In[21]:


zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
zero_crossings.shape


# In[22]:


print(sum(zero_crossings))


# # 2. Spectral Centroid

# Some people use "spectral centroid" to refer to the median of the spectrum. This is a different statistic, the difference being essentially the same as the difference between the unweighted median and mean statistics. Used in digital signal processing.

# In[25]:


spectral_centroids = librosa.feature.spectral_centroid(x,sr=sr)[0]
spectral_centroids.shape


# In[26]:


# Computing the time variable for visualization
plt.figure(figsize=(20,5))
frame = range(len(spectral_centroids))
t = librosa.frames_to_time(frame)

# Normalizing the spectral centroid for visualization
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x,axis=axis)

# Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x,sr=sr,alpha=0.4)
plt.plot(t,normalize(spectral_centroids),color='r')


# # 3. Spectral Rolloff

# It measures the right-skewedness of the power spectrum

# In[27]:


plt.figure(figsize=(20,5))
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01,sr=sr)[0]
librosa.display.waveplot(x,sr=sr,alpha=0.4)
plt.plot(t,normalize(spectral_rolloff),color='r')
plt.grid()


# # 4. MFCC

# recognition by using the Mel-Scale Frequency Cepstral Coefficients (MFCC) extracted from speech signal of spoken words.

# In[28]:


plt.figure(figsize=(20,5))
x,fs = librosa.load('bird.wav')
librosa.display.waveplot(x,sr=sr)


# In[29]:


plt.figure(figsize=(20,5))
mfccs = librosa.feature.mfcc(x,sr=sr)
print(mfccs.shape)

librosa.display.specshow(mfccs,sr=sr,x_axis='time')


# # Feature Scaling

# Scale the MFCC such that each coefficient dimension has zero mean and unit variance

# In[30]:


mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))


# In[31]:


plt.figure(figsize=(20,8))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


# # Chroma Frequencies

# In[32]:


x,sr = librosa.load('bird.wav')
ipd.Audio(x,rate=sr)


# In[33]:


hop_length = 512
chromagram = librosa.feature.chroma_stft(x,sr=sr,hop_length=hop_length)
plt.figure(figsize=(15,5))
librosa.display.specshow(chromagram, x_axis='time',y_axis='chroma', hop_length=hop_length,cmap='coolwarm')


# In[ ]:




