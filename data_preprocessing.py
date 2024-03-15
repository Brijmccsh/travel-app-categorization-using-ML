#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd

data_folder = 'p1'
emotions = ['A_folder', 'H_folder', 'C_folder', 'N_folder', 'S_folder']


data_dict = {}
for emotion in emotions:
    emotion_path = os.path.join(data_folder, emotion)
    users = os.listdir(emotion_path)
    user_data = {}
    for user in users:
        user_path = os.path.join(emotion_path, user)
        if os.path.isdir(user_path):  
            user_data[user] = [pd.read_csv(os.path.join(user_path, file)) for file in os.listdir(user_path)]
    data_dict[emotion] = user_data



# In[3]:


import numpy as np

# One-hot encode "keyCode" column
for emotion, users in data_dict.items():
    for user, user_data in users.items():
        for df in user_data:
            df = pd.get_dummies(df, columns=['keyCode'], prefix='keyCode', drop_first=True)


# In[4]:


from sklearn.preprocessing import StandardScaler

# Normalize numerical features
scaler = StandardScaler()

for emotion, users in data_dict.items():
    for user, user_data in users.items():
        for df in user_data:
            df[['D1U1', 'D1U2', 'U1D2', 'U1U2']] = scaler.fit_transform(df[['D1U1', 'D1U2', 'U1D2', 'U1U2']])


# In[14]:


sequence_length = 10  
sequences_list = []
labels = []

for emotion, users in data_dict.items():
    for user, user_data in users.items():
        sequences = []
        for df in user_data:
            # Assuming 'keyCode' columns are one-hot encoded, select relevant columns for sequences
            sequence_data = df[['keyCode_a', 'keyCode_b', ...]]  # Update column names based on your one-hot encoding

            # Check if there are enough data points for the sequence
            if len(sequence_data) >= sequence_length:
                seq = create_sequences(sequence_data.values, sequence_length)
                sequences.append(seq)

        # Check if there are valid sequences for the user
        if sequences:
            sequences_list.append(np.concatenate(sequences))
            labels.extend([label_mapping[emotion]] * len(sequences))

# Concatenate all sequences and labels
if sequences_list:
    all_sequences = np.concatenate(sequences_list)
    all_labels = np.array(labels)
else:
    raise ValueError("No valid sequences found.")


# In[10]:





# In[ ]:




