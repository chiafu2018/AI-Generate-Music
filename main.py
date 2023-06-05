#Importing Libraries
import tensorflow as tf
import numpy as np 
import pandas as pd 
from collections import Counter
from music21 import *
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, GlobalMaxPooling1D
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import matplotlib.pyplot as plt
import warnings
import loading_data
import subprocess
import os

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(50)

#Choose your Dataset
option = int(input("Do you want to load dataset online? Please enter 1 if yes, otherwise 0 :"))

#Getting the list of notes as Corpus
all_midis= loading_data.capture_data(option)
Corpus= loading_data.extract_notes(all_midis)
print("Total notes in all the Chopin midis in the dataset:", len(Corpus))
print("First one hundred values in the Corpus:", Corpus[:100])


'''''''''''''''''''''''''''''''''''''''
Data Analyzing
'''''''''''''''''''''''''''''''''''''''

#Creating a count dictionary
count_num = Counter(Corpus)
print("Total unique notes in the Corpus:", len(count_num))

#Exploring the notes dictionary
Notes = list(count_num.keys())
Recurrence = list(count_num.values())
#Average recurrenc for a note in Corpus
def Average(lst):
    return sum(lst) / len(lst)
print("Average recurrence for a note in Corpus:", Average(Recurrence))
print("Most frequent note in Corpus appeared:", max(Recurrence), "times")
print("Least frequent note in Corpus appeared:", min(Recurrence), "time")

# Plotting the distribution of Notes

'''
plt.figure(figsize=(18,3),facecolor="#97BACB")
bins = np.arange(0,(max(Recurrence)), 50) 
plt.hist(Recurrence, bins=bins, color="#97BACB")
plt.axvline(x=100,color="#DBACC1")
plt.title("Frequency Distribution Of Notes In The Corpus")
plt.xlabel("Frequency Of Chords in Corpus")
plt.ylabel("Number Of Chords")
plt.show()
'''


#Getting a list of rare chords & Eleminating them 
rare_note = []
for index, (key, value) in enumerate(count_num.items()):
    if value < 10:
        m =  key
        rare_note.append(m)    
print("Total number of notes that occur less than 10 times:", len(rare_note))

for element in Corpus:
    if element in rare_note:
        Corpus.remove(element)
print("Length of Corpus after elemination the rare notes:", len(Corpus))


'''''''''''''''''''''''''''''
Data Preprocessing
'''''''''''''''''''''''''''''
#Storing all the unique notes, which I call them symbols here, present in my corpus to bult a mapping dic 
symbols = sorted(list(set(Corpus))) 
corpus_length = len(Corpus) #length of corpus
symbol_length = len(symbols) #length of total unique notes

#Building dictionary to map an unique note to a number (ex:'E2': 115), and its reverse
mapping = dict((c, i) for i, c in enumerate(symbols))
reverse_mapping = dict((i, c) for i, c in enumerate(symbols))

print("Total number of notes:", corpus_length)
print("Number of unique notes:", symbol_length)

#Please be attention here, it's very important!!
#Splitting the Corpus in equal length of features and output targets
fearture_length = 10 #I set feature length as 10 to have both Composer's feature and the creativity of music (I don't know right or not) 
features = []
targets = []
for i in range(0, corpus_length - fearture_length, 1):
    feature = Corpus[i:i + fearture_length]  #Set sequence of notes as features 
    target = Corpus[i + fearture_length]     #The note next to feature is target (only one note)
    features.append([mapping[j] for j in feature]) #Change a sequence of notes feature into number 
    targets.append(mapping[target])
    
#One target is right behind of one sequence map (number of features == number of targets)    
features_number = len(targets)
print("Total number of sequence features:", features_number)


#Reshape X to 3 dimensional vectors and normalize them by dividing the number of unique notes
X = (np.reshape(features, (features_number, fearture_length, 1)))/ float(symbol_length)

#One hot encode the output variable (feature length of notes for one target)
y = tf.keras.utils.to_categorical(targets) 


#Taking out a subset of data to be used as seed, 20% data for testing, 80% for training
X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)


'''''''''''''''''''''''''''''''''''''''''''''''
Model Building: 1.LSTM  2.Conv1D 3.LSTM+Conv1D
'''''''''''''''''''''''''''''''''''''''''''''''

#####################################################################
# 1.Model_LSTM
model_LSTM = Sequential()
#Adding layers
model_LSTM.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model_LSTM.add(Dropout(0.1))
model_LSTM.add(LSTM(256))
model_LSTM.add(Dense(256))
model_LSTM.add(Dropout(0.1))
model_LSTM.add(Dense(y.shape[1], activation='softmax'))
#Compiling the model for training  
model_LSTM.compile(loss='categorical_crossentropy', optimizer=Adamax(learning_rate=0.01))
#Model's Summary   
print("Model_LSTM:")            
model_LSTM.summary()
#####################################################################

# 2.Model_Conv1D
model_Conv1D = Sequential()
#Adding layers

model_Conv1D.add(Conv1D(filters=256, kernel_size=3, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
model_Conv1D.add(Dropout(0.1))
#model_Conv1D.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model_Conv1D.add(Dense(64))
model_Conv1D.add(Dropout(0.1))
model_Conv1D.add(Dense(y.shape[1], activation='softmax'))
model_Conv1D.add(GlobalMaxPooling1D())
#Compiling the model for training  
model_Conv1D.compile(loss='categorical_crossentropy', optimizer=Adamax(learning_rate=0.01))
#Model's Summary   
print("Model_Conv1D:")            
model_Conv1D.summary()
#####################################################################

# 3.Model_Merge
model_Merge = Sequential()
#Adding layers
model_Merge.add(Conv1D(filters=256, kernel_size=3, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
model_Merge.add(Dropout(0.1))
model_Merge.add(LSTM(256))
model_Merge.add(Dense(256))
model_Merge.add(Dropout(0.1))
model_Merge.add(Dense(y.shape[1], activation='softmax'))
#Compiling the model for training  
model_Merge.compile(loss='categorical_crossentropy', optimizer=Adamax(learning_rate=0.01))
#Model's Summary   
print("Model_Merge:")            
model_Merge.summary()
#####################################################################

'''''''''''''''''''''''''''''''''''''''''''''''
Trainging Different Models
'''''''''''''''''''''''''''''''''''''''''''''''
#Training the Model with GPU
gpu = tf.config.list_physical_devices('GPU')
print("How many gpu is avaliable: ", len(gpu))

if gpu:
    # Restrict TensorFlow to only use the first GPU
    tf.config.set_visible_devices(gpu[0], 'GPU')
    #Start training data 
    model_LSTM.fit(X_train, y_train, batch_size=256, epochs=200)
    model_Conv1D.fit(X_train, y_train, batch_size=256, epochs=200) 
    model_Merge.fit(X_train, y_train, batch_size=256, epochs=200) 


'''''''''''''''''''''''''''''''''''''''''''''''
Generating Music using Different Models
'''''''''''''''''''''''''''''''''''''''''''''''
#####################################################################

def Malody_Generator_LSTM(Note_Count):
    seed = X_seed[np.random.randint(0,len(X_seed)-1)]
    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = seed.reshape(1,fearture_length,1)
        prediction = model_LSTM.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0 #diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(symbol_length)   
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    Melody = loading_data.chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)   
    return Music,Melody_midi

Music_notes_LSTM, Melody_LSTM = Malody_Generator_LSTM(300)
Melody_LSTM.write('midi','LSTM.mid')
#####################################################################

def Malody_Generator_Conv1D(Note_Count):
    seed = X_seed[np.random.randint(0,len(X_seed)-1)]
    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = seed.reshape(1,fearture_length,1)
        prediction = model_Conv1D.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0 #diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(symbol_length)   
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    Melody = loading_data.chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)   
    return Music,Melody_midi

Music_notes_Conv1D, Melody_Conv1D= Malody_Generator_Conv1D(300)
Melody_Conv1D.write('midi','Conv1D.mid')
#####################################################################

def Malody_Generator_Merge(Note_Count):
    seed = X_seed[np.random.randint(0,len(X_seed)-1)]
    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = seed.reshape(1,fearture_length,1)
        prediction = model_Merge.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0 #diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(symbol_length)   
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    Melody = loading_data.chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)   
    return Music,Melody_midi

Music_notes_Merge, Melody_Merge = Malody_Generator_Merge(300)
Melody_Merge.write('midi','Merge.mid')
#####################################################################

'''''''''''''''''''''''''''''''''''''''''''''''
Convert mid to MP3 & Delete them
'''''''''''''''''''''''''''''''''''''''''''''''
#os.environ["PATH"] += os.pathsep + r"C:\Program Files\fluidsynth-2.2.4-win10-x64\bin" #Add fluidsynth to path

subprocess.run(["fluidsynth", "-ni", r"C:\soundfonts\default.sf2", r"C:\Users\chiaf\source_vs\AI_project_new\LSTM.mid", 
                "-F", "Output Music\\LSTM.mp3", "-r", "44100"])
os.remove('LSTM.mid')

subprocess.run(["fluidsynth", "-ni", r"C:\soundfonts\default.sf2", r"C:\Users\chiaf\source_vs\AI_project_new\Conv1D.mid", 
                "-F", "Output Music\\Conv1D.mp3", "-r", "44100"])
os.remove('Conv1D.mid')

subprocess.run(["fluidsynth", "-ni", r"C:\soundfonts\default.sf2", r"C:\Users\chiaf\source_vs\AI_project_new\Merge.mid", 
                "-F", "Output Music\\Merge.mp3", "-r", "44100"])
os.remove('Merge.mid')