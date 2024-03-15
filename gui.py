# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageTk
import numpy
import numpy as np
import os
import librosa
# Loading the Model
from playsound import playsound
from keras.models import load_model
from IPython.display import Audio
import pygame
model=load_model('audio_classification.keras')

pygame.mixer.init()

# Initializing the GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Voice Gender detector')
top.configure(background='#CDCDCD')

# Initializing the Labels (1 for age and 1 for Sex)
label1=Label(top,background="#CDCDCD",font=('arial',15,"bold"))
label2=Label(top,background="#CDCDCD",font=('arial',15,'bold'))
sign_image=Label(top)

# Definig Detect fuction which detects the age and gender of the person in image using the model
def Detect(file_path):
    librosa_audio_data,librosa_sample_rate=librosa.load(file_path,res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=30)
    mfccs = np.mean(mfccs.T,axis=0)
    mfccs = mfccs.reshape(1,-1)
    res = model.predict(mfccs)
    predicted_class = int(res > 0.48)
    print("Predicted class:", predicted_class)
    print(predicted_class)
    sex_f=["Female","Male"]
    print("Predicted Gender is "+sex_f[predicted_class]) 
    label2.configure(foreground="#011638",text=sex_f[predicted_class]) 

# Defining Show_detect button function
def show_Detect_button(file_path):
    Detect_b=Button(top,text="Detect Voice",command=lambda: Detect(file_path),padx=10,pady=5)
    Detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    Detect_b.place(relx=0.79,rely=0.46) 
def show_play(file_path):
    play_button = Button(top,text="Play",command=lambda:play(file_path),padx=10,pady=10)
    play_button.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    play_button.place(relx=0.4,rely=0.46)
def play(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play(loops=0)
def show_stop(file_path):
    play_button = Button(top,text="Stop",command=lambda:stop(file_path),padx=10,pady=10)
    play_button.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    play_button.place(relx=0.5,rely=0.46)
def stop(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.stop()



# Definig Upload Image Function
def upload_audio():
    try:
        file_path=filedialog.askopenfilename()

        librosa_audio_data,librosa_sample_rate=librosa.load(file_path)

        Audio(data=librosa_audio_data,rate=librosa_sample_rate)
        show_Detect_button(file_path)
        show_play(file_path)
        show_stop(file_path)
    except:
        pass

upload=Button(top,text="Upload Audio",command=upload_audio,padx=10,pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)

sign_image.pack(side='bottom',expand=True)
label1.pack(side="bottom",expand=True)
label2.pack(side="bottom",expand=True)
heading=Label(top,text="Voice Gender Detector",pady=20,font=('arial',20,"bold"))
heading.configure(background="#CDCDCD",foreground="#364156")
heading.pack()
top.mainloop()
