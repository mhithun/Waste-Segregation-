import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import base64
import urllib.request
import array
# Importing the libraries
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
import os

import serial
import time


def read():
    data = str(arduino.readline(), 'UTF-8')
    return data

try:
    arduino = serial.Serial(port='COM8', baudrate=9600, timeout=.1)
    arduino_available = True
except:
    arduino_available = False
    print("Arduino not connected - predictions will work without hardware control")


import numpy as np
import matplotlib.pyplot as plt
import cv2

model1 = load_model("training.h5")
print("success")


my_w = tk.Tk()
my_w.geometry("400x400")  # Size of the window 
my_w.title('Waste Segregation system')
my_font1=('times', 18, 'bold')


l1 = tk.Label(my_w,text='Give Waste Images',width=30,font=my_font1)  
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1, padx=5, pady=5) 

b3 = tk.Button(my_w, text='Predict Output', 
   width=20,command = lambda:predict())
b3.grid(row=6,column=1, padx=5, pady=5) 


def upload_file():
    global img
    global filename
    f_types = [('ALL', '*')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    image=Image.open(filename)

    # Resize the image in the given (width, height)
    imgs=image.resize((234, 234))
    img = ImageTk.PhotoImage(imgs)
    b2 =tk.Button(my_w,image=img) # using Button 
    b2.grid(row=9,column=1, padx=5, pady=5)
    print(filename)
def predict():
    
        
    ft=0
    st=0
    lt=0
    rt=0
    ut=0

    h=""
    out=""
    outv=5
    img = image.load_img(filename,target_size=(224,224))
    img = image.img_to_array(img, dtype='uint8')
    
    
    img = np.expand_dims(img,axis=0)   ### flattening
    ypred1 = model1.predict(img)
    ypred1 = np.argmax(ypred1, axis=1)
    print(ypred1)
    if(ypred1[0]==0):
        out = "Result for the given Image: E-Waste Detected"
        outv=0
        if arduino_available:
            arduino.write(bytes(str(ypred1[0]), 'utf-8'))
    elif(ypred1[0]==1):
        out = "Result for the given Image: Bio Waste / Organic)"
        outv=1
        if arduino_available:
            arduino.write(bytes(str(ypred1[0]), 'utf-8'))
    elif(ypred1[0]==2):
        out = "Result for the given Image: Plastic Waste)"
        outv=2
        if arduino_available:
            arduino.write(bytes(str(ypred1[0]), 'utf-8'))
    
    ft=0
    st=0
    lt=0
    rt=0
    ut=0
    
    print(out)
         
    from tkinter import messagebox  
             
    my_w.geometry("100x100")      
      
    messagebox.showinfo("Result",out)  
      
    print(" ")

my_w.mainloop()  # Keep the window open
