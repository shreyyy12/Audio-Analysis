try:
    import pyaudio
    import numpy as np
    import pylab
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.io.wavfile import write
    import time
    import sys
    import wavio as wv
    import wave
    import matplotlib.animation as animation
    import cv2
    import os
    from dtw import dtw
    from math import abs

except:
    print ("Something didn't import")

i=0
f,ax = plt.subplots(2)

# Prepare the Plotting Environment with random starting values
x = np.arange(10000)
y = np.random.randn(10000)

# Plot 0 is for raw audio data
li, = ax[0].plot(x, y)
ax[0].set_xlim(0,1000)
ax[0].set_ylim(-5000,5000)
ax[0].set_title("Raw Audio Signal")

# Plot 1 is for the FFT of the audio
li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,5000)
ax[1].set_ylim(-100,100)
ax[1].set_title("Fast Fourier Transform")
# Show the plot, but without blocking updates
plt.pause(0.01)
plt.tight_layout()

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 1
RATE = 44100
CHUNK = 1024 # 1024bytes of data red from a buffer
RECORD_SECONDS = 0.1
FRAMES_PER_BUFFER=3200
#WAVE_OUTPUT_FILENAME = "sound.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True)#,
                    #frames_per_buffer=CHUNK)

global keep_going
keep_going = True

#obj = wave.open('rec1.wav','wb')

def plot_data(in_data):
    # get and convert the data to float
    global audio_data
    audio_data = np.fromstring(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))
    #freqs=np.dfft.fftfreq(len(signal),t[1]-t[0])
    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    li.set_xdata(np.arange(len(audio_data)))
    li.set_ydata(audio_data)
    li2.set_xdata(np.arange(len(dfft))*10.)
    li2.set_ydata(dfft)
    
    
    # Show the updated plot, but without blocking
    plt.pause(0.01)
    if keep_going:
        
        return True
    else:
        return False
print("start recording")

seconds = 10
frames = []

for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)

# Open the connection and start streaming the data
stream.start_stream()
print ("\n+---------------------------------+")
print ("| Press Ctrl+C to Break Recording |")
print ("+---------------------------------+\n")

# Loop so program doesn't end while the stream callback's
# itself for new data
while keep_going:
    try:
        plot_data(stream.read(CHUNK))
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
    except KeyboardInterrupt:
        keep_going=False
    except:
        pass

# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
stream.stop_stream()
stream.close()

audio.terminate()



obj = wave.open("output.wav", "wb")
obj.setnchannels(CHANNELS)
obj.setsampwidth(audio.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b"".join(frames))
obj.close()


import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

samplerate, data = wavfile.read("output.wav")
print(f"Sample rate: {samplerate}")

length = data.shape[0] / samplerate
print(f"length = {length}s")

fft_data = np.fft.fft(data)

def extract_peak_frequency(data, sampling_rate):
    global peak_freq
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    
    peak_coefficient = np.argmax(np.abs(fft_data))
    peak_freq = freqs[peak_coefficient]
    
    return abs(peak_freq * sampling_rate)

# Installing and importing necessary libraries

from python_speech_features import mfcc, logfbank
sampling_freq, sig_audio = wavfile.read("output.wav")
# We will now be taking the first 15000 samples from the signal for analysis
sig_audio = sig_audio[:15000]
# Using MFCC to extract features from the signal
mfcc_feat1 = mfcc(sig_audio, sampling_freq)
print('\nMFCC Parameters\nWindow Count =', mfcc_feat1.shape[0])
print('Individual Feature Length =', mfcc_feat1.shape[1])

import mysql.connector

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open("output.wav", 'rb') as file:
        binaryData = file.read()
    return binaryData


def insertBLOB(audio,freq,sample,window1,ind_len):
    print("Inserting BLOB into audio2")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='hack1',
                                             user='root',
                                             password='Root@1234')

        cursor = connection.cursor()
        sql_insert_blob_query = """ INSERT INTO audio2
                          (audio,freq,sample,window1,ind_len) VALUES (%s,%s,%s,%s,%s)"""

        
        file = convertToBinaryData(audio)

        # Convert data into tuple format
        insert_blob_tuple = (file,freq,sample,window1,ind_len)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection.commit()
        print("Image and file inserted successfully as a BLOB into python_employee table", result)

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

insertBLOB(r"C:\Users\srika\AppData\Local\Progra1ms\Python\Python311\output.wav",extract_peak_frequency(data, samplerate),length,mfcc_feat1.shape[0],mfcc_feat1.shape[1])





print("------------------------------------------------------------------")
print("MAPPING DATA- ENTER ANY AUDIO INPUT TO BE MAPPED WITH THE DATABASE")
'''i=0
f,ax = plt.subplots(2)

# Prepare the Plotting Environment with random starting values
x = np.arange(10000)
y = np.random.randn(10000)

# Plot 0 is for raw audio data
li, = ax[0].plot(x, y)
ax[0].set_xlim(0,1000)
ax[0].set_ylim(-5000,5000)
ax[0].set_title("Raw Audio Signal")
# Plot 1 is for the FFT of the audio
li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,5000)
ax[1].set_ylim(-100,100)
ax[1].set_title("Fast Fourier Transform")
# Show the plot, but without blocking updates
plt.pause(0.01)
plt.tight_layout()

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 1
RATE = 44100
CHUNK = 1024 # 1024bytes of data red from a buffer
RECORD_SECONDS = 0.1
FRAMES_PER_BUFFER=3200
#WAVE_OUTPUT_FILENAME = "sound.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True)#,
                    #frames_per_buffer=CHUNK)

#global keep_going
keep_going = True

#obj = wave.open('rec1.wav','wb')

def plot_data(in_data):
    # get and convert the data to float
    global audio_data
    audio_data = np.fromstring(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))
    #freqs=np.dfft.fftfreq(len(signal),t[1]-t[0])
    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    li.set_xdata(np.arange(len(audio_data)))
    li.set_ydata(audio_data)
    li2.set_xdata(np.arange(len(dfft))*10.)
    li2.set_ydata(dfft)
    
    
    # Show the updated plot, but without blocking
    plt.pause(0.01)
    if keep_going:
        
        return True
    else:
        return False
print("start recording")

seconds = 10
frames = []
for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)

stream.stop_stream()
stream.close()
np.terminate()



# Open the connection and start streaming the data
stream.start_stream()
print ("\n+---------------------------------+")
print ("| Press Ctrl+C to Break Recording |")
print ("+---------------------------------+\n")

# Loop so program doesn't end while the stream callback's
# itself for new data
while keep_going:
    try:
        plot_data(stream.read(CHUNK))
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
    except KeyboardInterrupt:
        keep_going=False
    except:
        pass

# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
stream.stop_stream()
stream.close()

audio.terminate()
obj = wave.open("output1.wav", "wb")
obj.setnchannels(CHANNELS)
obj.setsampwidth(audio.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b"".join(frames))
obj.close()

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

samplerate, data = wavfile.read("output1.wav")
print(f"Sample rate: {samplerate}")

length = data.shape[0] / samplerate
print(f"length = {length}s")

fft_data = np.fft.fft(data)

def extract_peak_frequency(data, sampling_rate):
    global peak_freq
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    
    peak_coefficient = np.argmax(np.abs(fft_data))
    peak_freq = freqs[peak_coefficient]
    
    return abs(peak_freq * sampling_rate)
'''
# Installing and importing necessary libraries

from python_speech_features import mfcc, logfbank
sampling_freq, sig_audio = wavfile.read("output1.wav")
# We will now be taking the first 15000 samples from the signal for analysis
sig_audio = sig_audio[:15000]
# Using MFCC to extract features from the signal
mfcc_feat = mfcc(sig_audio, sampling_freq)
print('\nMFCC Parameters\nWindow Count =', mfcc_feat.shape[0])
print('Individual Feature Length =', mfcc_feat.shape[1])

import mysql.connector


def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)


def readBLOB(audio,freq,sample,window1,ind_len):
    print("Reading BLOB data from table")

    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='hack1',
                                             user='root',
                                             password='Root@1234')

        cursor = connection.cursor()
        sql_fetch_blob_query = """SELECT * from audio2"""

        cursor.execute(sql_fetch_blob_query)
        record = cursor.fetchall()
        for row in record:
            print("freq", row[1], )
            print("sample", row[2])
            audio = row[0]
            dist=row[4]-mfcc_feat1.shape[1] 

            if abs(dist)<=5 and abs(row[1]-freq)<=150 and abs(row[2]-sample)<1.5:
                print("same audio")
            else:
                print("not same audio")

    except mysql.connector.Error as error:
        print("Failed to read BLOB data from MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


readBLOB("output.wav",extract_peak_frequency(data, samplerate),length,mfcc_feat.shape[0],mfcc_feat.shape[1])

