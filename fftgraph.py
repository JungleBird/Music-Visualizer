# Python example - Fourier transform using numpy.fft method
import matplotlib.pyplot as plotter
from matplotlib.animation import FuncAnimation
import pyaudio
import wave
import numpy as np
import time

'''
Sampling Rate: 44.1 Mhz * 512 Samples = 11.61 ms 

chunk size = 512*2 (sample width 2) = 1024

11.61 ms interval works for chunk size of 1024 (1024 frames)

readframes 2048 (for some reason this streams correctly) stream of bytes... stream of 8-bit bytes.... 1024 2-bytes

readframes int16 => 1024 (for some reason this ffts correctly)

#tpCount     = len(amplitude)
#values      = np.arange(int(tpCount/2))
#timePeriod  = tpCount/samplingFrequency
#frequencies = values/timePeriod
# sampling frequency / sample width
#print(512/timePeriod)
'''
#filename = 'C:/Users/Escobar/Documents/Audacity/Ellie Goulding - Love Me Like You Do (Mono).wav'
filename = 'C:/Users/Escobar/Documents/Audacity/Taylor Swift - All You Had To Do Was Stay (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Desire - Under Your Spell (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Madonna - Like A Prayer (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Canon in D - Pachelbel.wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Blink 182 - All Of This (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Green Day - Whatsername (Mono).wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Darker Than Black Ending 1.wav'
#filename = 'C:/Users/Escobar/Documents/Audacity/Boogiepop - Boogiepop and Others (Mono).wav'

chunk = 1024
wav_object = wave.open(filename, 'rb')
sample_rate = wav_object.getframerate()
sample_width = wav_object.getsampwidth()
num_frames = wav_object.getnframes()

samplingFrequency = 44100;
samplingInterval = 1 / samplingFrequency
beginTime = 0
endTime = samplingInterval*(chunk//2)

def getFFT(data, rate, chunk_size, log_scale=False):
    data = data * np.hamming(len(data))
    try:
        FFT = np.abs(np.fft.rfft(data)[1:])
        FFT = np.split(np.abs(FFT), 2)[0]
    except:
        FFT = np.fft.fft(data)
        left, right = np.split(np.abs(FFT), 2)
        FFT = np.add(left, right[::-1])

    fftx = np.fft.rfftfreq(chunk_size, d=1.0/rate)[1:]
    fftx = np.split(np.abs(fftx), 2)[0]
    
    if log_scale:
        try:
            FFT = np.multiply(len(FFT), np.log10(FFT))
        except Exception as e:
            print('Log(FFT) failed: %s' %str(e))

    return FFT, fftx

def downsample(data, bin_size):
    overhang=len(data)%bin_size
    if overhang: data=data[:-overhang]
    data=np.reshape(data,(len(data)//bin_size,bin_size))
    data=np.average(data,1)
    return data

#TODO: USE A ZIP LOOP TO LOOP THROUGH HALVES AND CORRESPONDING BIN WIDTHS (data[0],8), (data[1], 16), data(2,32)...
def partition_log2(data):
    
    partitions = np.zeros(16)
    #TODO: find faster/better way to do this, maybe (data, [8,16,24,32...]) then average([0:64],8)
    #split_data = np.split(data, [64,128,192,256,384])
    split_data = np.split(data, [32,64,96,128,256])

    #eight, one_six, three_two = np.split(split_data[0],8), np.split(split_data[1],4), np.split(split_data[2], 2)
    four = np.split(split_data[0],8)
    eight = np.split(split_data[1],4)
    one_six = np.split(split_data[2], 2)
    three_two = split_data[3]
    six_four = np.split(split_data[4], 2) #two arrays

    four_avg = np.average(four,1)
    eight_avg = np.average(eight,1)
    one_six_avg = np.average(one_six,1)
    three_two_avg = np.average(three_two,0)
    six_four_avg = np.average(six_four,1)

    partitions[:8] = four_avg
    partitions[8:12] = eight_avg
    partitions[12:14] = one_six_avg
    partitions[14:16] = six_four_avg
    
    return partitions

#WHEN WRITING TO STREAM INT8 SOUNDS APPROPRIATE
#WHEN TAKING FFT INT16 LOOKS RIGHT
def audio_chunk(audio_segment, chunk_size):
    data = np.frombuffer(audio_segment.readframes(chunk_size), dtype=np.int16)
    return data

p = pyaudio.PyAudio()

stream = p.open(format = 8,
                channels = wav_object.getnchannels(),
                rate = sample_rate,
                output = True)



data = audio_chunk(wav_object,chunk)
fft, fftx = getFFT(data, sample_rate, chunk, log_scale=False)
timespan = np.arange(beginTime, endTime, samplingInterval)
parttime = np.arange(0, 16, 1)


fig, ax = plotter.subplots(2,1)
plotter.subplots_adjust(hspace=1)
xdata, ydata = [], []
line0, = ax[0].plot([],[])
line1, = ax[1].plot([],[])
line = [line0, line1]

def init():
    ax[0].set_title('time domain')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlim(0, endTime)
    ax[0].set_ylim(-12000, 12000)
    
    ax[1].set_title('frequency domain')
    ax[1].set_xlabel('power')
    ax[1].set_ylabel('Frequency')

    ax[1].set_xlim(0,fftx[-1])
    ax[1].set_ylim(0,10)

    return line


def update(frame):

    stream_data = wav_object.readframes(chunk)
    stream.write(stream_data)
    
    buffer_data = np.frombuffer(stream_data, dtype=np.int16)
    fft, fftx = getFFT(buffer_data, sample_rate, chunk, log_scale=False)
    fft = np.log1p(fft/len(buffer_data))
    time_data = downsample(buffer_data, 2)

    line[0].set_data(timespan, time_data)
    line[1].set_data(fftx,fft)

    return line

ani = FuncAnimation(fig, update, frames=num_frames,
                    init_func=init, repeat=False, blit=True, interval=11.6, cache_frame_data=False)
plotter.show()