import matplotlib.pyplot as plotter
from matplotlib.animation import FuncAnimation
import pyaudio
import wave
import numpy as np
import time

class Audio_Parser():

    def __init__(self, path, chunk_size):
        self.file_path = path
        self.chunk_size = chunk_size
        self.wav_obj = wave.open(path, 'rb')
        self.sample_rate = self.wav_obj.getframerate()
        self.num_frames = self.wav_obj.getnframes()
        self.p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        self.stream = self.p.open(format = self.p.get_format_from_width(self.wav_obj.getsampwidth()),
                        channels = self.wav_obj.getnchannels(),
                        rate = self.wav_obj.getframerate(),
                        output = True)

    def audio_fft(self, data, rate, chunk_size, log_scale=False):
        data = data * np.hamming(len(data))
        try:
            FFT = np.abs(np.fft.rfft(data)[1:])
        except:
            FFT = np.fft.fft(data)
            left, right = np.split(np.abs(FFT), 2)
            FFT = np.add(left, right[::-1])

        fftx = np.fft.rfftfreq(chunk_size, d=1.0/rate)[1:]
        
        if log_scale:
            try:
                FFT = np.multiply(len(FFT), np.log10(FFT))
            except Exception as e:
                print('Log(FFT) failed: %s' %str(e))

        return FFT, fftx

    #reduce size of 1D array by averaging together pairs of elements
    def downsample(self, data, bin_size):
        overhang=len(data)%bin_size
        if overhang: data=data[:-overhang]
        data=np.reshape(data,(len(data)//bin_size,bin_size))
        data=np.average(data,1)
        return data

    #TODO: MAKE THIS A PROPER FUNCTION NOT SPAGHETTI CODE
    def partition_pow2(self, bins, data):
        #bins must be in powers of 2
        partitions = np.zeros(bins)
        split_bins = [32, 64, 96, 128, 160, 192, 288]
        #TODO: find faster/better way to do this, maybe (data, [8,16,24,32...]) then average([0:64],8)
        split_data = np.split(data, split_bins)

        #eight, one_six, three_two = np.split(split_data[0],8), np.split(split_data[1],4), np.split(split_data[2], 2)
        two = np.split(split_data[0],16)
        four = np.split(split_data[1],8)
        eight = np.split(split_data[2],4)
        one_six = np.split(split_data[3], 2)
        three_two = split_data[4]
        six_four = split_data[5] 

        #possible optimization to limit sig figs
        two_avg = np.average(two,1)
        four_avg = np.average(four,1)
        eight_avg = np.average(eight,1)
        one_six_avg = np.average(one_six,1)
        three_two_avg = np.average(three_two,0)
        six_four_avg = np.average(six_four,0)

        partitions[:16] = two_avg
        partitions[16:24] = four_avg
        partitions[24:28] = eight_avg
        partitions[28:30] = one_six_avg
        partitions[30:31] = three_two_avg
        partitions[31:32] = six_four_avg        

        return partitions

    def play_chunk(self):
        #stream data comes in 8 bit integers representing left/right channels
        stream_data = self.wav_obj.readframes(self.chunk_size)
        self.stream.write(stream_data)

        #taking the fft data requires combining left/right channels into single 16 bit integer
        if len(stream_data) > 0:
            buffer_data = np.frombuffer(stream_data, dtype=np.int16)
            return buffer_data
        
        return None

    def partition_chunk(self, fft_data):
        parts = self.partition_pow2(32, fft_data)/(self.chunk_size*32)
        energy = np.sum(parts)

        parts[parts > 31] = 31
        parts[parts < 0.5] = -1
        parts = np.floor(parts).astype(int)
        return parts, energy
