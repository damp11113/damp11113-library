"""
damp11113-library - A Utils library and Easy to use. For more info visit https://github.com/damp11113/damp11113-library/wiki
Copyright (C) 2021-2023 damp11113 (MIT)

Visit https://github.com/damp11113/damp11113-library

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import itertools
import math
import random
import numpy as np
import threading
import scipy
from scipy.signal import chirp

from .convert import str2bin, str2binnparray, bin2str

print("Please Using Float32 for this library on DSP")

class OQPSKModulator:
    class Settings:
        def __init__(self):
            self.bitrate = 18500
            self.carrier_freq = 8000
            self.alpha = 0.75
            self.max_bit_buffer_size = 40000

    def __init__(self, pDspGen):
        self.pDspGen = pDspGen
        self.pWaveTableCarrier = WaveTable(pDspGen, 1000)
        self.pWaveTableSymbol = WaveTable(pDspGen, 200)
        self.dscabpf = FastFIRFilter()
        self.buffer = []
        self.buffer_head = 0
        self.buffer_tail = 0
        self.buffer_used = 0
        self.buffer_size = 0
        self.askedformoredata = False
        self.spooling = False
        self.symbol_this = 0
        self.symbol_next = 0
        self.fir_re = FastFIRFilter()
        self.fir_im = FastFIRFilter()
        self.settings = self.Settings()
        self.buffer_mutex = threading.Lock()  # Use threading.Lock() for thread synchronization
        self.RefreshSettings()

    def LoadBits(self, bits):
        try:
            self.buffer_mutex.acquire()
            if self.buffer_used + len(bits) >= self.buffer_size:
                print("OQPSKModulator::LoadBits buffer overflow!!: buffer_head=", self.buffer_head, " buffer_tail=",
                      self.buffer_tail,
                      " buffer_used=", self.buffer_used, " buffer_size=", self.buffer_size)
                self.askedformoredata = False
                return False

            self.askedformoredata = False
            for bit in bits:
                self.buffer.append(bit)
                self.buffer_head += 1
                self.buffer_head %= self.buffer_size
                self.buffer_used += 1

        finally:
            self.buffer_mutex.release()

        return True

    def StartSpooling(self):
        try:
            self.buffer_mutex.acquire()
            self.buffer_head = 0
            self.buffer_tail = 0
            self.buffer_used = 0
            self.buffer_size = len(self.buffer)
            self.askedformoredata = True
            tmpbuffer_size = self.buffer_size
            self.spooling = True

        finally:
            self.buffer_mutex.release()

        self.CallForMoreData(tmpbuffer_size)

    def StopSpooling(self):
        self.buffer_mutex.locked()
        self.spooling = False
        self.buffer_mutex.acquire()

    def update(self):
        self.pWaveTableCarrier.WTnextFrame()
        self.pWaveTableSymbol.WTnextFrame()
        symbol = 0j

        if self.pWaveTableSymbol.IfPassesPointNextTime():
            self.buffer_mutex.locked()
            if self.buffer_used:
                bit = self.buffer[self.buffer_tail] % 2
                if bit < 0:
                    bit *= bit
                self.buffer_tail += 1
                self.buffer_tail %= self.buffer_size
                self.buffer_used -= 1

                if (not self.askedformoredata) and (self.buffer_used * 2) < self.buffer_size:
                    self.askedformoredata = True
                    self.buffer_mutex.acquire()
                    self.CallForMoreData(self.buffer_size - self.buffer_used)
                else:
                    self.buffer_mutex.acquire()
            else:
                self.buffer_mutex.acquire()
                bit = random.randint(0, 1)

            sel = random.randint(0, 1)
            if sel:
                symbol = complex(2.0 * (bit - 0.5), symbol.imag)
            else:
                symbol = complex(symbol.real, 2.0 * (bit - 0.5))

        carrier = self.pWaveTableCarrier.WTCISValue()
        signal = complex(carrier.real * self.fir_re.Update_Single(symbol.real),
                         carrier.imag * self.fir_im.Update_Single(symbol.imag))

        return self.dscabpf.Update_Single(0.5 * (signal.real + signal.imag))

    def RefreshSettings(self, bitrate=None, carrier_freq=None, alpha=None, max_bit_buffer_size=None):
        if bitrate is not None:
            self.settings.bitrate = bitrate
        if carrier_freq is not None:
            self.settings.carrier_freq = carrier_freq
        if alpha is not None:
            self.settings.alpha = alpha
        if max_bit_buffer_size is not None:
            self.settings.max_bit_buffer_size = max_bit_buffer_size

        try:
            self.buffer_mutex.acquire()
            self.buffer_head = 0
            self.buffer_tail = 0
            self.buffer_used = 0
            self.buffer = [0] * self.settings.max_bit_buffer_size
            self.buffer_size = len(self.buffer)
            self.askedformoredata = False

            self.pWaveTableCarrier.RefreshSettings(self.settings.carrier_freq)
            self.pWaveTableSymbol.RefreshSettings(self.settings.bitrate)

            symbolrate = self.settings.bitrate / 2.0
            firlen = int(5 * 6 * self.pDspGen.SampleRate / symbolrate)
            self.rrc = RRCFilter()
            self.rrc.create(symbolrate, firlen, self.settings.alpha, self.pDspGen.SampleRate)
            self.rrc.scalepoints(6.1)
            self.fir_re.setKernel(self.rrc.Points)
            self.fir_im.setKernel(self.rrc.Points)

            bw = 0.5 * (1.0 + self.settings.alpha) * self.settings.bitrate + 10
            minfreq = max(self.settings.carrier_freq - bw / 2.0, 1.0)
            maxfreq = min(self.settings.carrier_freq + bw / 2.0, self.pDspGen.SampleRate / 2.0 - 1.0)
            self.dscabpf.setKernel(FilterDesign.BandPassHanning(minfreq, maxfreq, self.pDspGen.SampleRate, 256 - 1))

            self.symbol_this = 0
            self.symbol_next = 0

        finally:
            self.buffer_mutex.acquire()

    def delayedStartSpooling(self):
        self.StartSpooling()

    def isSpooling(self):
        return self.spooling

class WaveTable:
    def __init__(self, pDspGen, frequency):
        self.pDspGen = pDspGen
        self.frequency = frequency

    def WTnextFrame(self):
        pass

    def IfPassesPointNextTime(self):
        return True

    def RefreshSettings(self, freq):
        self.frequency = freq

class FastFIRFilter:
    def __init__(self):
        self.kernel = None

    def setKernel(self, kernel):
        self.kernel = kernel

    def Update_Single(self, value):
        return value

class FilterDesign:
    @staticmethod
    def BandPassHanning(minfreq, maxfreq, sample_rate, length):
        return np.zeros(length)

class TDspGen:
    def __init__(self):
        self.SampleRate = 0  # Your desired sample rate value

class RRCFilter:
    def __init__(self):
        self.Points = None

    def create(self, symbolrate, firlen, alpha, sample_rate):
        # Implementation of RRC filter creation
        pass

    def scalepoints(self, scale_factor):
        # Implementation of scaling the RRC filter points
        pass

# use paFloat32
def FSKEncoder(data, samplerate=48000, baudrate=100, tone1=1000, tone2=2000):
    duration = 1 / baudrate  # in seconds

    birn = str2bin(data)
    bits = list(birn)
    for i in range(len(bits)):
        bits[i] = int(bits[i])

    # Convert the bit sequence to a sequence of frequencies
    freqs = [tone1 if bit == 0 else tone2 for bit in bits]

    # Generate the time vector for the output signal
    t = np.linspace(0, duration, round(duration * samplerate), False)

    # Generate the FSK signal by alternating between the two frequencies
    signal = np.concatenate([np.sin(2 * np.pi * f * t) for f in freqs])

    # Normalize the signal to the range [-1, 1]
    signal /= np.max(np.abs(signal))

    return signal

def FSKEncoderV2(data, samplerate=48000, baudrate=100, tone1=1000, tone2=2000):
    samples_per_bit = 1.0 / baudrate * samplerate

    birn = str2bin(data)
    bits = list(birn)
    for i in range(len(bits)):
        bits[i] = int(bits[i])

    # Convert the bit sequence to a sequence of frequencies
    freqs = [tone1 if bit == 0 else tone2 for bit in bits]

    bit_arr = np.array(freqs)

    symbols_freqs = np.repeat(bit_arr, samples_per_bit)

    t = np.arange(0, len(symbols_freqs) / samplerate, 1.0 / samplerate)
    #return np.sin(2.0 * np.pi * symbols_freqs * (t))

    # New lines here demonstrating continuous phase FSK (CPFSK)
    delta_phi = symbols_freqs * np.pi / (samplerate / 2.0)
    phi = np.cumsum(delta_phi)
    return np.sin(phi)


def FSKEncoderV3(data, samplerate=48000, baudrate=100, tone1=1000, tone2=2000):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    byte_data = np.zeros(0)

    for char in data:
        for i in range(0, 8):
            bit = (ord(char) >> i) & 1
            if bit:
                roffle = np.sin(2 * np.pi * tone2 * np.arange(samples_per_bit) / samplerate)
                byte_data = np.append(byte_data, roffle * 0.8)
            else:
                sinewave = np.sin(2 * np.pi * tone1 * np.arange(samples_per_bit) / samplerate)
                byte_data = np.append(byte_data, sinewave)

    return byte_data

def FSKEncoderV4(data, samplerate=48000, baudrate=100, center_freq=1500, freq_shift=500):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    byte_data = np.zeros(0)

    for char in data:
        for i in range(0, 8):
            bit = (ord(char) >> i) & 1
            if bit:
                frequency = center_freq + freq_shift
            else:
                frequency = center_freq - freq_shift

            sinewave = np.sin(2 * np.pi * frequency * np.arange(samples_per_bit) / samplerate)
            byte_data = np.append(byte_data, sinewave)

    return byte_data

def FSKDecoder(signal, samplerate=48000, baudrate=100, center_freq=1500, freq_shift=500):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    num_samples = len(signal)

    bit_stream = ''
    i = 0

    while i < num_samples:
        sample = signal[i:i + samples_per_bit]
        fft = np.fft.fft(sample)
        freqs = np.fft.fftfreq(len(fft), 1 / samplerate)

        # Find the peak frequency in the FFT
        peak_frequency_index = np.argmax(np.abs(fft))
        peak_frequency = np.abs(freqs[peak_frequency_index])

        # Determine bit based on frequency deviation from center frequency
        if peak_frequency > (center_freq + freq_shift / 2):
            bit_stream += '1'
        else:
            bit_stream += '0'

        i += samples_per_bit

    # Convert bit stream to characters
    decoded_data = ''.join([chr(int(bit_stream[i:i + 8], 2)) for i in range(0, len(bit_stream), 8)])

    return decoded_data


def FSKDecoderV2(encoded_data, samplerate=48000, baudrate=100, tone1=1000, tone2=2000, threshold=0.5):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)

    decoded_data = ""
    i = 0

    while i < len(encoded_data):
        chunk = encoded_data[i:i + samples_per_bit]

        # Compute Fourier Transform to identify frequencies in the chunk
        fft_result = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(fft_result), 1 / samplerate)

        # Find dominant frequencies in the chunk
        tone1_indices = np.where((freqs >= tone1 - 50) & (freqs <= tone1 + 50))[0]
        tone2_indices = np.where((freqs >= tone2 - 50) & (freqs <= tone2 + 50))[0]

        if len(tone1_indices) > len(tone2_indices):
            decoded_data += '0'
        else:
            decoded_data += '1'

        i += samples_per_bit

    # Convert binary string to ASCII characters
    decoded_text = ""
    for j in range(0, len(decoded_data), 8):
        byte = decoded_data[j:j + 8]
        decoded_text += chr(int(byte, 2))

    return decoded_text

def MFSKEncoder(data, samplerate=48000, baudrate=100, frequencies=[1000, 2000, 3000, 4000, 5000], symbol_length=2):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    # symbol_length Two bits at a time for 4 combinations with 3 frequencies
    num_symbols = 2 ** symbol_length

    encoded_data = np.zeros(0)

    for char in data:
        for i in range(0, 8, symbol_length):
            bits = [(ord(char) >> j) & 1 for j in range(i, i + symbol_length)]
            symbol_index = sum([bits[j - i] * (2 ** j) for j in range(i, i + symbol_length)])

            # Ensure symbol_index is within the valid range of indices
            symbol_index = symbol_index % num_symbols

            # Ensure frequencies list has enough elements
            if symbol_index < len(frequencies):
                sinewave = np.sin(2 * np.pi * frequencies[symbol_index] * np.arange(samples_per_bit) / samplerate)
                encoded_data = np.append(encoded_data, sinewave)
            else:
                print(f"Symbol index {symbol_index} out of range for frequencies list.")

    return encoded_data
def tonegen(freq, duration, samplerate=48000):
    t = np.linspace(0, duration, int(samplerate * duration), False)
    return np.sin(2 * np.pi * freq * t)

def RTtonegen(frequency, amplitude=1, sample_rate=48000, buffer=1024):
    t = np.arange(0, buffer) / sample_rate
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return sine_wave

def RTHighPass(signal, cutoff_freq, sample_rate=48000):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
    filtered_signal = scipy.signal.lfilter(b, a, signal)
    return filtered_signal

def RTLowPass(signal, cutoff_freq, sample_rate=48000):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_signal = scipy.signal.lfilter(b, a, signal)
    return filtered_signal

def RTBandPass(signal, low_cutoff, high_cutoff, sample_rate=48000):
    nyquist_freq = 0.5 * sample_rate
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq
    b, a = scipy.signal.butter(4, [low, high], btype='band', analog=False)
    filtered_signal = scipy.signal.lfilter(b, a, signal)
    return filtered_signal

def RTAGC(signal, target_rms):
    rms = np.sqrt(np.mean(signal**2))
    gain = target_rms / rms
    agc_signal = signal * gain
    return agc_signal

def RTAdd(signal1, signal2, mix_ratio=0.5):
    mixed_signal = signal1 * (1 - mix_ratio) + signal2 * mix_ratio
    return mixed_signal

def RTAddV2(signal1, signal2, mix_ratio=0.5):
    signal1 = np.asarray(signal1, dtype=np.float32)  # Convert to float32 if not already
    signal2 = np.asarray(signal2, dtype=np.float32)  # Convert to float32 if not already

    if signal1.shape != signal2.shape:
        raise ValueError("Input signals must have the same shape")

    mixed_signal = signal1 * (1 - mix_ratio) + signal2 * mix_ratio
    return mixed_signal

def RTSubtract(signal1, signal2):
    subtracted_signal = signal1 - signal2
    return subtracted_signal

def RTDemphasis(signal, tau=50e-6, sample_rate=48000):
    # Generate de-emphasis filter coefficients
    alpha = np.exp(-1 / (sample_rate * tau))
    b = [1 - alpha]
    a = [1, -alpha]

    # Apply the filter
    emphasized_signal = scipy.signal.lfilter(b, a, signal)
    return emphasized_signal

def RTResample(signal, original_rate, target_rate):
    resampled_signal = scipy.signal.resample_poly(signal, target_rate, original_rate)
    return resampled_signal

def RTResampleV2(pcm, desired_samples, original_samples, dataFormat):

    samples_to_pad = desired_samples - original_samples

    q, r = divmod(desired_samples, original_samples)
    times_to_pad_up = q + int(bool(r))
    times_to_pad_down = q

    pcmList = [pcm[i:i+dataFormat] for i in range(0, len(pcm), dataFormat)]

    if samples_to_pad > 0:
        # extending pcm times_to_pad times
        pcmListPadded = list(itertools.chain.from_iterable(
            itertools.repeat(x, times_to_pad_up) for x in pcmList)
            )
    else:
        # shrinking pcm times_to_pad times
        if times_to_pad_down > 0:
            pcmListPadded = pcmList[::(times_to_pad_down)]
        else:
            pcmListPadded = pcmList

    padded_pcm = ''.join(pcmListPadded[:desired_samples])

    return padded_pcm


def RTResampleV3(input_data, input_sample_rate, output_sample_rate):
    input_array = np.frombuffer(input_data, dtype=np.int16)

    # Calculate the length of the output array based on resampling ratio
    output_length = int(len(input_array) * output_sample_rate / input_sample_rate)

    # Resample using scipy's resample function
    output_array = scipy.signal.resample(input_array, output_length)

    # Convert the output array to binary data
    output_data = output_array.astype(np.int16).tobytes()

    return output_data

def RTComClipper(signal, low_cutoff, high_cutoff, sample_rate=48000):
    nyquist_freq = 0.5 * sample_rate
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq

    b_high, a_high = scipy.signal.butter(4, low, btype='high', analog=False)
    b_low, a_low = scipy.signal.butter(4, high, btype='low', analog=False)

    high_passed_signal = scipy.signal.lfilter(b_high, a_high, signal)
    clipped_signal = scipy.signal.lfilter(b_low, a_low, high_passed_signal)
    return clipped_signal

def RTLimiter(signal, threshold):
    limited_signal = np.clip(signal, -threshold, threshold)
    return limited_signal

# Real-time compressor function
def RTCompressor(signal, threshold, ratio):
    compressed_signal = np.where(np.abs(signal) > threshold, signal * ratio, signal)
    return compressed_signal

# QAM Constellation points for 16-QAM
qam_points = {
    (0, 0): -3 - 3j,
    (0, 1): -3 - 1j,
    (0, 2): -3 + 3j,
    (0, 3): -3 + 1j,
    (1, 0): -1 - 3j,
    (1, 1): -1 - 1j,
    (1, 2): -1 + 3j,
    (1, 3): -1 + 1j,
    (2, 0):  3 - 3j,
    (2, 1):  3 - 1j,
    (2, 2):  3 + 3j,
    (2, 3):  3 + 1j,
    (3, 0):  1 - 3j,
    (3, 1):  1 - 1j,
    (3, 2):  1 + 3j,
    (3, 3):  1 + 1j,
    # Add more constellation points if needed
}

def QAMGenerator(data, bitrate=56000, symbolrate=5600, carrierfreq=10000, samplerate=192000, qampoints=qam_points):
    binary_data = str2binnparray(data)
    num_samples = len(binary_data)
    duration = num_samples / bitrate  # Calculate duration based on data length and bit rate

    # Convert binary data to QAM symbols
    qam_symbols = [qampoints[(binary_data[i], binary_data[i + 1])] for i in range(0, len(binary_data), 2)]

    # Generate time values
    t = np.linspace(0, duration, num_samples)

    # Generate the modulated signal
    qam_signal = np.zeros(num_samples, dtype=np.complex128)  # Use np.complex128 for complex numbers
    for i, symbol in enumerate(qam_symbols):
        qam_signal[i * int(bitrate / symbolrate):(i + 1) * int(bitrate / symbolrate)] = symbol

    # Generate carrier signal
    carrier = np.exp(1j * 2 * np.pi * carrierfreq * t)

    # Modulated signal
    return qam_signal * carrier

def mono2iq(monosignal):
    i_channel = np.real(monosignal)
    q_channel = np.imag(monosignal)

    # Normalize the I and Q channels separately
    i_normalized = i_channel / np.max(np.abs(i_channel))
    q_normalized = q_channel / np.max(np.abs(q_channel))

    # Combine I and Q channels into stereo signal
    return np.column_stack((i_normalized, q_normalized))

def getDBFS(audio_array, full_scale=1):
    # Calculate the RMS value of the audio data
    rms = np.sqrt(np.mean(np.square(audio_array)))

    # Calculate dBFS
    dbfs = 20 * math.log10(rms / full_scale)

    return dbfs

def RTCompressor2(sample_data, Threshold=-20, Knee=10, Ratio=2, Attack=0.01, Release=0.1, Gain=1):

    # Convert threshold to linear scale
    threshold = 10 ** (Threshold / 20)

    # Convert knee width to linear scale
    knee_width = 10 ** (Knee / 20)

    # Initialize gain reduction and envelope variables
    gain_reduction = np.zeros_like(sample_data, dtype=np.float32)
    envelope = np.zeros_like(sample_data, dtype=np.float32)

    # Process the entire signal at once
    abs_sample_data = np.abs(sample_data)
    envelope = (1 - Attack) * envelope + Attack * abs_sample_data

    # Calculate the compression gain reduction
    above_threshold = envelope >= threshold
    gain_reduction[above_threshold] = (1 - (1 / Ratio)) * ((envelope[above_threshold] / threshold) - 1) ** 2
    above_knee = envelope > (threshold * knee_width)
    gain_reduction[above_knee] -= (1 - (1 / Ratio)) * ((knee_width - 1) ** 2)

    # Apply makeup gain and gain reduction
    compressed_audio = sample_data / (10 ** (Gain / 20)) * (1 - gain_reduction)

    return compressed_audio


def RTEqualizer(sample_data, bands, sample_rate=48000):
    # Check if sample_data is a numpy array, if not, convert it to one
    if not isinstance(sample_data, np.ndarray):
        sample_data = np.array(sample_data, dtype=np.float32)

    # Ensure the input audio data has the correct shape (n_samples, n_channels)
    if sample_data.ndim == 1:
        sample_data = sample_data[:, np.newaxis]

    n_samples, n_channels = sample_data.shape

    # Create arrays to store the equalized audio data for each channel
    equalized_audio_data = np.zeros_like(sample_data)

    for channel in range(n_channels):
        # Get the audio data for the current channel
        channel_data = sample_data[:, channel]

        # Calculate the FFT of the input audio data
        fft_data = np.fft.fft(channel_data)

        # Initialize an array to store the equalization filter
        equalization_filter = np.ones(len(fft_data), dtype=np.complex64)

        for band, gain in bands:
            center_freq = band
            bandwidth = 10  # Adjust this value as needed

            # Calculate the lower and upper frequencies of the band
            lower_freq = center_freq - (bandwidth / 2)
            upper_freq = center_freq + (bandwidth / 2)

            # Calculate the indices corresponding to the band in the FFT data
            lower_index = int(lower_freq * len(fft_data) / sample_rate)  # Adjust the sample rate if necessary
            upper_index = int(upper_freq * len(fft_data) / sample_rate)  # Adjust the sample rate if necessary

            # Apply the gain to the equalization filter within the band
            equalization_filter[lower_index:upper_index] *= 10 ** (gain / 20.0)

        # Apply the equalization filter to the FFT data
        equalized_fft_data = fft_data * equalization_filter

        # Calculate the inverse FFT to get the equalized audio data for the channel
        equalized_channel_data = np.fft.ifft(equalized_fft_data)

        # Ensure the resulting audio data is real and within the valid range
        equalized_channel_data = np.real(equalized_channel_data)
        equalized_channel_data = np.clip(equalized_channel_data, -1.0, 1.0)

        # Store the equalized audio data for the channel
        equalized_audio_data[:, channel] = equalized_channel_data

    return equalized_audio_data

DTMF_FREQUENCIES = {
    '1': (697, 1209),
    '2': (697, 1336),
    '3': (697, 1477),
    '4': (770, 1209),
    '5': (770, 1336),
    '6': (770, 1477),
    '7': (852, 1209),
    '8': (852, 1336),
    '9': (852, 1477),
    '0': (941, 1336),
    '*': (941, 1209),
    '#': (941, 1477),
    'A': (697, 1633),
    'B': (770, 1633),
    'C': (852, 1633),
    'D': (941, 1633)
}

def generate_dtmf_tone(input_string, duration, sampling_rate=44100, amplitude=1.0, DTMF_freqlist=DTMF_FREQUENCIES):
    dtmf_signal = np.array([])

    for char in input_string:
        char_upper = char.upper()
        if char_upper in DTMF_freqlist:
            frequencies = DTMF_freqlist[char_upper]
            t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
            tone = amplitude * np.sin(2 * np.pi * frequencies[0] * t) + amplitude * np.sin(
                2 * np.pi * frequencies[1] * t)
            dtmf_signal = np.concatenate([dtmf_signal, tone])

    return dtmf_signal

def MSKEncoder(data, samplerate=48000, baudrate=100, tone=1500):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    phase_shift = np.pi / 2  # Phase shift for MSK

    byte_data = np.zeros(0)

    for char in data:
        for i in range(0, 8):
            bit = (ord(char) >> i) & 1
            phase = 0  # Initial phase

            # MSK modulation
            for j in range(samples_per_bit // 2):
                if bit:
                    phase += 2 * np.pi * tone / samplerate
                else:
                    phase -= 2 * np.pi * tone / samplerate

                sample = np.sin(phase + phase_shift)
                byte_data = np.append(byte_data, sample)

    return byte_data

def MSKEncoderv2(data, samplerate=48000, baudrate=100, tone=1500):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    phase_shift = np.pi / 2  # Phase shift for MSK

    # Convert input string to binary representation
    binary_data = np.unpackbits(np.array([ord(char) for char in data], dtype=np.uint8))

    # Generate phase array
    phase = np.cumsum(2 * np.pi * tone / samplerate * binary_data) + phase_shift

    # Generate samples using sin function
    samples = np.sin(phase)

    return samples

def ASKEncoder(data, samplerate=48000, baudrate=100, tone1=1000, tone2=2000):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    encoded_signal = np.array([])

    for char in data:
        for i in range(0, 8):
            bit = (ord(char) >> i) & 1
            if bit:
                signal = np.sin(2 * np.pi * tone2 * np.arange(samples_per_bit) / samplerate)
            else:
                signal = np.sin(2 * np.pi * tone1 * np.arange(samples_per_bit) / samplerate)
            encoded_signal = np.append(encoded_signal, signal)

    return encoded_signal

def ASKEncoderv2(data, samplerate=48000, baudrate=100, tone1=1000, tone2=2000):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    total_samples = len(data) * 8 * samples_per_bit
    encoded_signal = np.empty(total_samples)

    tone1_signal = np.sin(2 * np.pi * tone1 * np.arange(samples_per_bit) / samplerate)
    tone2_signal = np.sin(2 * np.pi * tone2 * np.arange(samples_per_bit) / samplerate)
    bit_mask = np.array([1 << i for i in range(8)])

    index = 0
    for char in data:
        bits = np.bitwise_and(np.right_shift(np.array([ord(char)], dtype=np.uint8), bit_mask), 1)
        for bit in bits:
            if bit:
                encoded_signal[index:index + samples_per_bit] = tone2_signal
            else:
                encoded_signal[index:index + samples_per_bit] = tone1_signal
            index += samples_per_bit

    return encoded_signal

def PSKEncoder(data, samplerate=48000, baudrate=100, carrier_freq=1000):
    t = 1.0 / baudrate
    samples_per_bit = int(t * samplerate)
    byte_data = np.zeros(0)

    for char in data:
        for i in range(0, 8):
            bit = (ord(char) >> i) & 1
            phase_shift = 0 if bit == 0 else np.pi  # Phase shift for BPSK (0 or pi radians)
            sinewave = np.sin(2 * np.pi * carrier_freq * np.arange(samples_per_bit) / samplerate + phase_shift)
            byte_data = np.append(byte_data, sinewave)

    return byte_data

def PSKDecoder(signal, samplerate=48000, baudrate=100, phases=2, threshold=0):
    samples_per_bit = int((1.0 / baudrate) * samplerate)
    bits = []

    for i in range(0, len(signal), samples_per_bit):
        segment = signal[i:i+samples_per_bit]
        phase = np.angle(np.sum(segment) / len(segment), deg=True)  # Calculate phase in degrees
        decoded_bit = int(round(phase / (360 / phases))) % phases  # Map phase to 0 to phases-1
        bits.append(decoded_bit)

    decoded_data = "".join(str(bit) for bit in bits)
    return bin2str(decoded_data)

def BPSKEncoder(binary_data, bit_rate, fc, sample_rate):
    T = 1 / bit_rate
    t = np.linspace(0, T, int(sample_rate * T))
    carrier = np.cos(2 * np.pi * fc * t)

    modulated_signal = np.array([])
    for bit in binary_data:
        if bit == 0:
            modulated_signal = np.append(modulated_signal, carrier)
        else:
            modulated_signal = np.append(modulated_signal, -carrier)

    return modulated_signal

def BPSKDecoder(modulated_signal, bit_rate, fc, sample_rate):
    T = 1 / bit_rate
    t = np.linspace(0, T, int(sample_rate * T))
    carrier = np.cos(2 * np.pi * fc * t)

    demodulated_signal = np.array([])
    for i in range(0, len(modulated_signal), len(t)):
        dot_product = np.dot(modulated_signal[i:i + len(t)], carrier)
        if dot_product > 0:
            demodulated_signal = np.append(demodulated_signal, 0)
        else:
            demodulated_signal = np.append(demodulated_signal, 1)

    return demodulated_signal.astype(int)

def OFDMEncoder(data, samplerate=48000, subcarrier_frequency=1000, cyclic_prefix_length=0.25):
    symbol_duration = 1 / subcarrier_frequency  # Symbol duration in seconds
    subcarrier_spacing = 1 / symbol_duration  # Subcarrier spacing in Hz
    symbol_length = int(symbol_duration * samplerate)  # Length of each OFDM symbol in samples
    cyclic_prefix_samples = int(cyclic_prefix_length * symbol_length)  # Length of cyclic prefix in samples

    encoded_symbols = []
    for char in data:
        binary_data = format(ord(char), '08b')  # Convert character to 8-bit binary representation
        for i in range(0, 8, 2):
            bit_pair = binary_data[i:i + 2]
            if bit_pair == '00':
                subcarrier_frequency_i = subcarrier_frequency
            elif bit_pair == '01':
                subcarrier_frequency_i = subcarrier_frequency + subcarrier_spacing
            elif bit_pair == '10':
                subcarrier_frequency_i = subcarrier_frequency + 2 * subcarrier_spacing
            else:
                subcarrier_frequency_i = subcarrier_frequency + 3 * subcarrier_spacing

            subcarrier = np.sin(2 * np.pi * subcarrier_frequency_i * np.arange(symbol_length) / samplerate)
            symbol_with_prefix = np.concatenate((subcarrier[-cyclic_prefix_samples:], subcarrier))
            encoded_symbols.append(symbol_with_prefix)

    transmitted_signal = np.concatenate(encoded_symbols)
    return transmitted_signal

def OFDMDecoder(received_signal, samplerate=48000, subcarrier_frequency=1000, cyclic_prefix_length=0.25):
    symbol_duration = 1 / subcarrier_frequency
    subcarrier_spacing = 1 / symbol_duration
    symbol_length = int(symbol_duration * samplerate)
    cyclic_prefix_samples = int(cyclic_prefix_length * symbol_length)

    ofdm_symbols = []
    for i in range(0, len(received_signal), symbol_length + cyclic_prefix_samples):
        symbol_with_prefix = received_signal[i:i + symbol_length + cyclic_prefix_samples]
        symbol_without_prefix = symbol_with_prefix[cyclic_prefix_samples:]
        ofdm_symbols.append(symbol_without_prefix)

    decoded_data = ""
    for symbol in ofdm_symbols:
        subcarrier_fft = np.fft.fft(symbol)
        subcarrier_magnitudes = np.abs(subcarrier_fft[1:int(symbol_length / 2)])  # Exclude DC component
        subcarrier_indices = np.argsort(subcarrier_magnitudes)[-2:] + 1  # Get the indices of the two largest subcarriers
        bit_pair = ""
        for index in subcarrier_indices:
            # Determine bit pair based on subcarrier indices
            if index == int(subcarrier_frequency / subcarrier_spacing) + 1:
                bit_pair += "00"
            elif index == int((subcarrier_frequency + subcarrier_spacing) / subcarrier_spacing) + 1:
                bit_pair += "01"
            elif index == int((subcarrier_frequency + 2 * subcarrier_spacing) / subcarrier_spacing) + 1:
                bit_pair += "10"
            else:
                bit_pair += "11"
        decoded_char = chr(int(bit_pair, 2))
        decoded_data += decoded_char

    return decoded_data