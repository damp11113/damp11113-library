import itertools
import math
import random
import numpy as np
import threading
import scipy

from .convert import str2bin, str2binnparray

print("Please Using Float32 for this libray on DSP")

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