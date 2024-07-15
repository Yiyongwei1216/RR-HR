import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
import soundfile as sf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['DENOSIED_FOLDER'] = './denoised'

# 创建上传和降噪文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DENOSIED_FOLDER'], exist_ok=True)

def resample_audio(audio, orig_sr, target_sr=2000):
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr), target_sr

def calculate_shannon_envelope(signal, frame_size, hop_size):
    shannon_energy = -1 * (signal**2) * np.log(signal**2 + 1e-10)
    envelope = np.zeros(len(signal))
    for i in range(0, len(signal), hop_size):
        frame = shannon_energy[i:i+frame_size]
        envelope[i:i+frame_size] = np.mean(frame)
    return envelope

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def noise_reduction(audio, sr):
    bandpass_audio = bandpass_filter(audio, lowcut=25, highcut=450, fs=sr, order=6)
    bandpass_audio = np.nan_to_num(bandpass_audio, nan=0.0, posinf=0.0, neginf=0.0)
    noise_sample = bandpass_audio[:sr]
    noise_stft = librosa.stft(noise_sample)
    noise_profile = np.mean(np.abs(noise_stft), axis=1)
    audio_stft = librosa.stft(bandpass_audio)
    audio_stft_denoised = np.abs(audio_stft) - noise_profile[:, np.newaxis]
    audio_stft_denoised = np.maximum(audio_stft_denoised, 0)
    denoised_audio = librosa.istft(audio_stft_denoised * np.exp(1j * np.angle(audio_stft)))
    return denoised_audio

def load_audio_file(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def extract_envelope(audio, sr, cutoff_freq=0.4, window_size_ratio=0.1):
    envelope = np.abs(audio)
    cutoff = cutoff_freq / (sr / 2.0)
    b, a = butter(2, cutoff, btype='low')
    smooth_envelope = filtfilt(b, a, envelope)
    window_size = int(sr * window_size_ratio)
    smooth_envelope = np.convolve(smooth_envelope, np.ones(window_size) / window_size, mode='same')
    return smooth_envelope

def detect_breath_cycles(envelope, sr):
    peaks, _ = find_peaks(envelope, distance=sr*2)
    return peaks

def calculate_rr(peaks, duration):
    num_breaths = len(peaks)
    rr = (num_breaths / duration) * 60
    return round(rr, 2)

def calculate_heart_rate(audio_path):
    sound, sr = load_audio_file(audio_path)
    filtered_signal = bandpass_filter(sound, 20, 150, sr, order=5)
    frame_size = int(0.02 * sr)
    hop_size = int(0.01 * sr)
    envelope = calculate_shannon_envelope(filtered_signal, frame_size, hop_size)
    peaks, _ = find_peaks(envelope, distance=int(0.6 * sr))
    peak_intervals = np.diff(peaks) / sr
    instant_heart_rates = 60.0 / peak_intervals
    avg_heart_rate = np.mean(instant_heart_rates) if len(instant_heart_rates) > 0 else 0
    return int(avg_heart_rate)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/denoise', methods=['GET', 'POST'])
def denoise():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            audio, sr = load_audio_file(file_path)
            resampled_audio, sr_resampled = resample_audio(audio, sr, target_sr=2000)
            denoised_audio = noise_reduction(resampled_audio, sr_resampled)
            denoised_file_path = os.path.join(app.config['DENOSIED_FOLDER'], 'denoised_' + file.filename)
            sf.write(denoised_file_path, denoised_audio, sr_resampled)
            return render_template('result_denoise.html', filename='denoised_' + file.filename)
    return render_template('upload_denoise.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            heart_rate = calculate_heart_rate(file_path)
            audio, sr = load_audio_file(file_path)
            envelope = extract_envelope(audio, sr)
            peaks = detect_breath_cycles(envelope, sr)
            duration = len(audio) / sr
            rr = calculate_rr(peaks, duration)
            return render_template('result_analyze.html', heart_rate=heart_rate, rr=rr)
    return render_template('upload_analyze.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DENOSIED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
