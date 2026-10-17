import numpy as np
import matplotlib.pyplot as plt

def generate_radar_chirp(f0=1e6, f1=10e6, T=1e-4, fs=50e6):
    """
    Simulates a Linear Frequency Modulated (LFM / Chirp) Radar Pulse.
    Used in pulse compression for high range-resolution radar detection.
    """
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    k = (f1 - f0) / T  # Chirp rate
    signal = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2))
    
    # Add Gaussian Noise (channel noise)
    noise = 0.5 * np.random.normal(size=len(t))
    noisy_signal = signal + noise
    
    # FFT Analysis
    fft_spectrum = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(t), 1/fs)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(t[:500] * 1e6, noisy_signal[:500], color='#0969da')
    ax1.set_title("Time-Domain Chirp Radar Signal with Noise")
    ax1.set_xlabel("Time (microseconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    half_n = len(freqs) // 2
    ax2.plot(freqs[:half_n] * 1e-6, fft_spectrum[:half_n], color='#1a7f37')
    ax2.set_title("Frequency Spectrum (FFT Analysis)")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("radar_modulation_output.png", dpi=300)
    print("Grafik 'radar_modulation_output.png' olarak kaydedildi.")

if __name__ == "__main__":
    generate_radar_chirp()
