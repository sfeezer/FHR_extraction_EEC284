import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def plot_psd(data, fs, title="Power Spectral Density", x_lim=None, ax=None):
    """
    Plots the Power Spectral Density of a signal using Welch's method.
    """
    frequencies, psd = signal.welch(data, fs, nperseg=1024)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.semilogy(frequencies, psd)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V**2/Hz)')
    if x_lim:
        ax.set_xlim(x_lim)
    ax.grid(True)
    
    if ax is None:
        plt.show()

def plot_spectrogram(data, fs, title="Spectrogram", freq_lim=None, ax=None):
    """
    Plots the Spectrogram of a signal with Frequency on X and Time on Y.
    """
    frequencies, times, Sxx = signal.spectrogram(data, fs)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Transpose Sxx and swap times/frequencies to switch axes
    pcm = ax.pcolormesh(frequencies, times, 10 * np.log10(Sxx.T), shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (sec)')
    
    if freq_lim:
        ax.set_xlim(freq_lim)
    
    if ax is None:
        plt.colorbar(pcm, label='Intensity (dB)')
        plt.show()
    else:
        return pcm

def plot_stage1_validation(raw_data, filt_data, fs, channel_name):
    """
    Plots side-by-side PSDs and Spectrograms for pre and post filtering.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Row 1: PSDs
    plot_psd(raw_data, fs, title=f"Pre-Filter PSD ({channel_name})", x_lim=[0, 20], ax=axs[0, 0])
    plot_psd(filt_data, fs, title=f"Post-Filter PSD ({channel_name})", x_lim=[0, 20], ax=axs[0, 1])
    
    # Row 2: Spectrograms
    pcm1 = plot_spectrogram(raw_data, fs, title=f"Pre-Filter Spectrogram ({channel_name})", freq_lim=[0, 20], ax=axs[1, 0])
    pcm2 = plot_spectrogram(filt_data, fs, title=f"Post-Filter Spectrogram ({channel_name})", freq_lim=[0, 20], ax=axs[1, 1])
    
    fig.colorbar(pcm1, ax=axs[1, 0], label='Intensity (dB)')
    fig.colorbar(pcm2, ax=axs[1, 1], label='Intensity (dB)')
    
    plt.tight_layout()
    plt.show()

def plot_stage2_spectral_validation(mixed_data, cleaned_data, ref_data, fs, channel_name, ref_name):
    """
    Plots side-by-side PSDs and Spectrograms for pre-ANC, post-ANC, and Reference.
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    
    # Row 1: PSDs
    plot_psd(mixed_data, fs, title=f"Pre-ANC PSD ({channel_name})", x_lim=[0, 10], ax=axs[0, 0])
    plot_psd(cleaned_data, fs, title=f"Post-ANC PSD ({channel_name})", x_lim=[0, 10], ax=axs[0, 1])
    plot_psd(ref_data, fs, title=f"Reference PSD ({ref_name})", x_lim=[0, 10], ax=axs[0, 2])
    
    # Row 2: Spectrograms
    pcm1 = plot_spectrogram(mixed_data, fs, title=f"Pre-ANC Spectrogram ({channel_name})", freq_lim=[0, 10], ax=axs[1, 0])
    pcm2 = plot_spectrogram(cleaned_data, fs, title=f"Post-ANC Spectrogram ({channel_name})", freq_lim=[0, 10], ax=axs[1, 1])
    pcm3 = plot_spectrogram(ref_data, fs, title=f"Reference Spectrogram ({ref_name})", freq_lim=[0, 10], ax=axs[1, 2])
    
    fig.colorbar(pcm1, ax=axs[1, 0], label='Intensity (dB)')
    fig.colorbar(pcm2, ax=axs[1, 1], label='Intensity (dB)')
    fig.colorbar(pcm3, ax=axs[1, 2], label='Intensity (dB)')
    
    plt.tight_layout()
    plt.show()

def plot_stage3_psd_bpm(freqs, psd, title="Yule-Walker PSD", bpm_lim=[0, 300], ax=None):
    """
    Plots PSD on a BPM scale (BPM = Hz * 60).
    """
    is_standalone = ax is None
    bpm = freqs * 60
    
    if is_standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(bpm, psd)
    ax.set_title(title)
    ax.set_xlabel('Heart Rate (BPM)')
    ax.set_ylabel('Power Density')
    if bpm_lim:
        ax.set_xlim(bpm_lim)
    ax.grid(True)
    
    if is_standalone:
        plt.savefig("stage3_validation.png")
        print("  Stage 3 plot saved to stage3_validation.png")
        plt.show()

def plot_stage3_comparison(freqs_pre, psd_pre, freqs_post, psd_post, title="Stage 3: PSD Comparison (Pre vs Post ANC)", bpm_lim=[0, 300]):
    """
    Plots two PSDs side-by-side on a BPM scale for comparison.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    bpm_pre = freqs_pre * 60
    bpm_post = freqs_post * 60
    
    # Plot Pre-ANC
    axs[0].plot(bpm_pre, psd_pre, color='r', alpha=0.7)
    axs[0].set_title("Pre-ANC (Filtered Signal)")
    axs[0].set_xlabel('Heart Rate (BPM)')
    axs[0].set_ylabel('Power Density')
    axs[0].set_xlim(bpm_lim)
    axs[0].grid(True)
    
    # Plot Post-ANC
    axs[1].plot(bpm_post, psd_post, color='g')
    axs[1].set_title("Post-ANC (Cleaned Signal)")
    axs[1].set_xlabel('Heart Rate (BPM)')
    axs[1].set_ylabel('Power Density')
    axs[1].set_xlim(bpm_lim)
    axs[1].grid(True)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("stage3_comparison.png")
    print("  Stage 3 comparison plot saved to stage3_comparison.png")
    plt.show()

def plot_stage5_validation(fhr_tracks_df, fused_fhr, ref_fhr_series, window_step, window_size, title="Stage 5: Final FHR Fusion Validation"):
    """
    Plots the final fused FHR alongside individual tracks and ground truth.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Calculate time axis for windows (seconds)
    num_windows = len(fhr_tracks_df)
    window_times = np.arange(num_windows) * window_step + window_size / 2.0
    
    # Plot individual tracks (faintly)
    # To avoid legend clutter, only label the first of each detector if needed or just representative ones
    first_pass = True
    for col in fhr_tracks_df.columns:
        label = "Individual Tracks" if first_pass else ""
        ax.plot(window_times, fhr_tracks_df[col], alpha=0.2, label=label)
        first_pass = False
        
    # Plot fused FHR
    ax.plot(window_times, fused_fhr, 'b-', linewidth=2.5, label='Fused FHR (Estimate)')
    
    # Plot Ground Truth
    if ref_fhr_series is not None:
        ax.plot(np.arange(len(ref_fhr_series)), ref_fhr_series, 'r--', linewidth=1.5, label='Ground Truth (Hemodynamics)')
    
    ax.set_title(title)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Fetal Heart Rate (BPM)')
    ax.set_ylim([100, 280]) 
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("stage5_validation.png")
    print("  Stage 5 validation plot saved to stage5_validation.png")
    plt.show()
