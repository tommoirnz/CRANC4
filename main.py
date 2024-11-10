"""
Two-Input Crosstalk Resistant Noise Canceller with GUI
Author: OpenAI's ChatGPT and Tom Moir
Date: 10/11/2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import librosa
import librosa.display
import time
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from tkinter import ttk

# Configure logging
logging.basicConfig(
    filename='noise_canceller.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class NoiseCancellerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Two-Input Crosstalk Resistant Noise Canceller")
        master.geometry("900x900")

        # Initialize variables
        self.input_file = None
        self.output_files = []
        self.Fs = None  # Sampling frequency
        self.y_input1 = None  # First input channel
        self.y_input2 = None  # Second input channel
        self.e1 = None  # Output 1
        self.e2 = None  # Output 2
        self.mixed_output_file = None  # Path to mixed audio
        self.y_mixed_data = None  # In-memory mixed audio data
        self.processing_complete = threading.Event()
        self.queue = queue.Queue()
        self.cancel_event = threading.Event()

        # GUI Layout
        self.create_widgets()

        # Pre-initialize sounddevice
        threading.Thread(target=self._initialize_sounddevice, daemon=True).start()

        # Start processing the queue
        self.master.after(100, self.process_queue)

    def create_widgets(self):
        # Frame for File Selection
        file_frame = tk.Frame(self.master)
        file_frame.pack(pady=10)

        self.browse_button = tk.Button(
            file_frame, text="Browse Audio File", command=self.browse_file, width=20, height=2)
        self.browse_button.grid(row=0, column=0, padx=10)

        self.file_label = tk.Label(file_frame, text="No file selected", wraplength=500)
        self.file_label.grid(row=0, column=1, padx=10)

        # Frame for Parameters
        param_frame = tk.Frame(self.master)
        param_frame.pack(pady=10)

        tk.Label(param_frame, text="Number of Weights (N):").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.E)
        self.weights_entry = tk.Entry(param_frame, width=10)
        self.weights_entry.insert(0, "64")  # Default value
        self.weights_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_frame, text="Step Size (mu):").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.mu_entry = tk.Entry(param_frame, width=10)
        self.mu_entry.insert(0, "0.04")  # Default value
        self.mu_entry.grid(row=1, column=1, padx=5, pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(
            param_frame, orient=tk.HORIZONTAL, length=400, mode='determinate', maximum=100)
        self.progress.grid(row=2, column=0, columnspan=2, pady=10)

        # Process Button
        self.process_button = tk.Button(
            self.master, text="Process Audio", command=self.process_audio, state=tk.DISABLED, width=20, height=2)
        self.process_button.pack(pady=10)

        # Cancel Button
        self.cancel_button = tk.Button(
            self.master, text="Cancel Processing", command=self.cancel_processing, state=tk.DISABLED, width=20, height=2)
        self.cancel_button.pack(pady=10)

        # Playback Mode Selection
        self.playback_mode = tk.StringVar(value="Stereo")
        mode_frame = tk.Frame(self.master)
        mode_frame.pack(pady=10)

        self.stereo_radio = tk.Radiobutton(
            mode_frame, text="Stereo", variable=self.playback_mode, value="Stereo")
        self.stereo_radio.pack(side=tk.LEFT, padx=10)

        self.mono_radio = tk.Radiobutton(
            mode_frame, text="Mono (Sum Channels)", variable=self.playback_mode, value="Mono")
        self.mono_radio.pack(side=tk.LEFT, padx=10)

        # Play Buttons
        play_frame = tk.Frame(self.master)
        play_frame.pack(pady=10)

        self.play_mixed_button = tk.Button(
            play_frame, text="Play Mixed Signal", command=self.play_mixed_audio,
            state=tk.DISABLED, width=20, height=2)
        self.play_mixed_button.grid(row=0, column=0, padx=10)

        self.play_output1_button = tk.Button(
            play_frame, text="Play Output Signal 1", command=lambda: self.play_audio(0),
            state=tk.DISABLED, width=20, height=2)
        self.play_output1_button.grid(row=0, column=1, padx=10)

        self.play_output2_button = tk.Button(
            play_frame, text="Play Output Signal 2", command=lambda: self.play_audio(1),
            state=tk.DISABLED, width=20, height=2)
        self.play_output2_button.grid(row=0, column=2, padx=10)

        # Status Label
        self.status_label = tk.Label(
            self.master, text="Select a stereo audio file to begin.", fg="blue")
        self.status_label.pack(pady=20)

        # Frame for Plotting
        plot_frame = tk.Frame(self.master)
        plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Matplotlib Figure
        self.figure = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def process_queue(self):
        try:
            while True:
                func, args = self.queue.get_nowait()
                func(*args)
        except queue.Empty:
            pass
        self.master.after(100, self.process_queue)

    def _initialize_sounddevice(self):
        try:
            # Play a short silent buffer to initialize sounddevice
            silent_buffer = np.zeros((1,), dtype=np.float32)
            sd.play(silent_buffer, 44100)
            sd.wait()
            logging.debug("SoundDevice initialized.")
        except Exception as e:
            self.queue.put((self.status_label.config, ({"text": f"Error initializing SoundDevice: {e}", "fg": "red"},)))
            logging.error(f"Error initializing SoundDevice: {e}")

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.aac")],
        )
        if file_path:
            self.input_file = file_path
            self.queue.put((self.file_label.config, ({"text": os.path.basename(file_path)},)))
            self.queue.put((self.status_label.config, ({"text": "Ready to process.", "fg": "green"},)))
            self.queue.put((self.process_button.config, ({"state": tk.NORMAL},)))
            logging.debug(f"Selected file: {file_path}")

    def process_audio(self):
        logging.debug("Process Audio Initiated")
        if not self.input_file:
            messagebox.showerror("No File Selected", "Please select an audio file to process.")
            return

        try:
            N = int(self.weights_entry.get())
            mu = float(self.mu_entry.get())
            if N <= 0 or mu <= 0:
                raise ValueError("N and mu must be positive numbers.")
            if N > 1000:
                raise ValueError("Number of weights (N) is too large. Please choose a value <= 1000.")
            if mu > 1.0:
                raise ValueError("Step size (mu) is too large. Please choose a value <= 1.0.")
        except ValueError as ve:
            messagebox.showerror("Invalid Parameters", f"Please enter valid positive numbers for N and mu.\n{ve}")
            logging.error(f"Invalid parameters entered: {ve}")
            return

        # Reset relevant variables
        self.output_files = []
        self.e1 = None
        self.e2 = None
        self.e1_scaled = None
        self.e2_scaled = None
        self.y_mixed_data = None
        self.mixed_output_file = None
        self.cancel_event.clear()

        # Update GUI elements
        self.queue.put((self.status_label.config, ({"text": "Processing audio...", "fg": "orange"},)))
        self.queue.put((self.process_button.config, ({"state": tk.DISABLED},)))
        self.queue.put((self.play_mixed_button.config, ({"state": tk.DISABLED},)))
        self.queue.put((self.play_output1_button.config, ({"state": tk.DISABLED},)))
        self.queue.put((self.play_output2_button.config, ({"state": tk.DISABLED},)))
        self.queue.put((self.cancel_button.config, ({"state": tk.NORMAL},)))
        self.queue.put((self.progress.config, ({"value": 0},)))  # Initialize progress to 0%

        self.processing_complete.clear()

        # Start processing in a new thread
        threading.Thread(target=self._process_audio_thread, args=(N, mu), daemon=True).start()

    def cancel_processing(self):
        logging.debug("Processing cancellation requested by user.")
        self.cancel_event.set()
        self.queue.put((self.status_label.config, ({"text": "Processing canceled by user.", "fg": "red"},)))
        self.queue.put((self.process_button.config, ({"state": tk.NORMAL},)))
        self.queue.put((self.cancel_button.config, ({"state": tk.DISABLED},)))
        self.queue.put((self.progress.config, ({"value": 0},)))

    def _process_audio_thread(self, N, mu):
        logging.debug("Audio Processing Thread Started")
        try:
            # Read audio file using librosa
            try:
                y, self.Fs = librosa.load(self.input_file, sr=None, mono=False)
                logging.debug(f"Audio File Loaded: {self.input_file}")
                logging.debug(f"Number of Samples: {y.shape[1] if y.ndim > 1 else y.shape[0]}")
                logging.debug(f"Sampling Frequency: {self.Fs} Hz")
            except Exception as e:
                self.queue.put((messagebox.showerror, ("Audio Loading Error", f"Failed to load the audio file:\n{e}")))
                self.queue.put((self.status_label.config, ({"text": "Failed to load audio.", "fg": "red"},)))
                self.queue.put((self.process_button.config, ({"state": tk.NORMAL},)))
                logging.error(f"Failed to load audio file: {e}")
                return

            # Ensure stereo
            if y.ndim == 1:
                # If mono, duplicate the channel to make it stereo
                y = np.vstack([y, y])
                logging.debug("Mono audio detected. Duplicating channel to create stereo.")
            elif y.shape[0] < 2:
                # If less than 2 channels, duplicate to make stereo
                y = np.vstack([y, y])
                logging.debug("Single channel audio detected. Duplicating channel to create stereo.")

            # Convert to shape (num_samples, num_channels)
            if y.shape[0] > 2:
                # If more than 2 channels, take the first two
                y = y[:2, :]
                logging.debug("More than two channels detected. Using the first two channels.")

            y = y.T  # Transpose to shape (num_samples, num_channels)

            # Assign channels
            self.y_input1 = y[:, 0]
            self.y_input2 = y[:, 1]

            # Save mixed audio and load into memory
            self.mixed_output_file = self.save_mixed_audio()
            logging.debug(f"Mixed Output File Path: {self.mixed_output_file}")  # Debug

            # Ensure y_mixed_data is a NumPy array
            if not isinstance(self.y_mixed_data, np.ndarray):
                raise ValueError("Mixed audio data is not a NumPy array.")

            # Initialize weights and regressor vectors
            w1 = np.zeros(N)
            w2 = np.zeros(N)
            X1 = np.zeros(N)
            X2 = np.zeros(N)

            # Initialize outputs
            self.e1 = np.zeros(len(self.y_input1))
            self.e2 = np.zeros(len(self.y_input2))

            # Adaptive Filtering
            progress_interval = max(len(self.y_input1) // 100, 1)  # Update progress per 1%
            for n in range(len(self.y_input1)):
                if self.cancel_event.is_set():
                    logging.info("Processing canceled by user.")
                    return

                # Update regressor vectors
                if n > 0:
                    X1[1:] = X1[:-1]
                    X1[0] = self.e2[n - 1]
                    X2[1:] = X2[:-1]
                    X2[0] = self.e1[n - 1]
                else:
                    X1[1:] = X1[:-1]
                    X1[0] = 0.0
                    X2[1:] = X2[:-1]
                    X2[0] = 0.0

                # Compute filter outputs
                yn1 = np.dot(w1, X1)
                yn2 = np.dot(w2, X2)

                # Monitor yn1 and yn2 with adjusted threshold
                if abs(yn1) > 10.0 or abs(yn2) > 10.0:
                    # Issue a warning to the user
                    self.queue.put((messagebox.showwarning, ("Step Size Too Large",
                                                             "The step size (mu) is too big. Please reduce mu and try again.")))
                    # Update status label
                    self.queue.put((self.status_label.config, ({"text": "Processing stopped due to large yn1/yn2.", "fg": "red"},)))
                    # Re-enable the process and disable cancel button
                    self.queue.put((self.process_button.config, ({"state": tk.NORMAL},)))
                    self.queue.put((self.cancel_button.config, ({"state": tk.DISABLED},)))
                    # Set processing as complete
                    self.processing_complete.set()
                    logging.warning("Processing stopped: yn1 or yn2 exceeded threshold.")
                    return  # Exit the processing thread

                # Compute errors
                self.e1[n] = self.y_input1[n] - yn1
                self.e2[n] = self.y_input2[n] - yn2

                # Update weights
                w1 += mu * self.e1[n] * X1
                w2 += mu * self.e2[n] * X2

                # Update progress every 1%
                if n % progress_interval == 0 and n != 0:
                    progress = int((n / len(self.y_input1)) * 100)
                    self.queue.put((self.progress.config, ({"value": progress},)))
                    self.queue.put((self.status_label.config, ({"text": f"Processing... {progress}%", "fg": "orange"},)))
                    logging.debug(f"Processing... {progress}%")

            # Save outputs
            self.output_files = self.save_outputs()

            # Update status and enable play buttons
            self.queue.put((self.status_label.config, ({"text": "Processing completed successfully.", "fg": "blue"},)))
            self.queue.put((self.play_mixed_button.config, ({"state": tk.NORMAL},)))
            self.queue.put((self.play_output1_button.config, ({"state": tk.NORMAL},)))
            self.queue.put((self.play_output2_button.config, ({"state": tk.NORMAL},)))
            self.queue.put((self.process_button.config, ({"state": tk.NORMAL},)))
            self.queue.put((self.cancel_button.config, ({"state": tk.DISABLED},)))
            self.queue.put((self.progress.config, ({"value": 100},)))

            logging.info("Processing completed successfully.")

            # Plot waveforms
            self.plot_waveforms()

            # Set processing as complete
            self.processing_complete.set()

        except Exception as e:
            # Handle exceptions and update the GUI accordingly
            self.queue.put((
                self.status_label.config,
                ({"text": f"An error occurred: {e}", "fg": "red"},)
            ))
            self.queue.put((self.process_button.config, ({"state": tk.NORMAL},)))
            self.queue.put((self.cancel_button.config, ({"state": tk.DISABLED},)))
            self.queue.put((self.progress.config, ({"value": 0},)))
            logging.error(f"Error in processing audio: {e}")

    def convert_to_float(self, y):
        # Since librosa.load returns float32 in range [-1.0, 1.0], no conversion is necessary
        return y

    def save_mixed_audio(self, output_dir='output_signals'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.debug(f"Created Output Directory: {output_dir}")

        # Stack channels first
        y_mixed = np.vstack((self.y_input1, self.y_input2)).T

        # Scale based on the maximum absolute value across all channels
        max_val = np.max(np.abs(y_mixed))
        if max_val > 1.0:
            y_mixed_scaled = y_mixed / max_val
            logging.debug("Mixed audio scaled to prevent clipping.")
        else:
            y_mixed_scaled = y_mixed

        # Convert to int16
        y_mixed_scaled_int16 = np.int16(y_mixed_scaled * 32767)

        mixed_output_file = os.path.join(output_dir, 'mixed_signal.wav')
        wavfile.write(mixed_output_file, self.Fs, y_mixed_scaled_int16)
        logging.debug(f"Mixed Signal saved as: {mixed_output_file}")

        # Assign mixed audio data to self.y_mixed_data
        self.y_mixed_data = y_mixed_scaled.copy()

        return mixed_output_file

    def save_outputs(self, output_dir='output_signals'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.debug(f"Created Output Directory: {output_dir}")

        output_files = []
        # Scale to int16 for WAV format
        e1_max = np.max(np.abs(self.e1))
        e2_max = np.max(np.abs(self.e2))
        e1_scaled = self.e1 / e1_max if e1_max != 0 else self.e1
        e2_scaled = self.e2 / e2_max if e2_max != 0 else self.e2

        # Prevent clipping
        e1_scaled = np.clip(e1_scaled, -1.0, 1.0)
        e2_scaled = np.clip(e2_scaled, -1.0, 1.0)

        e1_scaled_int16 = np.int16(e1_scaled * 32767)
        e2_scaled_int16 = np.int16(e2_scaled * 32767)

        output_file1 = os.path.join(output_dir, 'output_signal_1.wav')
        wavfile.write(output_file1, self.Fs, e1_scaled_int16)
        output_files.append(output_file1)
        logging.debug(f"Output Signal 1 saved as: {output_file1}")

        output_file2 = os.path.join(output_dir, 'output_signal_2.wav')
        wavfile.write(output_file2, self.Fs, e2_scaled_int16)
        output_files.append(output_file2)
        logging.debug(f"Output Signal 2 saved as: {output_file2}")

        # Store in-memory data for playback
        self.e1_scaled = e1_scaled.copy()
        self.e2_scaled = e2_scaled.copy()

        return output_files

    def play_audio(self, index):
        if index >= len(self.output_files):
            messagebox.showerror("Playback Error", "Selected audio file does not exist.")
            logging.error("Playback attempted for non-existent audio file.")
            return

        threading.Thread(target=self._play_audio_thread, args=(index,), daemon=True).start()

    def play_mixed_audio(self):
        if not self.processing_complete.is_set():
            messagebox.showerror("Playback Error", "Audio processing is not yet complete.")
            logging.error("Playback attempted before processing completion.")
            return
        if self.y_mixed_data is None:
            messagebox.showerror("Playback Error", "No mixed audio available to play.")
            logging.error("Playback attempted with no mixed audio data.")
            return

        threading.Thread(target=self._play_mixed_audio_thread, daemon=True).start()

    def _play_audio_thread(self, index):
        try:
            if index == 0:
                y = self.e1_scaled
                logging.debug("Playing Output Signal 1")
            elif index == 1:
                y = self.e2_scaled
                logging.debug("Playing Output Signal 2")
            else:
                raise ValueError("Invalid audio index.")

            sd.play(y, self.Fs)
            sd.wait()
            logging.debug(f"Finished playing Output Signal {index+1}")
        except Exception as e:
            self.queue.put((messagebox.showerror, ("Playback Error", f"An error occurred while playing the audio:\n{e}")))
            logging.error(f"Error during playback: {e}")

    def _play_mixed_audio_thread(self):
        try:
            mode = self.playback_mode.get()
            if mode == "Stereo":
                y = self.y_mixed_data
                logging.debug("Playing Mixed Signal in Stereo mode.")
            elif mode == "Mono":
                y = np.mean(self.y_mixed_data, axis=1)
                logging.debug("Playing Mixed Signal in Mono mode.")
            else:
                self.queue.put((messagebox.showerror, ("Playback Mode Error", "Unknown playback mode selected.")))
                logging.error("Unknown playback mode selected.")
                return

            # Ensure y is in float32 format and normalized
            if y.dtype != np.float32:
                y = self.convert_to_float(y)
            max_val = np.max(np.abs(y))
            if max_val > 1.0:
                y = y / max_val

            sd.play(y, self.Fs)
            sd.wait()
            logging.debug(f"Finished playing Mixed Signal in {mode} mode.")
        except Exception as e:
            self.queue.put((messagebox.showerror, ("Playback Error", f"An error occurred while playing the mixed audio:\n{e}")))
            logging.error(f"Error during mixed audio playback: {e}")

    def plot_waveforms(self):
        def _plot():
            try:
                num_channels = 2  # Since input is stereo
                time_axis = np.linspace(0, len(self.y_input1) / self.Fs, num=len(self.y_input1))

                self.figure.clf()  # Clear previous plots

                # Mixed Signals - Channel 1
                ax1 = self.figure.add_subplot(2, 2, 1)
                ax1.plot(time_axis, self.y_input1, color='blue')
                ax1.set_title('Mixed Signal - Channel 1', pad=20)
                ax1.grid(True)

                # Mixed Signals - Channel 2
                ax2 = self.figure.add_subplot(2, 2, 2)
                ax2.plot(time_axis, self.y_input2, color='orange')
                ax2.set_title('Mixed Signal - Channel 2', pad=20)
                ax2.grid(True)

                # Output Signals - Signal 1
                ax3 = self.figure.add_subplot(2, 2, 3)
                ax3.plot(time_axis, self.e1, color='green')
                ax3.set_title('Output Signal 1', pad=20)
                ax3.grid(True)

                # Output Signals - Signal 2
                ax4 = self.figure.add_subplot(2, 2, 4)
                ax4.plot(time_axis, self.e2, color='red')
                ax4.set_title('Output Signal 2', pad=20)
                ax4.grid(True)

                self.figure.tight_layout()
                self.canvas.draw()
                logging.debug("Waveforms plotted successfully.")
            except Exception as e:
                self.queue.put((messagebox.showerror, ("Plotting Error", f"An error occurred while plotting waveforms:\n{e}")))
                logging.error(f"Error during plotting waveforms: {e}")

        threading.Thread(target=_plot, daemon=True).start()


def main():
    root = tk.Tk()
    app = NoiseCancellerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
