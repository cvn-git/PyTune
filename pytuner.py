import numpy as np
import scipy.signal
import sys
import time
import os
import soundfile
from functools import partial

import pygame
from pygame._sdl2 import get_num_audio_devices, get_audio_device_name, AudioDevice, AUDIO_F32, AUDIO_ALLOW_FORMAT_CHANGE

from PySide2.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy, QGridLayout, QToolBar
from PySide2.QtWidgets import QWidget, QPushButton, QMessageBox, QLineEdit, QComboBox, QLabel, QGroupBox, QFileDialog
from PySide2.QtWidgets import QStatusBar, QCheckBox
from PySide2.QtSvg import QSvgWidget
from PySide2.QtCore import Qt, QTimer, QObject, QCoreApplication, QSettings
from PySide2 import QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class AudioRecord:
    def __init__(self, sample_rate, duration, chunk_size=512):
        self._chunk_size = chunk_size
        self._num_chunks = int(np.ceil(duration * sample_rate / chunk_size))
        self._audio_data = np.zeros(self._num_chunks * chunk_size)
        self._chunk_idx = 0

    def completed(self):
        return self._chunk_idx >= self._num_chunks

    def audio_data(self):
        return self._audio_data

    def add_chunk(self, _, chunk_mem):
        if not self.completed():
            assert len(chunk_mem) == self._chunk_size * 4, 'Invalid chunk size'
            self._audio_data[self._chunk_size * self._chunk_idx: self._chunk_size * (self._chunk_idx + 1)] =\
                np.frombuffer(chunk_mem, dtype='float32')
            self._chunk_idx += 1


class PyTune(QMainWindow):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)

        self.busy_message = None
        self.busy_cursor = None
        self.signal = None
        self.f_spec = None
        self.t_spec = None
        self.spec = None
        self.start_idx = None
        self.end_idx = None
        self.note_str = None
        self.note_freq = None
        self.note_idx = None

        pygame.init()
        num = get_num_audio_devices(True)
        device_names = [(get_audio_device_name(k, True)).decode() for k in range(num)]

        # Main widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QGridLayout(main_widget)

        # Status bar
        toolbar = QToolBar()
        self.addToolBar(Qt.BottomToolBarArea, toolbar)
        self.status_bar = QStatusBar()
        toolbar.addWidget(self.status_bar)

        # Control widget
        control_widget = QWidget()
        layout_control = QGridLayout(control_widget)
        main_layout.addWidget(control_widget, 0, 0, 1, 1)

        # Record control
        group_record = QGroupBox('Recording')
        layout_record = QGridLayout()
        group_record.setLayout(layout_record)
        layout_control.addWidget(group_record, 0, 0, 2, 1)

        layout_record.addWidget(QLabel('Device:'), 0, 0)
        self.combo_device = QComboBox()
        self.combo_device.addItems(device_names)
        layout_record.addWidget(self.combo_device, 0, 1)

        layout_record.addWidget(QLabel('Sample rate:'), 1, 0)
        self.textbox_sample_rate = QLineEdit('48000')
        layout_record.addWidget(self.textbox_sample_rate, 1, 1)

        layout_record.addWidget(QLabel('Duration:'), 2, 0)
        self.textbox_duration = QLineEdit('5.0')
        layout_record.addWidget(self.textbox_duration, 2, 1)

        button_record = QPushButton('Record')
        button_record.clicked.connect(self.button_record_clicked)
        layout_record.addWidget(button_record, 3, 1)

        # Note control
        group_note = QGroupBox('Note')
        layout_note = QVBoxLayout(group_note)
        layout_control.addWidget(group_note, 0, 1, 1, 1)

        widget_note1 = QWidget()
        layout_note1 = QHBoxLayout(widget_note1)
        layout_note.addWidget(widget_note1)

        layout_note1.addWidget(QLabel('RefHz:'))

        self.textbox_ref_freq = QLineEdit('440.0')
        self.textbox_ref_freq.returnPressed.connect(self.ref_freq_changed)
        layout_note1.addWidget(self.textbox_ref_freq)

        widget_padding = QLabel('')
        widget_padding.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout_note1.addWidget(widget_padding)

        self.label_note = QLabel('N/A')
        layout_note1.addWidget(self.label_note)

        self.checkbox_auto = QCheckBox('Auto')
        self.checkbox_auto.setCheckState(Qt.Checked)
        layout_note1.addWidget(self.checkbox_auto)

        widget_note2 = QWidget()
        layout_note2 = QHBoxLayout(widget_note2)
        layout_note.addWidget(widget_note2)

        button_down_more = QPushButton('<<')
        button_down_more.clicked.connect(partial(self.adjust_note, -12))
        layout_note2.addWidget(button_down_more)

        button_down = QPushButton('<')
        button_down.clicked.connect(partial(self.adjust_note, -1))
        layout_note2.addWidget(button_down)

        button_up = QPushButton('>')
        button_up.clicked.connect(partial(self.adjust_note, 1))
        layout_note2.addWidget(button_up)

        button_up_more = QPushButton('>>')
        button_up_more.clicked.connect(partial(self.adjust_note, 12))
        layout_note2.addWidget(button_up_more)

        # File control
        group_file = QGroupBox('File')
        layout_file = QHBoxLayout(group_file)
        layout_control.addWidget(group_file, 1, 1, 1, 1)

        button_save = QPushButton('Save')
        button_save.clicked.connect(self.button_save_clicked)
        layout_file.addWidget(button_save)

        button_load = QPushButton('Load')
        button_load.clicked.connect(self.button_load_clicked)
        layout_file.addWidget(button_load)

        # Waveform display
        group_waveform = QGroupBox('')
        layout_waveform = QVBoxLayout(group_waveform)
        main_layout.addWidget(group_waveform, 1, 0, 1, 1)

        figure_waveform = Figure(figsize=(10, 50), dpi=100)
        axes = figure_waveform.subplots(1, 2, sharey='row', gridspec_kw=dict(width_ratios=[2, 3]))
        self.axis_waveform = axes[0]
        self.axis_spectrum = axes[1]
        figure_waveform.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0)
        self.canvas_waveform = FigureCanvas(figure_waveform)
        self.canvas_waveform.mpl_connect('button_press_event', self.figure_clicked)
        self.canvas_waveform.draw()
        layout_waveform.addWidget(self.canvas_waveform)
        layout_waveform.addWidget(NavigationToolbar(self.canvas_waveform, self))

        # Harmonics display
        group_harmonics = QGroupBox('Harmonics')
        layout_harmonics = QVBoxLayout(group_harmonics)
        main_layout.addWidget(group_harmonics, 0, 1, 2, 1)

        figure_harmonics = Figure(figsize=(50, 50), dpi=100)
        axes = figure_harmonics.subplots(2, 2)
        self.axis_harmonics = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
        figure_harmonics.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        self.canvas_harmonics = FigureCanvas(figure_harmonics)
        self.canvas_harmonics.draw()
        layout_harmonics.addWidget(self.canvas_harmonics)
        layout_harmonics.addWidget(NavigationToolbar(self.canvas_harmonics, self))

        # Done
        self.setWindowIcon(QtGui.QIcon(r'./rc/PikPng.com_piano-png_4272313.png'))
        self.compute_notes(440.0)
        self.not_busy()

    def button_record_clicked(self):
        self.clear_waveform()
        self.clear_harmonics()

        fs = int(self.textbox_sample_rate.text())
        duration = float(self.textbox_duration.text())
        chunk_size = 512

        record = AudioRecord(sample_rate=fs, duration=duration, chunk_size=chunk_size)
        audio = AudioDevice(devicename=self.combo_device.currentText().encode(), iscapture=True,
                            frequency=fs, audioformat=AUDIO_F32, numchannels=1, chunksize=chunk_size,
                            allowed_changes=AUDIO_ALLOW_FORMAT_CHANGE, callback=record.add_chunk)

        self.busy('Recording')
        audio.pause(False)
        time.sleep(duration + 0.5)
        audio.close()
        assert record.completed(), 'Recording not completed'

        self.new_signal(record.audio_data())
        self.not_busy()

    def button_save_clicked(self):
        if self.signal is None:
            return

        settings = QSettings()
        key = 'sound_path'

        path_name, *_ = QFileDialog.getSaveFileName(self, 'Select WAV file to save', dir=settings.value(key, ''),
                                                    filter='WAV files (*.wav)')
        if len(path_name) == 0:
            return

        settings.setValue(key, os.path.split(path_name)[0])

        soundfile.write(path_name, self.signal, int(self.textbox_sample_rate.text()))
        self.info(f'Sound saved to {path_name} successfully')

    def button_load_clicked(self):
        settings = QSettings()
        key = 'sound_path'

        path_name, *_ = QFileDialog.getOpenFileName(self, 'Select sound file to open', dir=settings.value(key, ''),
                                                    filter='Sound files (*.wav;*.aiff)')
        if len(path_name) == 0:
            return

        settings.setValue(key, os.path.split(path_name)[0])

        self.clear_waveform()
        self.clear_harmonics()

        signal, fs = soundfile.read(path_name, dtype='float32')
        if len(signal.shape) > 1:
            signal = signal[:, 0]
        self.textbox_sample_rate.setText(str(fs))

        max_len = int(round(fs * 10))
        if len(signal) > max_len:
            signal = signal[0:max_len]

        self.new_signal(signal)

    def figure_clicked(self, event):
        if event.inaxes != self.axis_waveform:
            return
        if (event.button != 1) and (event.button != 3):
            return

        self.set_boundary(event.ydata, event.button == 1)

    def set_boundary(self, t, is_start):
        if self.signal is None:
            return

        fs = int(self.textbox_sample_rate.text())
        idx = int(round(t * fs))
        if is_start:
            if idx >= self.end_idx:
                return
            self.start_idx = idx
        else:
            if idx <= self.start_idx:
                return
            self.end_idx = idx

        self.clear_waveform()
        self.clear_harmonics()
        self.plot_waveform()
        self.plot_harmonics()

    def new_signal(self, signal):
        signal = signal - np.mean(signal)

        fs = int(self.textbox_sample_rate.text())
        self.signal = signal

        # Raw spectrogram
        n_dft = 4096
        self.f_spec, self.t_spec, self.spec = scipy.signal.spectrogram(signal, fs=fs, nperseg=n_dft, mode='psd')

        # Find peak frequency
        peak_freq = self.f_spec[np.unravel_index(np.argmax(self.spec), self.spec.shape)[0]]
        self.do_extraction(peak_freq)
        print(peak_freq)
        if self.checkbox_auto.checkState() == Qt.Checked:
            self.set_note(np.argmin(np.abs(peak_freq - self.note_freq)))

        # Plotting
        self.plot_waveform()
        self.plot_harmonics()

    def plot_waveform(self):
        fs = int(self.textbox_sample_rate.text())
        time_span = np.arange(len(self.signal), dtype=float) / fs
        ax = self.axis_waveform
        ax.plot(self.signal, time_span, 'g')
        x_lim = ax.get_xlim()
        ax.plot(x_lim, time_span[self.start_idx] * np.ones(2), 'b')
        ax.plot(x_lim, time_span[self.end_idx] * np.ones(2), 'r')
        ax.set_ylabel('Time [s]')
        ax.grid(True)

        x, y = np.meshgrid(self.f_spec, self.t_spec)
        im = 10 * np.log10(self.spec.T)
        min_val = np.amax(im) - 40.0
        im[im < min_val] = min_val
        ax = self.axis_spectrum
        ax.pcolor(x, y, im, shading='auto')
        ax.set_xlabel('Frequency [Hz]')

        ax.set_ylim((self.t_spec[0], self.t_spec[-1]))

        self.canvas_waveform.draw()

    def ref_freq_changed(self):
        self.compute_notes(float(self.textbox_ref_freq.text()))
        self.clear_harmonics()
        self.plot_harmonics()

    def set_note(self, note_idx):
        self.note_idx = note_idx
        self.label_note.setText(self.note_str[self.note_idx])

    def adjust_note(self, offset):
        note_idx = self.note_idx + offset
        if (note_idx >= 0) and (note_idx < len(self.note_freq)):
            self.set_note(note_idx)
            self.clear_harmonics()
            self.plot_harmonics()

    def clear_waveform(self):
        self.axis_waveform.clear()
        self.axis_spectrum.clear()
        self.canvas_waveform.draw()

    def clear_harmonics(self):
        for ax in self.axis_harmonics:
            ax.clear()
        self.canvas_harmonics.draw()

    def plot_harmonics(self):
        fs = int(self.textbox_sample_rate.text())
        for harm_idx in range(4):
            cen_freq = self.note_freq[self.note_idx] * (harm_idx + 1)
            if cen_freq > self.note_freq[-1]:
                break

            signal = self.signal[self.start_idx:self.end_idx]
            mod_sig = signal * np.exp((-2j * np.pi * cen_freq / fs) * np.arange(len(signal), dtype=float))
            ratio = 2 ** (1 / 12)
            sos = scipy.signal.butter(4, cen_freq * (ratio - 1) * 2, output='sos', fs=fs)
            mod_sig = scipy.signal.sosfilt(sos, mod_sig)

            bin_width = cen_freq * (ratio - 1) / 50
            n_dft = int(round(2 ** np.ceil(np.log2(fs / bin_width))))
            if n_dft > len(mod_sig):
                # print('Zero padding')
                mod_sig = np.hstack((mod_sig, np.zeros(n_dft - len(mod_sig))))
            f_span, _, spec = scipy.signal.spectrogram(mod_sig, fs=fs, nperseg=n_dft, detrend=False, mode='psd',
                                                       return_onesided=False)
            spec = np.mean(spec, axis=1)

            f1 = cen_freq / ratio
            f2 = cen_freq * ratio
            idx1 = np.argwhere(f_span < ((f1 - cen_freq) * 1.1)).flatten()[-1]
            idx2 = np.argwhere(f_span > ((f2 - cen_freq) * 1.1)).flatten()[0]
            spec = np.hstack((spec[idx1:], spec[0:idx2]))
            f_span = np.hstack((f_span[idx1:], f_span[0:idx2])) + cen_freq

            ax = self.axis_harmonics[harm_idx]
            ax.plot(f_span, 10 * np.log10(spec))
            y_lim = ax.get_ylim()
            ax.plot(f1 * np.ones(2), y_lim, 'k:')
            ax.plot(f2 * np.ones(2), y_lim, 'k:')
            ax.plot(cen_freq * np.ones(2), y_lim, 'k')
            ax.grid(True)
            if harm_idx == 0:
                ax.set_title(self.note_str[self.note_idx])
            else:
                ax.set_title(f'#{harm_idx + 1}')

        self.canvas_harmonics.draw()

    def do_extraction(self, peak_freq):
        fs = int(self.textbox_sample_rate.text())
        t_span = np.arange(len(self.signal), dtype=float) / fs
        envelope = self.signal * np.exp((-2j * np.pi * peak_freq) * t_span)
        sos = scipy.signal.butter(4, fs / 1000, output='sos', fs=fs)
        envelope = 20 * np.log10(np.abs(scipy.signal.sosfilt(sos, envelope)))

        max_idx = np.argmax(envelope)
        self.start_idx = max(max_idx - int(round(fs * 0.1)), 0)
        self.end_idx = min(np.argwhere(envelope > (envelope[max_idx] - 50)).flatten()[-1], max_idx + int(round(fs * 3)))

    def compute_notes(self, ref_freq):
        bases = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        self.note_str = [bases[(k - 3) % 12] + str(((k - 3) // 12) + 1) for k in range(88)]
        ratio = 2 ** (1 / 12)
        self.note_freq = ref_freq * (ratio ** (np.arange(88, dtype=float) - 48))

    def info(self, msg):
        QMessageBox.information(self, 'Info', msg)

    def error(self, msg):
        QMessageBox.critical(self, 'Error', msg)

    def busy(self, message='Busy'):
        self.set_busy(message, Qt.WaitCursor)

    def not_busy(self):
        self.set_busy('Ready', Qt.ArrowCursor)

    def get_busy(self):
        return self.busy_message, self.busy_cursor

    def set_busy(self, message, cursor):
        self.busy_message = message
        self.busy_cursor = cursor
        self.status_bar.showMessage(message)
        self.setCursor(cursor)
        QCoreApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # For QSettings
    QCoreApplication.setOrganizationName('CVN')
    QCoreApplication.setOrganizationDomain('github.com')
    QCoreApplication.setApplicationName('PyTune')

    my_app = PyTune()
    my_app.setWindowTitle('PyTune')
    my_app.show()
    # my_app.resize(1600, 900)
    sys.exit(app.exec_())
