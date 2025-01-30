"""
author: ankit anand
created on: 28/01/25
"""

from FBC import FBC

import numpy as np
import librosa

class getSpectrogram(FBC):
	"""
	To compute spectrogram of an audio signal using librosa
	
	inps
	----
	audioSignal: ndarray
		audio signal of vocals
	audioSr: int
		audio sample rate
	params: dict
		controllable parameters
	
	outs
	----
	powerSpec: ndarray
		spectrogram matrix [F X T]
	fs: ndarray
		frequency index to frequency in hz map
	ts: ndarray
		frame to time in sec
	
	"""
	
	def __init__(self, audioSignal:np.ndarray=None, audioSr:int=None):
		"""set parameters"""
		super().__init__(
			audioSignal=audioSignal,
			audioSr=audioSr
		) # pass the parameters as kwargs
	
	
	def test(self):
		"""runs custom tests on function"""
		
		pass
				
	def validate(self):
		"""validate inputs"""
		
		for k, v in self.inps.items(): # validating if input is passed
			if v is None:
				raise PermissionError(f"inputs ({k}) need to be initialized before calling run")
		
			
	def run(self, debugMode=False):
		"""main implementation of the code"""
		
		self.ran = True # flag to check if run command was called
		self.debugMode = debugMode # turn on debug mode
		self.validate() # inputs validation
		
		# INPS EXTRACTION
		audioSignal, audioSr = self.inps.values()
		
		# LOGIC IMPLEMENTATION
		N, H = 512, 160 # 32 ms, 10ms
		_gamma = 10 # log compression factor decided the contrast of the spectrogram
		
		X = librosa.stft(audioSignal, n_fft=N, hop_length=H, win_length=N, window='hamming', center=True) # computing STFT
		powerSpec = np.log(1 + _gamma * np.abs(X)**2) # log compression to enhance contrast
		
		ts = np.arange(powerSpec.shape[1]) * (H/audioSr)
		fs = np.arange(powerSpec.shape[0]) * (audioSr/N)
		
		# OUPUTS
		self.outs["powerSpec"] = powerSpec
		self.outs["fs"] = fs
		self.outs["ts"] = ts
		
		# DEBUGS
		if self.debugMode:
			self.debugs = {} # add all the debug varibles
			
			
		return self
	
	def plot(self, show=False):
		"""plot to debug"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before plotting")
			
		import matplotlib.pyplot as plt
		
		powerSpec, fs, ts = self.outs.values()
		
		fig = plt.figure(figsize=(14, 4))
		plt.imshow(powerSpec, origin='lower', aspect='auto', cmap='gray_r', extent=[ts[0], ts[-1], fs[0], fs[-1]])
		plt.title("Spectrogram")
		plt.xlabel("Time (s)")
		plt.ylabel("Frequency (hz)")
		plt.xticks(np.arange(ts[0], ts[-1], 5), rotation=90)
		plt.grid()
		
		if show:
			plt.tight_layout()
			plt.show()
		
		return fig
		
	def save(self, outputPath=None):
		"""save debug/output files"""
		
		from pathlib import Path
		
		if not self.ran:
			raise PermissionError("run method needs to be called before saving output")
		if outputPath is None:
			raise ValueError("outputPath can't be None")
		if isinstance(outputPath, str):
			outputPath = Path(outputPath)
		if not outputPath.exists():
			outputPath.mkdir(exist_ok=True, parents=True) 
			
		pass