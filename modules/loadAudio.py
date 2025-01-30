"""
author: ankit anand
created on: 28/01/25
"""

from FBC import FBC

from pathlib import Path

import librosa
import numpy as np

DEFAULT_CONTROLS = {
	"targetSampleRate": 16000 # default to 16000
}

class loadAudio(FBC):
	"""
	Load audio given a path using librosa library
	
	inps
	----
	audioFp: str
		audio filepath
	controls: dict
		audio loading parameter controls
	
	outs
	----
	audioSignal: ndarray
		audio signal
	audioSr: int
		audio sampling rate (#samples/sec)
	
	"""
	
	def __init__(self, audioFp:str=None, controls:dict=None):
		"""set parameters"""
		super().__init__(
			audioFp = Path(audioFp) if audioFp else None,
			controls=controls if controls is not None else DEFAULT_CONTROLS
		) # pass the parameters as kwargs
	
	
	def test(self):
		"""runs custom tests on function"""
		
		pass
				
	def validate(self):
		"""validate inputs"""
		
		for k, v in self.inps.items(): # validating if input is passed
			if v is None and k not in ["controls"]:
				raise PermissionError(f"inputs ({k}) need to be passed in the constructor before calling run")
		
			
	def run(self, debugMode=False):
		"""main implementation of the code"""
		
		self.ran = True # flag to check if run command was called
		self.debugMode = debugMode # turn on debug mode
		self.validate() # inputs validation
		
		# INPS EXTRACTION
		audioFp, controls = self.inps.values()
		
		# LOGIC IMPLEMENTATION
		audioSignal, audioSr = librosa.load(path=audioFp, sr=controls["targetSampleRate"]) # loading audio at 16Khz to keep it standard
		
		# OUPUTS
		self.outs["audioSignal"] = audioSignal
		self.outs["audioSr"] = audioSr
		
		# DEBUGS
		if self.debugMode:
			self.debugs = {} # add all the debug varibles
			
		return self
	
	def plot(self, show=False):
		"""plot to debug"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before plotting")
			
		import matplotlib.pyplot as plt
		
		audioSignal, audioSr = self.outs.values()
		ts = np.arange(audioSignal.shape[0])/audioSr
		
		fig = plt.figure(figsize=(13, 4))
		plt.plot(ts, audioSignal)
		plt.title("Waveform")
		plt.xlabel("Time (s)")
		plt.ylabel("Amplitude")
		plt.xticks(np.arange(ts[0], ts[-1], 5), rotation=90)
		plt.xlim(ts[0], ts[-1]) # to remove extra space from start so that all plots align
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

