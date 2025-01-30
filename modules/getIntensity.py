"""
author: ankit anand
created on: 28/01/25
"""

from FBC import FBC

import numpy as np
import parselmouth
import scipy

DEFAULT_CONTROLS = {
	"minimum_pitch": 75.0,  # min pitch (in Hz) used for intensity calculation; helps focus on voiced parts
	"time_step": 0.01,      # time step between consecutive intensity measurements (in seconds)
	"subtract_mean": False  # Whether to subtract the mean intensity; useful for normalization
}

class getIntensity(FBC):
	"""
	To compute the intensity contour of an audio signal using praat
	
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
	intensityValues: ndarray
		pitch values
	intensityTs: ndarray
		time in sec for each intensity frame
	
	"""
	
	def __init__(self, audioSignal:np.ndarray=None, audioSr:int=None, controls:dict=None):
		"""set parameters"""
		super().__init__(
			audioSignal=audioSignal,
			audioSr=audioSr,
			controls=controls if controls is not None else DEFAULT_CONTROLS
		) # pass the parameters as kwargs
	
	
	def test(self):
		"""runs custom tests on function"""
		
		pass
				
	def validate(self):
		"""validate inputs"""
		
		for k, v in self.inps.items(): # validating if input is passed
			if v is None and k not in ["controls"]: # exclude optional params
				raise PermissionError(f"inputs ({k}) need to be initialized before calling run")
		
			
	def run(self, debugMode=False):
		"""main implementation of the code"""
		
		self.ran = True # flag to check if run command was called
		self.debugMode = debugMode # turn on debug mode
		self.validate() # inputs validation
		
		# INPS EXTRACTION
		audioSignal, audioSr, controls = self.inps.values()
		
		# LOGIC IMPLEMENTATION
		snd = parselmouth.Sound(values=audioSignal, sampling_frequency=audioSr)
		intensity = snd.to_intensity(
			minimum_pitch=controls["minimum_pitch"], 
			time_step=controls["time_step"], 
			subtract_mean=controls["subtract_mean"]
		) # below min pitch, it will consider it as silence region
		
		intensityValues = intensity.values[0,:]
		intensityTs = np.arange(intensityValues.shape[0]) * 0.01 # but note that there is a lag of around 0.04s in what timestamps parselmouth gives and what is computed here
		
		# OUPUTS
		self.outs["intensityValues"] = intensityValues
		self.outs["intensityTs"] = intensityTs
		
		# DEBUGS
		if self.debugMode:
			self.debugs = {} # add all the debug varibles
			
		return self
	
	def plot(self, show=False):
		"""plot to debug"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before plotting")
			
		import matplotlib.pyplot as plt
		
		intensityValues, intensityTs = self.outs.values()
		
		fig = plt.figure(figsize=(14, 4))
		plt.plot(intensityTs, intensityValues)
		plt.title("Intensity contour")
		plt.xlabel("Time (s)")
		plt.ylabel("Intensity (dB)")
		plt.xticks(np.arange(intensityTs[0], intensityTs[-1], 5), rotation=90)
		plt.xlim(intensityTs[0], intensityTs[-1]) # to remove extra space from start so that all plots align
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