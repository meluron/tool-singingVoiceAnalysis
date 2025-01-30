"""
author: ankit anand
created on: 28/01/25
"""

from FBC import FBC

import numpy as np
import parselmouth
import scipy

DEFAULT_CONTROLS = {
	"time_step": 0.01,  # time step between consecutive pitch measurements (in seconds)
	"pitch_floor": 75.0,  # minimum pitch value to detect (in Hz), useful for filtering out low-frequency noise
	"max_number_of_candidates": 15,  # max number of pitch candidates per frame to evaluate
	"very_accurate": False,  # increases accuracy at the cost of performance
	"silence_threshold": 0.01,  # energy threshold to distinguish silence from voiced parts
	"voicing_threshold": 0.35,  # threshold for deciding whether a frame is voiced
	"octave_cost": 0.05,  # cost for selecting a pitch candidate an octave apart from the previous one
	"octave_jump_cost": 0.35,  # penalty for sudden jumps between octaves
	"voiced_unvoiced_cost": 0.4,  # cost for transitioning between voiced and unvoiced frames
	"pitch_ceiling": 600  # maximum pitch to detect (in hz)
}

class getPitch(FBC):
	"""
	To get the pitch time series of a vocals audio file
	
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
	pitchValues: ndarray
		pitch values
	pitchTs: ndarray
		time in sec for each pitch frame
	
	"""
	
	def __init__(self, audioSignal:np.ndarray=None, audioSr:int=None, controls:int=None):
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
			if v is None and k not in ["controls"]:
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
		pitch = snd.to_pitch_ac(
			time_step=controls["time_step"],
			pitch_floor=controls["pitch_floor"],
			max_number_of_candidates=controls["max_number_of_candidates"],
			very_accurate=controls["very_accurate"],
			silence_threshold=controls["silence_threshold"],
			voicing_threshold=controls["voicing_threshold"],
			octave_cost=controls["octave_cost"],
			octave_jump_cost=controls["octave_jump_cost"],
			voiced_unvoiced_cost=controls["voicing_threshold"],
			pitch_ceiling=controls["pitch_ceiling"]
		) # creates a pitch object
		pitchValues = pitch.selected_array['frequency'] # get pitch values in hz
		pitchTs = np.arange(pitchValues.shape[0]) * (0.01) # timestamp
		# median filtering to removing unwanted spikes in the pitch contour
		pitchValues = scipy.signal.medfilt(pitchValues, kernel_size=15) # 0.1 sec, use if needed
	
		# OUPUTS
		self.outs["pitchValues"] = pitchValues
		self.outs["pitchTs"] = pitchTs
		
		# DEBUGS
		if self.debugMode:
			self.debugs = {} # add all the debug varibles
			
		return self
	
	def plot(self, show=False):
		"""plot to debug"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before plotting")
			
		import matplotlib.pyplot as plt
		
		pitchValues, pitchTs = self.outs.values()
		
		fig = plt.figure(figsize=(14, 4))
		plt.plot(pitchTs, pitchValues)
		plt.title("Pitch contour")
		plt.xlabel("Time (s)")
		plt.ylabel("Pitch (hz)")
		plt.xticks(np.arange(pitchTs[0], pitchTs[-1], 5), rotation=90)
		plt.xlim(pitchTs[0], pitchTs[-1]) # to remove extra space from start so that all plots align
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