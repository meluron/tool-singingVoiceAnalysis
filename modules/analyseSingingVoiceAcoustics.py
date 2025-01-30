"""
author: ankit anand
created on: 29/01/25
"""

from FBC import FBC
from loadAudio import loadAudio
from getPitch import getPitch
from getIntensity import getIntensity
from getSpectrogram import getSpectrogram
from getSBEOnsets import getSBEOnsets

from pathlib import Path

import numpy as np

class analyseSingingVoiceAcoustics(FBC):
	"""
	To put together all the modules, specifically customised as per the needs of IIT-B DAPLAB team working on SRGM project
	
	inps
	----
	audioSignal: ndarray
		audio signal that needs to be analysed
	audioSr: int
		audio sampling rate
	controls: dict[dict]
		controls over various acoustic features computation

	outs
	----
	
	
	"""
	
	def __init__(self, audioFp:str=None, gtTxtFp:str=None, hubertTxtFp:str=None, audioControls:dict=None, pitchControls:dict=None, intensityControls:dict=None, SBEOnsetsControls:dict=None):
		"""set parameters"""
		super().__init__(
			audioFp=Path(audioFp) if audioFp is not None else audioFp,
			gtTxtFp=Path(gtTxtFp) if gtTxtFp is not None else gtTxtFp,
			hubertTxtFp=Path(hubertTxtFp) if hubertTxtFp is not None else hubertTxtFp,
			audioControls=audioControls,
			pitchControls=pitchControls,
			intensityControls=intensityControls,
			SBEOnsetsControls=SBEOnsetsControls,
		) # pass the parameters as kwargs
	
	def test(self):
		"""runs custom tests on function"""
		
		pass
				
	def validate(self):
		"""validate inputs"""
		
		for k, v in self.inps.items(): # validating if input is passed
			if v is None and k not in ["gtTxtFp","hubertTxtFp","audioControls", "pitchControls", "intensityControls", "SBEOnsetsControls"]:
				raise PermissionError(f"inputs ({k}) needs to be passed in the constructor before calling run")
		
			
	def run(self, debugMode=False):
		"""main implementation of the code"""
		
		self.ran = True # flag to check if run command was called
		self.debugMode = debugMode # turn on debug mode
		self.validate() # inputs validation
		
		# INPS EXTRACTION
		audioFp, gtTxtFp, hubertTxtFp, audioControls, pitchControls, intensityControls, SBEOnsetsControls = self.inps.values()
		
		# LOGIC IMPLEMENTATION
		audioSignal, audioSr = loadAudio(audioFp=audioFp, controls=audioControls).run().outs.values()
		pitchValues, pitchTs = getPitch(audioSignal=audioSignal, audioSr=audioSr, controls=pitchControls).run().outs.values()
		intensityValues, intensityTs = getIntensity(audioSignal=audioSignal, audioSr=audioSr, controls=intensityControls).run().outs.values()
		powerSpec, fs, ts = getSpectrogram(audioSignal=audioSignal, audioSr=audioSr).run().outs.values()
		onsets, proms = getSBEOnsets(audioSignal=audioSignal, audioSr=audioSr, controls=SBEOnsetsControls).run().outs.values()
		
		# OUPUTS
		self.outs["audioSignal"] = audioSignal
		self.outs["audioSr"] = audioSr
		self.outs["pitchValues"] = pitchValues
		self.outs["pitchTs"] = pitchTs
		self.outs["intensityValues"] = intensityValues
		self.outs["intensityTs"] = intensityTs
		self.outs["powerSpec"] = powerSpec
		self.outs["fs"] = fs
		self.outs["ts"] = ts
		self.outs["onsets"] = onsets
		self.outs["proms"] = proms
		
		# DEBUGS
		if self.debugMode:
			pass # add all the debug varibles
			
		return self
	
	def plot(self, show=False):
		"""plot to debug"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before plotting")
			
		import matplotlib.pyplot as plt
		
		audioSignal, audioSr, pitchValues, pitchTs, intensityValues, intensityTs, powerSpec, fs, ts, onsets, proms = self.outs.values()
		gtTxtFp, hubertTxtFp = self.inps["gtTxtFp"], self.inps["hubertTxtFp"]
		
		# load onsets from text files to show for comparison
		gtOnsets, gtPhones, hubertOnsets, hubertPhones = np.array([]), np.array([]), np.array([]), np.array([]) # initialising in case not available
		if gtTxtFp is not None:
			try:
				with open(gtTxtFp,  "r") as f:
					content = [line.split("\t") for line in f.read().split("\n")]
				gtOnsets = np.array([float(line[0]) for line in content if line and line[0].strip]) # strip check to avoid empty line in the txt
				gtPhones = np.array([line[2] for line in content if line and line[0].strip]) #TODO: change as per the new format, no end time info
			except Exception as e:
				pass #TODO

		if hubertTxtFp is not None:
			try:
				with open(hubertTxtFp, "r") as f:
					content = [line.split("\t") for line in f.read().split("\n")]
				hubertOnsets = np.array([float(line[0]) for line in content if line and line[0].strip()])
				hubertPhones = np.array([line[1] for line in content if line and line[0].strip()])
			except Exception as e:
				pass #TODO
		
		if self.debugMode == True:
			self.debugs["gtOnsets"] = gtOnsets
			self.debugs["gtPhones"] = gtPhones
			self.debugs["hubertOnsets"] = hubertOnsets
			self.debugs["hubertPhones"] = hubertPhones
			
		fig, ax = plt.subplots(nrows=4, ncols=1, width_ratios=[1], height_ratios=[0.05, 0.1, 0.1, 0.25], figsize=(16, 9), sharex=True)
		
		ax[0].plot(np.arange(audioSignal.shape[0])/audioSr, audioSignal)
		ax[0].set_title("Waveform")
		ax[0].set_ylabel("Amplitude")
		
		ax[1].plot(pitchTs, pitchValues, c="r")
		ax[1].set_title("Pitch contour")
		ax[1].set_ylabel("F0 (hz)")
		
		ax[2].plot(intensityTs, intensityValues, c="k")
		ax[2].set_title("Intensity contour")
		ax[2].set_ylabel("Intensity (dB)")
		
		ax[3].imshow(powerSpec, origin='lower', aspect='auto', cmap='gray_r', extent=[ts[0], ts[-1], fs[0], fs[-1]])
		ax[3].vlines(onsets, np.ones(onsets.shape[0])*1000, 1000+(proms*2000), linewidth=2, color="b", label="SBE-onsets")
		
		# GT
		for gtOnset, gtPhone in zip(gtOnsets, gtPhones):
			ax[3].text(gtOnset, 3500, gtPhone, fontsize=12, color='r', clip_on=True)
		ax[3].vlines(gtOnsets, 4000, 4300, linewidth=2, color="r", label="GT-phone-onsets")
		
		# HUBERT
		for hubertOnset, hubertPhone in zip(hubertOnsets, hubertPhones):
			ax[3].text(hubertOnset, 5200, hubertPhone, fontsize=12, color='k', clip_on=True)
		ax[3].vlines(hubertOnsets, 4700, 5000, linewidth=2, color="k", label="HUBERT-phone-onsets")
		
		ax[3].set_title("Spectrogram with onsets")
		ax[3].set_xlabel("Time (s)")
		ax[3].set_ylabel("Frequency (hz)")
		ax[3].legend(loc="upper left")
		
		for i in range(4):
			ax[i].tick_params(axis="x", labelbottom=True)
			ax[i].grid()
			
		if show:
			plt.tight_layout()
			plt.show()
		else:
			plt.close(fig)
		
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