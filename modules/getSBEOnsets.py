"""
author: ankit anand
created on: 28/01/25
"""

from FBC import FBC

import numpy as np
import librosa, scipy


DEFAULT_CONTROLS = {
	"promsThreshold": None # filter onsets with prominence above this threshold
}

def adaptationFilter(nov, novSr, filterLength):
	"""
	To find smoothened derivative using two gaussians
	
	inps
	----
	nov: ndarray	
		novelty function whose derivative needs to be computed
	novSr: int
		novelty function sampling rate (to be able to adjust parameters)
	filterLength: int
		length of the filter (-filterLength to filterLength+1)

	outs
	----
	der: ndarray
		smoothened derivate of the novelty function
	kernel: ndarray
		kernel used to find out the derivative (only for developers to verify)

	"""
	tau1 = int(0.015 * novSr)
	d1 = int(0.020 * novSr)
	tau2 = int(0.015 * novSr)
	d2 = int(0.020 * novSr)
	
	kernel = np.zeros(2 * filterLength)
	t = np.arange(-filterLength, +filterLength+1) 
	kernel = (1/(tau1*np.sqrt(2*np.pi))) * np.exp(-(t-d1)**2/(2*tau1**2)) - (1/(tau2*np.sqrt(2*np.pi))) * np.exp(-(t+d2)**2/(2*tau2**2))
	kernel =  np.exp(-(t-d1)**2/(2*tau1**2)) - np.exp(-(t+d2)**2/(2*tau2**2))
	
	# Apply the biphasic filter using convolution
	der = scipy.signal.convolve(nov, np.array(list(reversed(kernel))), mode='same') # reversed to perform convolution in the right orientation
	
	der[der < 0] = 0 # as we are only intereseted in peaks
	
	return der, kernel

class getSBEOnsets(FBC):
	"""
	To find the locations of onsets of vowels using sub-band energy
	
	inps
	----
	audioSignal: ndarray
		audio signal
	audioSr: int
		audio sampling rate
	outs
	----
	onsets: ndarray
		onsets position in sec
	proms: ndarray
		prominence value for each of the onsets
	
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
			if v is None:
				raise PermissionError(f"inputs ({k}) need to be initialized before calling run")
			
	def run(self, debugMode=False):
		"""main implementation of the code"""
		
		self.ran = True # flag to check if run command was called
		self.debugMode = debugMode # turn on debug mode
		self.validate() # inputs validation
		
		# INPS EXTRACTION
		audioSignal, audioSr, controls = self.inps.values()
		
		# LOGIC IMPLEMENTATION
		N, H = 512, 160 # 32 ms, 10 ms
		_gamma = 10
		
		X = librosa.stft(audioSignal, n_fft=N, hop_length=H, win_length=N, window='hamming')
		powerSpec = np.log(1 + _gamma * np.abs(X)**2) # this is used only for plots
		
		ts = np.arange(powerSpec.shape[1]) * (H/audioSr)
		fs = np.arange(powerSpec.shape[0]) * (audioSr/N)
		
		# Choose the frequency bands to combine
		franges = np.array([[300, 500],[640, 2800]])
		
		E = np.zeros(X.shape[1])
		onsets = []
		proms = []
		
		for i in range(len(franges)):
			low = franges[i,0]
			high = franges[i,1]
			selectedFsBins = np.where((fs > low) & (fs<high))[0]
			
			E += np.sum((np.abs(X[selectedFsBins,:])**2), axis=0)
			
			# E = E/(selectedFsBins.shape[0] * N) # some sort of normalisation if required
			E = np.log(1+E)
			
		# adaptation filter
		nov, kernel = adaptationFilter(E, int(audioSr/H), 40)
		nov[nov < 0] = 0
		nov = nov/np.max(nov)
		onsets, props = scipy.signal.find_peaks(nov, prominence=0.1, distance=int(0.1 * int(audioSr/H))) # onsets (samples) and properties
		
		onsets = ts[onsets]
		proms = props["prominences"]
		
		if controls["promsThreshold"] is not None:
			onsets = onsets[proms>controls["promsThreshold"]]
			proms = proms[proms>controls["promsThreshold"]]
		
		# OUPUTS
		self.outs["onsets"] = onsets
		self.outs["proms"] = proms
		
		# DEBUGS
		if self.debugMode:
			self.debugs["ts"], self.debugs["fs"], self.debugs["powerSpec"] = ts, fs, powerSpec
			
		return self
	
	def plot(self, show=False):
		"""plot to debug"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before plotting")
		if not self.debugMode:
			raise PermissionError("run method needs to be called with debugMode before plotting")
		
		import matplotlib.pyplot as plt
		
		onsets, proms = self.outs.values()
		_, _, ts, fs, powerSpec = self.debugs.values()
		
		fig = plt.figure(figsize=(14, 4))
		plt.imshow(powerSpec, origin='lower', aspect='auto', cmap='gray_r', extent=[ts[0], ts[-1], fs[0], fs[-1]])
		plt.vlines(onsets, np.ones(onsets.shape[0])*1000, 1000+(proms*1000), linewidth=4, label="SBE-onsets")
		plt.title("Spectrogram with onsets")
		plt.xlabel("Time (s)")
		plt.ylabel("Frequency (hz)")
		plt.xticks(np.arange(ts[0], ts[-1], 5), rotation=90)
		plt.legend()
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