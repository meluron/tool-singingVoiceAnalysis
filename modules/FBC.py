"""
author: ankit anand
created on: 23/01/25
"""

from pathlib import Path
import logging

class FBC:
	"""Functional Base Class (by Ankit Anand)"""
	
	def __init__(self, *args, **kwargs):
		"""set parameters"""
		
		self.inps = kwargs if kwargs else {}
		
		self.outs = {
			
		} # this will be returned on call
		
		self.debugs = {
			"logLevel": "INFO",
			"logFp": None #Path(__file__).resolve().parents[0]/"log.txt"
		} # variables to be used for plotting/debugging
		
		self.debugMode = False
		self.ran = False
		
		self.logger = self._getLogger(self.debugs["logLevel"], self.debugs["logFp"])

	def info(self):
		"""returns methods and params available"""
		
		print(f"\nINFO about FC <{self.__class__.__name__}>")
		print("-"*45)
		print(self.__doc__)
		print("-"*45)
		print(f"\n:: METHODS")
		for method in self._methods():
			print(f"  - {method}")
			
		print(f"\n:: INPS")
		for k, v in self.inps.items():
			print(f"  - {k} [{type(v)}]")
			
		print(f"\n:: DEBUGS")
		for k, v in self.debugs.items():
			print(f"  - {k} [{type(v)}]")
		
		print(f"\n:: OUTS")
		for k, v in self.outs.items():
			print(f"  - {k} [{type(v)}]")
	
	def _methods(self):
		"""returns methods as a list"""
		
		return [
			method for method in self.__dir__()
			if callable(getattr(self, method))  # ensure it's a callable (method)
			and not method.startswith("_")  # excludes special methods (e.g., __init__)
			and method != "methods"  # exclude the methods() function itself
			]
				
	def _getLogger(self, logLevel:str, logFp:str|Path):
		"""log info about function calls, call it from the run function, wherever needed"""
		
		numericLevel = logging.getLevelName(logLevel.upper())
		if isinstance(numericLevel, int):
			logLevel = numericLevel
		else:
			logLevel = logging.INFO
			
		logging.basicConfig(
			level=logLevel,
			format=(
				"%(asctime)s - File: %(filename)s - Function: %(funcName)s "
				"- Line: %(lineno)d - Level: %(levelname)s - Message: %(message)s"
			),
			datefmt="%d %b %Y, %H:%M:%S",
		)
	
		# If logFp is not None, log to the specified file
		if logFp:
			file_handler = logging.FileHandler(logFp)
			file_handler.setFormatter(logging.Formatter(
				"%(asctime)s - File: %(filename)s - Function: %(funcName)s "
				"- Line: %(lineno)d - Level: %(levelname)s - Message: %(message)s",
				datefmt="%d %b %Y, %H:%M:%S"
			))
			logging.getLogger().addHandler(file_handler)
			
		return logging.getLogger()

	

		
	