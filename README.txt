README

	This program contains an implementation of a motion tracking system relying on the KLT-algorithm. The 
	implementation is written using the Anaconda distribution of Python 2.7, and the program relies on the 
	library scikit-image for image manipulation and gradient calculation, and matplotlib for visualization.
	
	Video output is handled through the ffmpeg codec, available at https://www.ffmpeg.org/.
	
INSTRUCTIONS	

	The program should be accessed through the main file MotionTracker.py, either from the command line with "python
	MotionTracker.py", or imported as a module.
	
	If run in the command line, a small text interface is provided. Here, the user is allowed to alter the functionality
	of the program with regards to patch size, eigenvalue limit, update rate, and weight function. Moreover, the user
	can specify a directory containing a sequence of images to be used as input.

	If imported as a module, the functionality should be accessed through the function 'track'.
	
	The library file VideoLoader.py contains functionality for video I/O.

CONTACT
	
	If there is any problem, please contact me (Michael Schlichtkrull) through my student mail qwt774@alumni.ku.dk.
