# ml-chess-hikaru

Machine learning engine for playing chess built from GM Hikaru's blitz games scraped from chess.com
 - `PLAYCHESS.ipynb`: to play chess
 - `TRAIN-ENGINE/.ipynb`: to generate a new set of weights for the engine by modifying training approach
 - `chessfun.py`: custom library written for scraping and parsing training data
 - `SmoothBrainChess.py`: classes for the model and engine, functions for training the model
 
Environment:
 - Docker container in works
 - `Python 3.8` (necessary for `PyTorch`, do NOT use newer)
	- `PyTorch`: I installed using the following sequence of commands, worked for me using CUDA 12.1 (likely will not work with newer!)
		- CUDA toolkit is a pre-req, unless you just want to run on CPU (which is probably fine for just inference, and is likely much easier)
		- `conda install pytorch -c pytorch`
		- `conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c pytorch -c nvidia`
		- `conda install pytorch-cuda=12.1 -c pytorch -c nvidia`
		- `conda install pytorch -c pytorch` (yes I had to run this again for some reason to get it working)
	- Conda:
		- `numpy`, `pandas`, `requests`, `re`
	- Pip: 
		- `chess`