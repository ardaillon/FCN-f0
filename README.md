# FCN-f0
Fully-Convolutional Network for Pitch Estimation of Speech Signals

The code provided in this repository aims at performing monophonic pitch (f0) estimation.
It is partly based on the code from the CREPE repository => https://github.com/marl/crepe

Two different fully-convolutional pre-trained models are provided.
Those models have been trained exclusively on speech data and may thus not perform as well on other types of sounds.

The code currently provided only allows to run the pitch estimation on given sound files using the provided pretrained models (no code is currently provided to train the model on new data).

The models, algorithm, training, and evaluation procedures have been described in a publication entitled "Fully-Convolutional Network for Pitch Estimation of Speech Signals", submitted to the Interspeech 2019 conference (currently under review).

#Example command-line usage
python /path_to/FCN-f0/prediction.py -i /path_to/test.wav -o /path_to/test.f0.csv -m /path_to/FCN-f0/models/FCN_1953/model.json -w /Upath_to/FCN-f0/models/FCN_1953/weights.h5 -FC 1 -is 1953 -sr 8000 -sc -vit -v -p -f csv

#References
[1] Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello. "CREPE: A Convolutional Representation for Pitch Estimation", Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.
