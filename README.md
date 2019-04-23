# FCN-f0
Fully-Convolutional Network for Pitch Estimation of Speech Signals

## Description
The code provided in this repository aims at performing monophonic pitch (f0) estimation.
It is partly based on the code from the CREPE repository => https://github.com/marl/crepe

Two different fully-convolutional pre-trained models are provided.
Those models have been trained exclusively on speech data and may thus not perform as well on other types of sounds.

The code currently provided only allows to run the pitch estimation on given sound files using the provided pretrained models (no code is currently provided to train the model on new data).

The models, algorithm, training, and evaluation procedures have been described in a publication entitled "Fully-Convolutional Network for Pitch Estimation of Speech Signals", submitted to the Interspeech 2019 conference (currently under review).

## Example command-line usage (using provided pretrained models)
#### model FCN-1953
python /path_to/FCN-f0/prediction.py -i /path_to/test.wav -o /path_to/test-FCN_1953.f0.csv -m /path_to/FCN-f0/models/FCN_1953/model.json -w /path_to/FCN-f0/models/FCN_1953/weights.h5 --use_single_core --verbose --plot

or

python /path_to/FCN-f0/prediction.py -i /path_to/test.wav -o /path_to/test-FCN_1953-no_json.f0.csv -w /path_to/FCN-f0/models/FCN_1953/weights.h5 -is 1953 --use_single_core --verbose --plot

#### model FCN-929
python /path_to/FCN-f0/prediction.py -i /path_to/test.wav -o /path_to/test-FCN_929.f0.csv -m /path_to/FCN-f0/models/FCN_929/model.json -w /path_to/FCN-f0/models/FCN_929/weights.h5 --use_single_core --verbose --plot

or

python /path_to/FCN-f0/prediction.py -i /path_to/test.wav -o /path_to/test-FCN_929-no_json.f0.csv -w /path_to/FCN-f0/models/FCN_929/weights.h5 -is 929 --use_single_core --verbose --plot

## References
[1] Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello. "CREPE: A Convolutional Representation for Pitch Estimation", Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.
