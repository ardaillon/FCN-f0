
import numpy as np
from models.f0_to_target_convertor import freq2cents, cents2freq
import time as timeModule

def to_local_average_cents(salience, center=None, fmin=30., fmax=1000., vecSize=486):
    '''
    find the weighted average cents near the argmax bin in output pitch class vector

    :param salience: output vector of salience for each pitch class
    :param fmin: minimum ouput frequency (correponding to the 1st pitch class in output vector)
    :param fmax: maximum ouput frequency (correponding to the last pitch class in output vector)
    :param vecSize: number of pitch classes in output vector
    :return: predicted pitch in cents
    '''

    if not hasattr(to_local_average_cents, 'mapping'):
        # the bin number-to-cents mapping
        fmin_cents = freq2cents(fmin)
        fmax_cents = freq2cents(fmax)
        to_local_average_cents.mapping = np.linspace(fmin_cents, fmax_cents, vecSize) # cents values corresponding to the bins of the output vector

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience)) # index of maximum value in output vector
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience, vecSize=486, smoothing_factor=12):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(vecSize) / vecSize

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(vecSize), range(vecSize))
    transition = np.maximum(smoothing_factor - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(vecSize) * self_emission + np.ones(shape=(vecSize, vecSize)) *
                ((1 - self_emission) / vecSize))

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(vecSize, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])

def get_global_pooling_factor(model):
    globalPoolingFactor = 1
    layers = model.layers
    for l in layers:
        conf = l.get_config()
        if('pool_size' in conf):
            globalPoolingFactor *= conf['pool_size'][0]
    return globalPoolingFactor

def predict_fullConv(model, audio, model_input_size = 1953, viterbi=False, model_srate = 8000):
    activations = model.predict(audio, verbose=1)
    confidence = activations.max(axis=3)[0,:,0]
    activations = np.reshape(activations, (np.shape(activations)[1], np.shape(activations)[3]))

    if (viterbi):
        cents = to_viterbi_cents(activations)
    else:
        cents = []
        for act in activations:
            cents.append(to_local_average_cents(act))
        cents = np.array(cents)
    frequencies = cents2freq(cents)
    frequencies[np.isnan(frequencies)] = 0

    globalPoolingFactor = get_global_pooling_factor(model) # total max pooling on whole network to get the temporal subsampling factor between the input and output of the network. = 8 for FCN-1953 and = 4 for FCN-929
    timeVec = (np.arange(len(frequencies)) * stride * globalPoolingFactor) / model_srate
    return (timeVec, frequencies, confidence, activations)

def get_activation(audio, model, step_size=10, inputSize = 1953):
    from numpy.lib.stride_tricks import as_strided

    # make inputSize-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - inputSize) / hop_length)
    frames = as_strided(audio, shape=(inputSize, n_frames), strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]

    # run prediction and convert the frequency bin weights to Hz
    return model.predict(frames, verbose=1)

def predict_frameWise(audio, model, model_input_size = 1953, viterbi=False, step_size=10):
    activation = get_activation(audio, model, step_size=step_size, inputSize = model_input_size)
    confidence = activation.max(axis=1)

    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence, activation

def sliding_norm(audio, frame_sizes = 1953):
    '''
    Normalize each sample by mean and variance on a sliding window

    :param audio: input audio (full length)
    :param frame_sizes: size of the frames used during training, that should be used for the normalization
    :return: normalized audio
    '''

    from numpy.lib.stride_tricks import as_strided

    if(not frame_sizes%2==0):
        frame_sizes += 1
    n_frames = len(audio)
    audio = np.pad(audio, frame_sizes//2, mode = 'wrap')

    hop_length = 1
    frames = as_strided(audio, shape=(frame_sizes, n_frames), strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose()

    # normalize each frame -- this is expected by the model
    mean = np.mean(frames, axis=1)[:, np.newaxis]
    std = np.std(frames, axis=1)[:, np.newaxis]

    audio = audio[frame_sizes//2:-frame_sizes//2]
    print("np.shape(audio) = "+str(np.shape(audio)))
    mean = mean.flatten()
    print("np.shape(mean) = "+str(np.shape(mean)))
    std = std.flatten()
    print("np.shape(std) = "+str(np.shape(std)))
    audio -= mean
    audio /= std

    return np.array(audio)

def get_audio(sndFile, model_input_size = 1953):
    # read sound :
    from scipy.io import wavfile
    (sr, audio) = wavfile.read(sndFile)
    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)

    sndDuration = len(audio)/sr
    print("duration of sound is "+str(sndDuration))

    if sr != model_srate: # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_srate)

    # pad so that frames are centered around their timestamps (i.e. first frame is zero centered).
    audio = np.pad(audio, int(model_input_size//2), mode='constant', constant_values=0)

    return audio

if __name__ == '__main__':

    # get command-line input arguments :

    from argparse import ArgumentParser

    parser = ArgumentParser(description="train CNN-based f0 analysis given a list of pickle data files with aligned waveform and f0 values")
    parser.add_argument('-i', "--input_file", default=None, help='input sound file to be analyzed')
    parser.add_argument('-o', "--output_file", default=None, help='output f0 file (in either sdif or csv format')
    parser.add_argument('-m', "--model_file", default=None, help='model file store in json format')
    parser.add_argument('-w', "--weights_file", default=None, help='file containing the weights of the model')
    parser.add_argument("-FC", "--full_conv_mode", type=int, default=1, help="run analysis in fully-convolutional mode (otherwise run it frame-wise but slower. Might be used for comparison purpose.)")
    parser.add_argument("-is", "--input_size", type=int, default=1953, help="input size of the network for outputing one value"
                                                                            "(it is the size for the exemples in the training batch during training, and that will be used for the normalisation "
                                                                            "with a sliding window for prediction)")
    parser.add_argument("-sr", "--model_srate", type=int, default=8000, help="sampling rate expected by the model (on which it has been trained) for prediction.")
    parser.add_argument("-sc", "--use_single_core", action="store_true", help="run analysis on a single core CPU instead")
    parser.add_argument("-vit", "--viterbi", action="store_true", help="use viterbi for prediction")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode. Print some informations")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the predicted f0 curve at the end")
    parser.add_argument('-f', "--out_format", default='csv', help='format to use for storing f0 curve. Either sdif or csv')

    args = parser.parse_args()

    ## get parameters :
    # global options :
    FULLCONV = bool(args.full_conv_mode) # about 3x faster using fully-convolutional prediction
    SINGLE_CORE_CPU = args.use_single_core # about 3 to 4 times slower using 1 single CPU for prediction instead of the default configuration
    viterbi = args.viterbi
    # I/O files :
    sndFile = args.input_file
    f0File = args.output_file
    # model files :
    modelFile = args.model_file
    weightsFile = args.weights_file
    # model params :
    model_srate = args.model_srate
    model_input_size = args.input_size
    stride = 1

    if(SINGLE_CORE_CPU): # use a single core of CPU
        import tensorflow as tf
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(config=session_conf)

    #  load model
    if(modelFile is not None):
        json_file = open(modelFile, 'r')
        loaded_model_json = json_file.read()
        from keras.models import model_from_json
        model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights(weightsFile)
        model_input_size = int(model.layers[0].input.shape[1])  # get the input size required by the model
    else:
        if(model_input_size == 1953):
            from models.FCN_1953.core import build_model
            model = build_model(weightsFile=weightsFile, inputSize=1953, training=False)
        elif(model_input_size == 929):
            from models.FCN_929.core import build_model
            model = build_model(weightsFile=weightsFile, inputSize=929, training=False)
        else:
            raise("You need to either provide a prebuilt model file in json format with -m ; or give the expected "
                  "minimum input size of the model (either 1953 or 929 for the provided models).")

    # read the audio
    audio = get_audio(sndFile, model_input_size)

    # run prediction :
    if(not FULLCONV):
        # run prediction :
        startPredictTime = timeModule.time()
        (timeVec, frequencies, confidence, activations) = predict_frameWise(audio, model, model_input_size, viterbi, step_size=10)
        stopPredictTime = timeModule.time()
        predictDuration = stopPredictTime - startPredictTime

    else:
        # normalize audio :
        audio = sliding_norm(audio, frame_sizes=model_input_size)
        audioLen = len(audio)
        audio = np.array([audio])

        # build and load model :
        if(model_input_size == 1953):
            from models.FCN_1953.core import build_model
        elif(model_input_size == 993):
            from models.FCN_993.core import build_model
        elif(model_input_size == 929):
            from models.FCN_929.core import build_model
        model = build_model(weightsFile=weightsFile, inputSize=audioLen, training=False)

        # run prediction :
        startPredictTime = timeModule.time()
        (timeVec, frequencies, confidence, activations) = predict_fullConv(model, audio, model_input_size, viterbi, model_srate)
        stopPredictTime = timeModule.time()
        predictDuration = stopPredictTime - startPredictTime

    if(args.verbose):
        model.summary()

        # computation time :
        print("prediction time = "+str(predictDuration)+"s")

    # store and plot f0 curve :
    if(args.out_format == 'sdif'):
        try:
            from fileio.sdif import Fstoref0
            Fstoref0(f0File, timeVec, frequencies)
        except:
            print("unable to save f0 curve as an sdif file")
    elif(args.out_format == 'csv'):
        f0_data = np.vstack([timeVec, frequencies]).transpose()
        np.savetxt(f0File, f0_data, fmt=['%.3f', '%.3f'], delimiter='   ', comments='') # format is "time   frequency" in 2 columns
    print("Saved the estimated frequencies values at {}".format(f0File))

    if(args.plot):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(timeVec, frequencies)
        plt.show()

