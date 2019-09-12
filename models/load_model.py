
import os


def load_model(modelName, from_json = False):
    '''
    load model from json file and corresponding weights from hdf5 file
    '''

    curDir = os.path.dirname(os.path.abspath(__file__))
    print(os.path.isdir(curDir))
    if modelName == '929':
        modelDir = os.path.join(curDir, 'FCN_929')
        modelFile = os.path.join(modelDir, 'model.json')
        weightsFile = os.path.join(modelDir, 'weights.h5')
        from models.FCN_929.core import build_model
    elif modelName == '993':
        modelDir = os.path.join(curDir, 'FCN_993')
        modelFile = os.path.join(modelDir, 'model.json')
        weightsFile = os.path.join(modelDir, 'weights.h5')
        from models.FCN_993.core import build_model
    elif modelName == '1953':
        modelDir = os.path.join(curDir, 'FCN_1953')
        modelFile = os.path.join(modelDir, 'model.json')
        weightsFile = os.path.join(modelDir, 'weights.h5')
        from models.FCN_1953.core import build_model
    elif modelName == 'CREPE':
        modelDir = os.path.join(curDir, 'CREPE-speech')
        modelFile = os.path.join(modelDir, 'model.json')
        weightsFile = os.path.join(modelDir, 'weights.h5')
    else:
        raise("Model doesn't exist. Available options are ['929', '993', '1953', 'CREPE']")

    if(from_json):
        json_file = open(modelFile, 'r')
        loaded_model_json = json_file.read()
        from keras.models import model_from_json
        model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights(weightsFile)
    else:
        # for FULLCONV mode, input size is not defined
        model = build_model(learning_rate=0.0002, weightsFile=weightsFile, inputSize=None, training=False)

    return model


def get_infos_from_tag(modelTag):
    if(modelTag == 'CREPE'):
        model_input_size = 1024
        model_srate = 16000.
    elif(modelTag == '929'):
        model_input_size = 929
        model_srate = 8000.
    elif(modelTag == '993'):
        model_input_size = 993
        model_srate = 8000.
    elif(modelTag == '1953'):
        model_input_size = 1953
        model_srate = 8000.
    return (model_input_size, model_srate)

