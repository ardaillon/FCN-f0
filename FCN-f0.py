
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from prediction import run_prediction


def main():
    """
    This is a script for running the pre-trained pitch estimation model, FCN-f0,
    by taking sound files(s) as input. For each input WAV, a CSV file containing:
        time, frequency, confidence
        0.00, 424.24, 0.42
        0.01, 422.42, 0.84
        ...
    is created as the output, where the first column is a timestamp in seconds,
    the second column is the estimated frequency in Hz, and the third column is
    a value between 0 and 1 indicating the model's voicing confidence (i.e.
    confidence in the presence of a pitch for every frame).
    The script can also optionally save the output activation matrix of the
    model to an npy file, where the matrix dimensions are (n_frames, 360) using
    a hop size of 10 ms (there are 360 pitch bins covering 20 cents each).
    The script can also output a plot of the activation matrix, including an
    optional visual representation of the model's voicing detection.
    """

    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('filename', nargs='+',
                        help='path to one ore more WAV file(s) to analyze OR '
                             'can be a directory')
    parser.add_argument('--output', '-o', default=None,
                        help='directory to save the ouptut file(s). if not '
                             'given, the output will be saved to the same '
                             'directory as the input WAV file(s)')
    parser.add_argument('--modelTag', '-m', default='993',
                        choices=['929', '993', '1953', 'CREPE'],
                        help='String specifying which model to use; '
                             'results are close in terms of accuracy '
                             'but differ in terms of latency and speed '
                             '(see article). We advice to use the model 993')
    parser.add_argument('--viterbi', '-V', action='store_true',
                        help='perform Viterbi decoding to smooth the pitch '
                             'curve')
    parser.add_argument('-f', "--out_format", default='sdif',
                        help='format to use for storing f0 curve. Either '
                             'sdif or csv')
    parser.add_argument("-FC", "--full_conv_mode", type=int, default=1,
                        help="run analysis in fully-convolutional mode "
                             "(otherwise run it frame-wise but slower. Might "
                             "be used for comparison purpose.)")
    parser.add_argument("-sc", "--use_single_core", action="store_true",
                        help="run analysis on a single core CPU instead")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose mode. Print some informations")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot the predicted f0 curve at the end")

    args = parser.parse_args()

    ## get parameters :

    # I/O :
    filename = args.filename
    output = args.output

    # model :
    modelTag = args.modelTag

    # global analysis options :
    viterbi = args.viterbi
    outFormat = args.out_format

    # computation mode:
    FULLCONV = bool(args.full_conv_mode)
    SINGLE_CORE_CPU = args.use_single_core

    # other options :
    verbose = args.verbose
    plot = args.plot


    if(SINGLE_CORE_CPU): # use a single core of CPU
        import tensorflow as tf
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(config=session_conf)

    if(args.modelTag == 'CREPE'):
        if(FULLCONV == True):
            print("It is not possible to use the fully-convolutional model witht the CREPE model."
                  "The prediction will be run frame-wise with a hop size of 10ms.")
            FULLCONV = False

    run_prediction(filename, output, modelTag, viterbi, outFormat, FULLCONV, verbose, plot)

    return


if __name__ == '__main__':
    main()

