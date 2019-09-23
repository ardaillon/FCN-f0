# FCN-f0
Code for running monophonic pitch (F0) estimation using the fully-convolutional neural network models described in the publication :

L. Ardaillon and A. Roebel, "Fully-Convolutional Network for Pitch Estimation of Speech Signals", Proc. Interspeech, 2019.

We kindly request academic publications making use of our FCN models to cite this paper, which can be dowloaded from the following url : https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2815.pdf

## Description
The code provided in this repository aims at performing monophonic pitch (F0) estimation using Fully-Convolutional Neural Networks. 
It is partly based on the code from the CREPE repository => https://github.com/marl/crepe

The provided code allows to run the pitch estimation on given sound files using the provided pretrained models, but no code is currently provided to train the model on new data.
Three different fully-convolutional pre-trained models are provided.
Those models have been trained exclusively on (synthetic) speech data and may thus not perform as well on other types of sounds such as music instruments. Note that the output F0 values are also limited to the target range [30-1000]Hz, which is suitable for vocal signals (including high-pitched soprano singing). 

The models, algorithm, training, and evaluation procedures have been described in a publication entitled "Fully-Convolutional Network for Pitch Estimation of Speech Signals", presented at the Interspeech 2019 conference (https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2815.pdf).

Below are the results of our evaluations comparing our models to the SWIPE algorithm and CREPE model, in terms of Raw Pitch Accuracy (average value and standard deviation, on both a test database of synthetic speech "PAN-synth" and a database of real speech samples with manually-corrected ground truth "manual"). For this evaluation, the CREPE model has been evaluated both with the provided pretrained model from the CREPE repository ("CREPE" in the table) and with a model retrained from scratch on our synthetic database ("CREPE-speech"). FCN models have been evluated on 8kHz audio, while CREPE and SWIPE have been trained and evaluated on 16kHz audio.
<table>
    <thead>
        <tr>
            <th> </th>
            <th><sub>FCN-1953</sub></th>
            <th><sub>FCN-993</sub></th>
            <th><sub>FCN-929</sub></th>
            <th><sub>CREPE</sub></th>
            <th><sub>CREPE-speech</sub></th>
            <th><sub>SWIPE</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><sub>PAN-synth (25 cents)</sub></td>
            <td><sub>93.62 &plusmn 3.34%</sub></td>
            <td><sub><strong>94.31 &plusmn 3.15%</strong></sub></td>
            <td><sub>93.50 &plusmn 3.43%</sub></td>
            <td><sub>77.62 &plusmn 9.31%</sub></td>
            <td><sub>86.92 &plusmn 8.28%</sub></td>
            <td><sub>84.56 &plusmn 11.68%</sub></td>
        </tr>        
        <tr>
            <td><sub>PAN-synth (50 cents)</sub></td>
            <td><sub>98.37 &plusmn 1.62%</sub></td>
            <td><sub><strong>98.53 &plusmn 1.54%</strong></sub></td>
            <td><sub>98.27 &plusmn 1.73%</sub></td>
            <td><sub>91.23 &plusmn 6.00%</sub></td>
            <td><sub>97.27 &plusmn 2.09%</sub></td>
            <td><sub>93.10 &plusmn 7.26%</sub></td>
        </tr>        
        <tr>
            <td><sub>PAN-synth (200 cents)</sub></td>
            <td><sub><strong>99.81 &plusmn 0.64%</strong></sub></td>
            <td><sub>99.79 &plusmn 0.65%</sub></td>
            <td><sub>99.77 &plusmn 0.73%</sub></td>
            <td><sub>95.65 &plusmn 5.17%</sub></td>
            <td><sub>99.25 &plusmn 1.07%</sub></td>
            <td><sub>97.51 &plusmn 4.90%</sub></td>
        </tr>        
        <tr>
            <td><sub>manual (50 cents)</sub></td>
            <td><sub>88.32 &plusmn 6.33%</sub></td>
            <td><sub>88.57 &plusmn 5.77%</sub></td>
            <td><sub><strong>88.88 &plusmn 5.73%</strong></sub></td>
            <td><sub>87.03 &plusmn 7.35%</sub></td>
            <td><sub>88.45 &plusmn 5.70%</sub></td>
            <td><sub>85.93 &plusmn 7.62%</sub></td>
        </tr>        
        <tr>
            <td><sub>manual (200 cents)</sub></td>
            <td><sub>97.35 &plusmn 3.02%</sub></td>
            <td><sub>97.31 &plusmn 2.56%</sub></td>
            <td><sub><strong>97.36 &plusmn 2.51%</strong></sub></td>
            <td><sub>92.57 &plusmn 5.22%</sub></td>
            <td><sub>96.63 &plusmn 2.91%</sub></td>
            <td><sub>95.03 &plusmn 4.04%</sub></td>
        </tr>
    </tbody>
</table>

Our synthetic speech database has been created by resynthesizing the BREF [2] and TIMIT [3] databases using the PAN synthesis engine, described in [4, Section 3.5.2].

We also compared the different models and algorithms in terms of potential latency (with real-time implementation in mind), where the latency corresponds to the duration of half the (minimal) input size, and computation times on both a GPU and single-core CPU :
<table>
    <thead>
        <tr>
            <th> </th>
            <th><sub>FCN-1953</sub></th>
            <th><sub>FCN-993</sub></th>
            <th><sub>FCN-929</sub></th>
            <th><sub>CREPE</sub></th>
            <th><sub>SWIPE</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><sub>latency (s)</sub></td>
            <td><sub>0.122</sub></td>
            <td><sub>0.062</strong></sub></td>
            <td><sub>0.058</sub></td>
            <td><sub><strong>0.032</strong></sub></td>
            <td><sub>0.128</sub></td>
        </tr>        
        <tr>
            <td><sub>Computation time on GPU (s)</sub></td>
            <td><sub>0.016</sub></td>
            <td><sub><strong>0.010</sub></td>
            <td><sub>0.021</sub></td>
            <td><sub>0.092</sub></td>
            <td><sub>X</sub></td>
        </tr>        
        <tr>
            <td><sub>Computation time on CPU (s)</sub></td>
            <td><sub>1.65</sub></td>
            <td><sub>0.89</sub></td>
            <td><sub>3.34</sub></td>
            <td><sub>14.79</sub></td>
            <td><sub><strong>0.63</strong></sub></td>
        </tr>
    </tbody>
</table>

## Example command-line usage (using provided pretrained models)
#### Default analysis : This will run the FCN-993 model and output the result as a csv file in the same folder than the input file (replacing the file extension by ".csv")
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav

#### Run the analysis on a whole folder of audio files :
python /path_to/FCN-f0/FCN-f0.py /path_to/audio_files

#### Specify an output directory or file name with "-o" option(if directory doesn't exist, it will be created):
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -o /path_to/output.f0.csv
python /path_to/FCN-f0/FCN-f0.py /path_to/audio_files -o /path_to/output_dir

#### Choose a specific model for running the analysis (default is FCN-993):
Use FCN-929 model :
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -m 929 -o /path_to/output.f0-929.csv

Use FCN-993 model :
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -m 993 -o /path_to/output.f0-993.csv

Use FCN-1953 model :
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -m 1953 -o /path_to/output.f0-1953.csv

Use CREPE-speech model :
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -m CREPE -o /path_to/output.f0-CREPE.csv

#### Apply viterbi smoothing of output :
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -vit

#### Output result to sdif format (requires installing the eaSDIF python library. Default format is csv):
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -f sdif

#### Deactivate fully-convolutional mode (For comparison purpose, but not recommanded otherwise, as it makes the computation much slower):
python /path_to/FCN-f0/FCN-f0.py /path_to/test.wav -FC 0

## References
[1] Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello. "CREPE: A Convolutional Representation for Pitch Estimation", Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

[2] J. L. Gauvain, L. F. Lamel, and M. Eskenazi, “Design Considerations and Text Selection for BREF, a large French Read-Speech Corpus,” 1st International Conference on Spoken Language Processing, ICSLP, no. January 2013, pp. 1097–1100, 1990. http://www.limsi.fr/~lamel/kobe90.pdf

[3] V. Zue, S. Seneff, and J. Glass, “Speech Database Development At MIT : TIMIT And Beyond,” vol. 9, pp. 351–356, 1990.

[4] L. Ardaillon, “Synthesis and expressive transformation of singing voice,” Ph.D. dissertation, EDITE; UPMC-Paris 6 Sorbonne Universités, 2017.
