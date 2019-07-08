# FCN-f0
Code for running monophonic pitch (F0) estimation using the fully-convolutional neural network models described in the publication :

L. Ardaillon and A. Roebel, "Fully-Convolutional Network for Pitch Estimation of Speech Signals", Proc. Interspeech, 2019.

We kindly request academic publications making use of our FCN models to cite this paper.

## Description
The code provided in this repository aims at performing monophonic pitch (F0) estimation using Fully-Convolutional Neural Networks. 
It is partly based on the code from the CREPE repository => https://github.com/marl/crepe

The provided code allows to run the pitch estimation on given sound files using the provided pretrained models, but no code is currently provided to train the model on new data.
Three different fully-convolutional pre-trained models are provided.
Those models have been trained exclusively on (synthetic) speech data and may thus not perform as well on other types of sounds such as music instruments.

The models, algorithm, training, and evaluation procedures have been described in a publication entitled "Fully-Convolutional Network for Pitch Estimation of Speech Signals", to be presented at the Interspeech 2019 conference.

Below are the results of our evaluations comparing our models to the SWIPE algorithm and CREPE model, in terms of Raw pitch accuracy (average value and standard deviation, on both a test database of synthetic speech "PAN-synth" and a database of real speech samples with manually-corrected ground truth "manual"). For this evaluation, the CREPE model has been evaluated both with the provided pretrained model from the CREPE repository ("CREPE" in the table) and with a model retrained from scratch on our synthetic database. FCN models have been evluated on 8kHz audio, while CREPE and SWIPE have been trained and evaluated on 16kHz audio.
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

[2] J. L. Gauvain, L. F. Lamel, and M. Eskenazi, “Design Considerations and Text Selection for BREF, a large French Read-Speech Corpus,” 1st International Conference on Spoken Language Processing, ICSLP, no. January 2013, pp. 1097–1100, 1990. http://www.limsi.fr/~lamel/kobe90.pdf

[3] V. Zue, S. Seneff, and J. Glass, “Speech Database Development At MIT : TIMIT And Beyond,” vol. 9, pp. 351–356, 1990.

[4] L. Ardaillon, “Synthesis and expressive transformation of singing voice,” Ph.D. dissertation, EDITE; UPMC-Paris 6 Sorbonne Universités, 2017.
