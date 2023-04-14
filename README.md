## Code and data for the paper "[Evaluating Morphological Generalisation in Machine Translation by Distribution-Based Compositionality Assessment](https://openreview.net/forum?id=1sGdp5g0NP)"

### Structure
* scripts numbered 01-13 are meant to be run in succession 
* [run.sh](run.sh) provides examples of running the scripts
* [exp/subset-d-1m/data](exp/subset-d-1m/data) contains the 1M sentence pair dataset
* `exp/subset-d-1m/splits/*/*/*/ids_{train,test_full}.txt.gz` contain the data splits with different compound divergences and different random initialisations

### Dependencies
* Data is from the [Tatoeba Challenge data release](https://github.com/Helsinki-NLP/Tatoeba-Challenge) (eng-fin set)
* Data filtering is done using [OpusFilter](https://github.com/Helsinki-NLP/OpusFilter)
* Morphological parsing is done using [TNPP](https://turkunlp.org/Turku-neural-parser-pipeline/), CoNLL-U format parsed using [this parser](https://github.com/EmilStenstrom/conllu)
* Data split algorithm uses [PyTorch](https://pytorch.org/)
* Tokenisers are trained using [sentencepiece](https://github.com/google/sentencepiece)
* Translation systems are trained with [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* Evaluating translations is done with [sacreBLEU](https://github.com/mjpost/sacrebleu)
