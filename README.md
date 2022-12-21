# Twitter Sentiment Towards Family During the Holidays
This is the implementation corresponding to a Twitter sentiment analysis. Two notebooks are included:
- [train.ipynb](train.ipynb): Train an SVM classifier on the Tweeteval [[1]](#1) sentiment dataset (SemEval-2017 task 4 subtask A [[5]](#5). The SVM features are heavily inspired by [[4]](#4). Includes cross-validation, hyperparameter optimzalization, and ablation testing.
- [experiment.ipynb](experiment.ipynb): Scrape Twitter for tweets about family, and classify them using the model trained in [train.ipynb](train.ipynb). Includes filtering, validation, and result figures & tables.

## Installation
To get started, follow these steps:
- Download the Tweet POS-tagger and tokenizer `ark-tweet-nlp-0.3.2.jar`[[2]](#2) from https://code.google.com/archive/p/ark-tweet-nlp/downloads and place it in the root folder.
- Create the `data/lexicons` folder and download the following lexicons:
    - The lexicon from [[3]](#3), available at https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107. Place `negative-words.txt` and `positive-words.txt` in `data/lexicons`.
    - The Hashtag and Sentiment140 lexicons from [[4]](#4), available at http://www.saifmohammad.com/WebPages/Abstracts/NRC-SentimentAnalysis.htm. Rename their `unigrams-pmilexion.txt` files to `hashtag.txt` and `sentiment140.txt` respectively, and place them in `data/lexicons`.
    - The MPQA lexicon from [[6]](#6), available at https://github.com/candlewill/neo4j-sentiment-analysis. Rename `sentimentDict.csv` to `MPQA.txt` and place it in `data/lexicons`.
- Create the `data/tweeteval` folder. Download the Tweeteval sentiment dataset from [[1]](#1) and [[5]](#5), available at https://github.com/cardiffnlp/tweeteval and place the `.txt` files in `data/tweeteval`.
- Create the `data/tweets` folder.
- Create the `data/temp` folder.
- Create the `models` folder.

You can now run the notebooks!

## References
<a id="1">[1]</a> Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, and Leonardo Neves. 2020. TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. In Findings of the Association for Computational Linguistics: EMNLP 2020. Association for Computational Linguistics, Online, 1644–1650. https://doi.org/10.18653/v1/2020.findings-emnlp.148

<a id="2">[2]</a> Kevin Gimpel, Nathan Schneider, Brendan O’Connor, Dipanjan Das, Daniel P Mills, Jacob Eisenstein, Michael Heilman, Dani Yogatama, Jeffrey Flanigan, and Noah A Smith. 2011. Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. 42–47.

<a id="3">[3]</a> Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining. 168–177.

<a id="4">[4]</a> Saif Mohammad, Svetlana Kiritchenko, and Xiaodan Zhu. 2013. NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets. In Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval-2013). 321–327.

<a id="5">[5]</a> Sara Rosenthal, Noura Farra, and Preslav Nakov. 2017. SemEval-2017 task 4: Sentiment analysis in Twitter. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). 502–518.

<a id="6">[6]</a> Theresa Wilson, Janyce Wiebe, and Paul Hoffmann. 2005. Recognizing contextual polarity in phrase-level sentiment analysis. In Proceedings of human language technology conference and conference on empirical methods in natural language processing. 347–354.
