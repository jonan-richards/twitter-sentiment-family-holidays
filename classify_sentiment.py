import pandas as pd
import numpy as np
import re
import csv
import subprocess
from functools import partial
import pickle
import logging
from copy import deepcopy

from nltk.corpus import stopwords

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer

import optunity
import optunity.metrics

logger = logging.getLogger(__name__)

########################
# Constants
########################

stop_words = set(stopwords.words('english'))
patterns = {key: re.compile(pattern) for key, pattern in {
    'negation': ''.join(r"""(?:
            ^(?:never|no|nothing|nowhere|noone|none|not|
                havent|hasnt|hadnt|cant|couldnt|shouldnt|
                wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
            )$
        )
        |
        n't""".split()),
    'clause_punct': r'^[.:;!?]+$',  
    'newline': r'\n|\r',  
    'multi_quote': r'"{2,}',
    'multi_whitespace': r'\s{2,}',
    'repeated_alpha': r'([a-z])\1{2,}',  
    'username': r'@\w{1,15}',
    'hashtag': r'#\w+',
    'link': r'(https?://)?t.co/\w*',
    'all_caps': r'^[^a-z]*[A-Z][^a-z]*$',
}.items()}
polarities = [1, -1]
pos_tags = {'D', 'A', '!', 'T', '@', 'R', '&', 'V', 'E', 'Y', 'G', 'N', '^', '#', ',', 'U', 'S', 'Z', 'O', '~', '$', 'X', 'L', 'P'}

########################
# Preprocessing
########################

def preprocess(data, temp_dir):
    input_path = f'{temp_dir}/tokenize_input.tsv'
    output_path = f'{temp_dir}/tokenize_output.tsv'

    data['text'] = data['text'].apply(lambda text: patterns['multi_whitespace'].sub(' ', patterns['multi_quote'].sub('"', patterns['newline'].sub(' ',text))))

    with open(input_path, 'w', encoding='utf-8') as input_file:
        data['text'].to_csv(input_path, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        subprocess.call('java -jar ark-tweet-nlp-0.3.2.jar {} '.format(input_file.name).split(' '), stdout=output_file)

    output = pd.read_csv(output_path, sep='\t', header=None, names = ['tokens', 'pos_tags', 'confidence', 'text'], quoting=csv.QUOTE_NONE)

    data['pos_tags'] = output['pos_tags'].apply(lambda tags: tags.split(' '))
    data['tokens'] = output['tokens'].apply(lambda tokens: [token for token in tokens.split(' ') if token != ''])
    
    data['text_lower'] = data['text'].str.lower()
    data['tokens_lower'] = data['tokens'].apply(lambda tokens: [token.lower() for token in tokens])
    data['tokens_processed'] = data['tokens_lower'].apply(get_tokens_processed)

    data['tokens_polarity'] = data['tokens_processed'].apply(get_tokens_polarity)
    data['text_processed'] = data['tokens_processed'].apply(lambda tokens: ' '.join(tokens))
    data['text_polarized'] = data.apply(lambda row: get_text_polarized(row['tokens_processed'], row['tokens_polarity']), axis=1)

    data['pos_polarized'] = data.apply(lambda row: get_pos_polarized(row['pos_tags'], row['tokens_polarity']), axis=1)

def get_tokens_polarity(tokens):
    polarity = 1
    tokens_polarity = []

    for i, token in enumerate(tokens):
        tokens_polarity.append(polarity)
        
        if patterns['negation'].search(token):
            polarity = polarity * -1
        elif patterns['clause_punct'].search(token):
            polarity = 1

    return tokens_polarity

def get_text_polarized(tokens, tokens_polarity):
    return ' '.join([token if polarity == 1 else token + '_NEG' for (token, polarity) in zip(tokens, tokens_polarity) if token not in stop_words])

def get_pos_polarized(pos_tags, tokens_polarity):
    return ' '.join([tag if polarity == 1 else tag + '_NEG' for (tag, polarity) in zip(pos_tags, tokens_polarity)])

def get_tokens_processed(tokens_lower):
    tokens = tokens_lower
    tokens = [patterns['repeated_alpha'].sub('\\1\\1', token) for token in tokens]
    tokens = [patterns['username'].sub('USERNAME', token) for token in tokens]
    tokens = [patterns['hashtag'].sub('HASHTAG', token) for token in tokens]
    tokens = [patterns['link'].sub('LINK', token) for token in tokens]
    return tokens

########################
# Lexicons
########################

def get_lexicon_simple(filename, value):
    with open(filename, 'r') as file:
        return {word: value for word in file.read().splitlines() if not word.startswith(';') and not word == ''}

def get_lexicon_nrc(filename):
    lexicon = {}

    with open(filename, 'r') as file:
        for line in file.read().splitlines():
            parts = line.split('\t')
            lexicon[parts[0]] = float(parts[1])

    return lexicon

def get_lexicon_mpqa(filename):
    lexicon = {}

    with open(filename, 'r') as file:
        for i, line in enumerate(file.read().splitlines()):
            if i == 0:
                continue

            parts = line.split(',')

            lexicon[parts[2]] = (1 if parts[5] == 'positive' else -1) * (5 if parts[0] == 'strongsubj' else 1)

    return lexicon

def get_lexicons(positive_path, negative_path, hashtag_path, sent140_path, mpqa_path):
    return {
        # Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews.", http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
        'simple': dict(
            list(get_lexicon_simple(positive_path, 1).items()) +
            list(get_lexicon_simple(negative_path, -1).items())
        ),

        # "NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets", http://www.saifmohammad.com/WebPages/Abstracts/NRC-SentimentAnalysis.htm
        'hashtag': get_lexicon_nrc(hashtag_path),
        'sent140': get_lexicon_nrc(sent140_path),

        # "Theresa Wilson, Janyce Wiebe, and Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis", https://github.com/candlewill/neo4j-sentiment-analysis
        'mpqa': get_lexicon_mpqa(mpqa_path)
    }

########################
# Feature extraction
########################

def extract_features_counts(input, exclude=[]):
    features = []

    # Feature: Word count
    features.append(len(input['tokens']))

    # Feature: Word count excluding stopwords
    ex_stop_words = len([x for x in input['tokens_lower'] if x not in stop_words])
    features.append(ex_stop_words)

    # Feature: Number of characters
    features.append(len(input['text']))

    # Feature: Number of all-caps words
    features.append(sum([patterns['all_caps'].search(token) is not None for token in input['tokens']]))

    # Feature: Number of words with a character repeated 3+ times
    features.append(sum([patterns['repeated_alpha'].search(token) is not None for token in input['tokens_lower']]))

    # Feature: Number of negated contexts
    features.append(sum([1 if polarity == -1 and (i == 0 or input['tokens_polarity'][i-1] == 1) else 0 for i, polarity in enumerate(input['tokens_polarity'])]))

    return features
    
def extract_features_lexicon(input, exclude=[], **kwargs):
    features = []

    for key, lexicon in kwargs.items():
        if f'lexicon_{key}' in exclude:
            continue

        scores = [
            lexicon[token] * token_polarity for token_polarity, token in zip(input['tokens_polarity'], input['tokens_lower']) 
            if token in lexicon
        ]

        for polarity in polarities:
            polarity_scores = [abs(score) for score in scores if score/abs(score) == polarity]
            count = len(polarity_scores)
            score = sum(polarity_scores)
            maximum = max(polarity_scores, default = 0)
            last = polarity_scores[-1] if len(polarity_scores) > 0 else 0

            features.extend([count, score, maximum, last])


    return features

feature_groups = {
    'counts': extract_features_counts,
    'lexicon': extract_features_lexicon,
}
def extract_features(input, exclude = [], **kwargs):
    features = []
    
    for group, extract in feature_groups.items():
        if group not in exclude:
            features.extend(extract(input, exclude=exclude, **kwargs.get(group, {})))

    return np.array(features, dtype='float32')

def extract_features_dataframe(input, **kwargs):
    return np.stack(input.apply(lambda row: extract_features(row.to_dict(), **kwargs), axis=1).values, axis=0)

########################
# Pipeline
########################

def create_ngrams_pipeline(column, type, n_components, min_n, max_n):
    return ColumnTransformer([(f'ngram_{column}', Pipeline([
        (f'ngram_tfidf', TfidfVectorizer(
            sublinear_tf=True,
            use_idf=True,
            lowercase=False,
            token_pattern=r'(?u)(?:^|(?<= ))[^ ]{1,}(?:$|(?= ))',
            ngram_range=(min_n, max_n),
            analyzer=type,
        )), 
        (f'ngram_svd', TruncatedSVD(
            n_components=n_components
        ))
    ]), column)])

def create_features_pipeline(n_pos, n_word, n_char, exclude = [], **kwargs):
    return FeatureUnion([
        ('custom_features', FunctionTransformer(partial(extract_features_dataframe, exclude=exclude, **kwargs))),
        ('pos_ngram', create_ngrams_pipeline('pos_polarized', 'word', n_pos, 1, 3)),
        ('word_ngram', create_ngrams_pipeline('text_polarized', 'word', n_word, 1, 3)),
        ('char_ngram', create_ngrams_pipeline('text_processed', 'char', n_char, 1, 5)),
    ])

def create_classifier_pipeline(log_C):
    return Pipeline([
        ('scale', StandardScaler()), 
        ('svc', LinearSVC(C=10**log_C))
    ])

def create_pipeline(n_pos, n_word, n_char, log_C, exclude=[], **kwargs):
    return Pipeline([
        ('features', create_features_pipeline(n_pos, n_word, n_char, exclude, **kwargs)),
        ('classifier', create_classifier_pipeline(log_C))
    ])

########################
# Evaluation
########################

def evaluate(y_true, y_pred):
    # Positive
    pp = sum([true == pred == 2 for (true, pred) in zip(y_true, y_pred)])
    pred_p = sum([pred == 2 for pred in y_pred])
    true_p = sum([true == 2 for true in y_true])
    pos_precision = pp / pred_p if pred_p != 0 else 0
    pos_recall = pp / true_p if true_p != 0 else 0
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall) if pos_precision + pos_recall != 0 else 0

    # Negative
    nn = sum([true == pred == 0 for (true, pred) in zip(y_true, y_pred)])
    pred_n = sum([pred == 0 for pred in y_pred])
    true_n = sum([true == 0 for true in y_true])
    neg_precision = nn / pred_n if pred_n != 0 else 0
    neg_recall = nn / true_n if true_n != 0 else 0
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if neg_precision + neg_recall != 0 else 0

    # Neutral
    qq = sum([true == pred == 1 for (true, pred) in zip(y_true, y_pred)])
    pred_q = sum([pred == 1 for pred in y_pred])
    true_q = sum([true == 1 for true in y_true])
    q_precision = qq / pred_q if pred_q != 0 else 0
    q_recall = qq / true_q if true_q != 0 else 0
    q_f1 = 2 * q_precision * q_recall / (q_precision + q_recall) if q_precision + q_recall != 0 else 0

    return {
        'positive': {
            'predicted': pred_p,
            'precision': pos_precision,
            'recall': pos_recall,
            'f1': pos_f1,
        },
        'negative': {
            'predicted': pred_n,
            'precision': neg_precision,
            'recall': neg_recall,
            'f1': neg_f1,
        },
        'neutral': {
            'predicted': pred_q,
            'precision': q_precision,
            'recall': q_recall,
            'f1': q_f1,
        },
        'recall': (pos_recall + neg_recall + q_recall) / 3,
        'f1pn': (pos_f1 + neg_f1) / 2
    }

def average_scores(scores):
    return {
        'positive': {
            'predicted': sum([score['positive']['predicted'] for score in scores]) / len(scores),
            'precision': sum([score['positive']['precision'] for score in scores]) / len(scores),
            'recall': sum([score['positive']['recall'] for score in scores]) / len(scores),
            'f1': sum([score['positive']['f1'] for score in scores]) / len(scores),
        },
        'negative': {
            'predicted': sum([score['negative']['predicted'] for score in scores]) / len(scores),
            'precision': sum([score['negative']['precision'] for score in scores]) / len(scores),
            'recall': sum([score['negative']['recall'] for score in scores]) / len(scores),
            'f1': sum([score['negative']['f1'] for score in scores]) / len(scores),
        },
        'neutral': {
            'predicted': sum([score['neutral']['predicted'] for score in scores]) / len(scores),
            'precision': sum([score['neutral']['precision'] for score in scores]) / len(scores),
            'recall': sum([score['neutral']['recall'] for score in scores]) / len(scores),
            'f1': sum([score['neutral']['f1'] for score in scores]) / len(scores),
        },
        'recall': sum([score['recall'] for score in scores]) / len(scores),
        'f1pn': sum([score['f1pn'] for score in scores]) / len(scores)
    }

########################
# Training
########################

def train_cross_validation(train_data, lexicons, n_pos, n_word, n_char, log_C):
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    scores = []

    logger.info(f'Creating features...')

    features = create_features_pipeline(n_pos, n_word, n_char, lexicon=lexicons).fit_transform(train_data)
    labels = train_data['label'].values

    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(features, labels)):
        logger.info('Fold {}:'.format(fold_id + 1))

        train_features = features[train_indexes]
        train_labels = labels[train_indexes]

        validation_features = features[validation_indexes]
        validation_labels = labels[validation_indexes]

        classifier = create_classifier_pipeline(log_C).fit(train_features, train_labels)
        y_pred = classifier.predict(validation_features)

        score = evaluate(validation_labels, y_pred)
        scores.append(score)

        logger.debug(f'Score: {score}')

    score = average_scores(scores)

    logger.info('Done!')
    logger.debug(f'Average score: {score}')

def train_hyperparameter_optimalization(
    train_data, lexicons, 
    n_svd_evals, 
    n_classify_evals, 
    range_n_pos, 
    range_n_word, 
    range_n_char, 
    range_log_C
):
    @optunity.constraints.constrained([lambda n_pos, n_word, n_char: n_pos + n_word + n_char < 2000])
    def svm_recall_svd(n_pos, n_word, n_char):
        n_pos, n_word, n_char = round(n_pos), round(n_word), round(n_char)

        logger.info(f'POS components: {n_pos}, word components: {n_word}, char components: {n_char}')
        logger.info('Creating features...')

        features = create_features_pipeline(n_pos, n_word, n_char, lexicon=lexicons).fit_transform(train_data)
        labels = train_data['label'].values

        @optunity.cross_validated(x=features, y=labels, num_folds=10, num_iter=1)
        def svm_recall(x_train, y_train, x_test, y_test, log_C):
            logger.info(f'Log C: {log_C}')

            classifier = create_classifier_pipeline(log_C).fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            scores = evaluate(y_test, y_pred)

            logger.debug(scores)

            return scores['recall']

        hps, details, _ = optunity.maximize(svm_recall, num_evals=n_classify_evals, log_C=range_log_C)
        logger.info(f'Best log C: {hps["log_C"]}, average recall: {details.optimum}')

        return details.optimum

    hps, _, _ = optunity.maximize(
        svm_recall_svd, 
        num_evals=n_svd_evals, 
        n_pos=range_n_pos, 
        n_word=range_n_word, 
        n_char=range_n_char
    )
    logger.info(hps)

def train(train_data, lexicons, n_pos, n_word, n_char, log_C):
    pipeline = create_pipeline(n_pos, n_word, n_char, log_C, lexicon=lexicons)
    pipeline.fit(train_data, train_data['label'].values)

    return pipeline

def test(test_data, pipeline):
    y_pred = pipeline.predict(test_data)

    return evaluate(test_data['label'].values, y_pred)

def ablation(pipeline, train_data, lexicons, test_data, n_pos, n_word, n_char, log_C):
    ngram_transformers = {
        'pos': pipeline.named_steps.features.transformer_list[1], 
        'word': pipeline.named_steps.features.transformer_list[2], 
        'char': pipeline.named_steps.features.transformer_list[3], 
    }

    scores = {}

    for exclude in [
        'counts', 
        'lexicon', 
        *[f'lexicon_{lexicon}' for lexicon in lexicons.keys()], 
        'ngrams', 
        *[f'ngram_{transformer}' for transformer in ngram_transformers.keys()], 
    ]:        
        logger.info(f'Extracting features, excluding {exclude}...')

        feature_transformers = [
            ('custom_features', FunctionTransformer(partial(extract_features_dataframe, lexicon=lexicons, exclude=[exclude])))
        ]

        if exclude != 'ngrams':
            for key, transformer in ngram_transformers.items():
                if exclude != f'ngram_{key}':
                    feature_transformers.append(transformer)

        feature_pipeline = FeatureUnion(feature_transformers)
        features = feature_pipeline.transform(train_data)

        logger.info('Training...')
        classifier_pipeline = create_classifier_pipeline(log_C)
        classifier_pipeline.fit(features, train_data['label'].values)

        logger.info('Testing...')
        y_pred = classifier_pipeline.predict(feature_pipeline.transform(test_data))
        scores[exclude] = evaluate(test_data['label'].values, y_pred)

        logger.info('Done!')
        logger.debug(scores[exclude])

    return scores

########################
# Loading
########################

multi_quote_pattern = re.compile(r'\\?"{2,}')
single_quote_pattern = re.compile(r'(?<!")\\?"(?!")')
def load_lines(path, fix_quoting = False):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()

    if fix_quoting:
        lines = [multi_quote_pattern.sub('"', single_quote_pattern.sub('', line)) for line in lines]

    return lines


########################
# Classification
########################

class SentimentClassification:
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir

    ########################
    # Training
    ########################
    def load_lexicons(self, lexicons):
        self.lexicons = get_lexicons(**lexicons)

    def load_train_data(self, text_path, label_path):
        logger.info('Loading training data...')

        self.train_data = pd.DataFrame({
            'text': load_lines(text_path, fix_quoting=True),
            'label': [int(label) for label in load_lines(label_path)]
        })

        logger.info('Preprocessing training data...')

        preprocess(self.train_data, self.temp_dir)

        logger.info('Done!')

    def load_test_data(self, text_path, label_path):
        logger.info('Loading test data...')

        self.test_data = pd.DataFrame({
            'text': load_lines(text_path),
            'label': [int(label) for label in load_lines(label_path)]
        })

        logger.info('Preprocessing test data...')

        preprocess(self.test_data, self.temp_dir)

        logger.info('Done!')

    def train_cross_validation(self, **kwargs):
        if self.train_data is None:
            raise 'No training data loaded'
        if self.lexicons is None:
            raise 'No lexicons loaded'

        return train_cross_validation(self.train_data, lexicons=self.lexicons, **kwargs)


    def train_hyperparameter_optimalization(self, **kwargs):
        if self.train_data is None:
            raise 'No training data loaded'
        if self.lexicons is None:
            raise 'No lexicons loaded'

        return train_hyperparameter_optimalization(self.train_data, lexicons=self.lexicons, **kwargs)

    def train(self, **kwargs):
        if self.train_data is None:
            raise 'No training data loaded'
        if self.lexicons is None:
            raise 'No lexicons loaded'

        self.pipeline = train(self.train_data, lexicons=self.lexicons, **kwargs)

    def test(self):
        if self.test_data is None:
            raise 'No test data loaded'
        if self.pipeline is None:
            raise 'No pipeline trained'

        return test(self.test_data, self.pipeline)

    def ablation(self, **kwargs):
        if self.pipeline is None:
            raise 'No pipeline trained'
        if self.train_data is None:
            raise 'No training data loaded'
        if self.lexicons is None:
            raise 'No lexicons loaded'
        if self.test_data is None:
            raise 'No test data loaded'

        return ablation(self.pipeline, self.train_data, self.lexicons, self.test_data, **kwargs)

    def save_pipeline(self, file_path):
        if not self.pipeline:
            raise 'No pipeline trained'

        with open(file_path, 'wb') as file:
            pickle.dump(self.pipeline, file)

    ########################
    # Inference
    ########################
    def load_pipeline(self, file_path):
        with open(file_path, 'rb') as file:
            self.pipeline = pickle.load(file)

    def predict(self, texts):
        if  self.pipeline is None:
            raise 'No pipeline loaded'

        data = pd.DataFrame(list(texts), columns=['text'])

        logger.info('Preprocessing...')

        preprocess(data, self.temp_dir)

        logger.info('Classifying...')
        
        labels = self.pipeline.predict(data)

        logger.info('Done!')

        return labels