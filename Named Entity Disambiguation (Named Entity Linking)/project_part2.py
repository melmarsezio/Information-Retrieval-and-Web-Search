import numpy as np
import xgboost as xgb
import spacy
from math import log
import re

class InvertedIndex:
    def __init__(self):
        self.number_doc = 0
        self.tf_tokens = dict()
        self.idf_tokens = dict()
        self.docs = dict()

    def index_documents(self, documents):
        nlp = spacy.load("en_core_web_sm")
        self.number_doc = len(documents)
        for i in documents:
            doc = nlp(documents[i])
            self.docs[i] = list()
            for tok in doc:
                if not tok.is_punct and not tok.is_space and not tok.is_stop:
                    word = tok.lemma_.lower()
                    self.docs[i].append(word)
                    if word in self.tf_tokens:
                        if i in self.tf_tokens[word]:
                            self.tf_tokens[word][i] += 1
                        else:
                            self.tf_tokens[word][i] = 1
                            self.idf_tokens[word] += 1
                    else:
                        self.tf_tokens[word] = {i : 1}
                        self.idf_tokens[word] = 1

    def score(self, token_list, doc_title):
        token_score = 0
        token_list = get_sub_token_list(token_list, 25)
        for token in token_list:
            lemma = token.lower()
            if lemma in self.tf_tokens and doc_title in self.tf_tokens[lemma]:
                tf_norm_token = 1 + log(1 + log(self.tf_tokens[lemma][doc_title]))
                idf_token = 1 + log(self.number_doc / (1 + self.idf_tokens[lemma]))
                token_score += tf_norm_token * idf_token
        return token_score

def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    doc_index = InvertedIndex()
    doc_index.index_documents(men_docs)

    train_data, train_group, train_label = generate_features_group_labels(doc_index, train_mentions, parsed_entity_pages, train_labels)
    xgboost_train = transform_data(train_data, train_group, train_label)

    ## Parameters for XGBoost, you can fine-tune these parameters according to your settings...
    param = {'max_depth': 7, 'eta': 0.05, 'silent': 1, 'objective': 'rank:pairwise', 'min_child_weight': 0.01, 'lambda':100, 'subsample': 0.5}
    classifier = xgb.train(param, xgboost_train, num_boost_round = 2500)

    test_data, test_group = generate_features_group_labels(doc_index, dev_mentions, parsed_entity_pages)
    xgboost_test = transform_data(test_data, test_group)
    preds = classifier.predict(xgboost_test)

    possibility = []
    index = 0
    for cand_number in test_group:
        possibility.append(preds[index:index+cand_number])
        index += cand_number
    retVal = dict()
    for i in range(1, len(possibility)+1):
        mention_index = possibility[i-1].argmax()
        retVal[i] = dev_mentions[i]['candidate_entities'][mention_index]
    return retVal

def generate_features_group_labels(object, mentions, parsed_list, labels = None):
    data = []
    group = []
    label = []
    for mention_id in mentions:
        candidate_list = mentions[mention_id]['candidate_entities']
        doc_title = mentions[mention_id]['doc_title']
        group.append(len(candidate_list))
        mention = mentions[mention_id]['mention']
        mention_length = mentions[mention_id]['length']

        for cand in candidate_list:
            cand_length = len(cand)
            word_similarity = compute_mention_candidate_similarity(mention, cand)
            wiki_token_score = object.score(parsed_list[cand], doc_title)
            length_diff = abs(mention_length - cand_length)
            missing_word_percentage = compute_missing_word_percentage(object.docs[doc_title], parsed_list[cand])
            data.append([wiki_token_score, missing_word_percentage, length_diff/mention_length, word_similarity])
            if labels != None:
                if cand == labels[mention_id]['label']:
                    label.append(1)
                else:
                    label.append(0)
    if labels == None:
        return data, group
    else:
        return data, group, label

def get_sub_token_list(token_list, num_of_elem):
    sublist = []
    num = 0
    i = 0
    while num < num_of_elem and i < len(token_list):
        if token_list[i][3] != 'DET' and token_list[i][3] != 'ADV' and token_list[i][3] != 'ADJ':
            num += 1
            sublist.append(token_list[i][2])
        i += 1
    return sublist

def transform_data(features, groups, labels=None):
    xgb_data = xgb.DMatrix(data=features, label=labels)
    xgb_data.set_group(groups)
    return xgb_data

def compute_mention_candidate_similarity(mention, cand):
    mention = re.sub('[^0-9a-zA-Z]+', ' ', mention)
    cand = re.sub('[^0-9a-zA-Z]+', ' ', cand)
    cur = 0
    match = 0
    for letter in cand:
        if letter in mention[cur:]:
            match += 1
            cur = mention.index(letter) + 1
    return match/len(cand)

def compute_missing_word_percentage(doc, parsed_list):
    match = 0
    parsed_list = [i[2].lower() for i in parsed_list]
    for word in doc:
        if word not in parsed_list:
            match += 1
    return match/len(doc)
