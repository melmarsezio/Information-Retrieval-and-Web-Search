import spacy
from math import log
from itertools import combinations
from pprint import pprint

class InvertedIndex:
    def __init__(self):
        self.lambd = 0.4
        self.number_doc = 0
        self.tf_tokens = dict()
        self.tf_entities = dict()
        self.idf_tokens = dict()
        self.idf_entities = dict()

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        nlp = spacy.load("en_core_web_sm")
        self.number_doc = len(documents)

        for i in documents:
            doc = nlp(documents[i])

            entity_text = []
            for ent in doc.ents:
                entity_text.append(ent.text)

            token_text = []
            for token in doc:
                if not token.is_punct and not token.is_space and not token.is_stop:
                    token_text.append(token.text)

            for ent in entity_text:
                if ent in token_text:
                    token_text.remove(ent)
                    continue

            for ent in entity_text:
                if ent in self.tf_entities:
                    if i in self.tf_entities[ent]:
                        self.tf_entities[ent][i] += 1
                    else:
                        self.tf_entities[ent][i] = 1
                        self.idf_entities[ent] += 1
                else:
                    self.tf_entities[ent] = {i : 1}
                    self.idf_entities[ent] = 1

            for tok in token_text:
                if tok in self.tf_tokens:
                    if i in self.tf_tokens[tok]:
                        self.tf_tokens[tok][i] += 1
                    else:
                        self.tf_tokens[tok][i] = 1
                        self.idf_tokens[tok] += 1
                else:
                    self.tf_tokens[tok] = {i : 1}
                    self.idf_tokens[tok] = 1

    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        query_comb = []
        query_token = Q.split()
        for i in range(len(query_token) + 1):
            for comb in combinations(query_token, i):
                query_comb.append(list(comb))

        for i in range(len(query_comb)):
            query_comb[i] = ' '.join(query_comb[i])

        select_entity = []
        for entity in DoE:
            if entity in query_comb:
                select_entity.append(entity)

        all_possible_entity_subset = []
        for i in range(len(select_entity) + 1):
            all_possible_entity_subset.extend(list(combinations(select_entity, i)))

        filtered_subset = []
        print('this is all possible entity subset')
        print(all_possible_entity_subset)
        print()
        for entity_subset in all_possible_entity_subset:
            subset_list = ' '.join(entity_subset).split()
            wrong_subset = False
            for word in subset_list:
                if subset_list.count(word) > query_token.count(word):
                    wrong_subset = True
                    break
            if not wrong_subset:
                filtered_subset.append(list(entity_subset))

        retVal = []
        for subset in filtered_subset:
            Q_copy = Q.split()
            subset_list = ' '.join(subset).split()
            for word in subset_list:
                Q_copy.remove(word)
            retVal.append({'tokens' : Q_copy, 'entities' : subset})
        return retVal

    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        score_dict = dict()     # split id: score
        max_score = 0
        max_index = 0
        for i in range(len(query_splits)):
            token_score = 0
            for token in query_splits[i]['tokens']:
                if token in self.tf_tokens and doc_id in self.tf_tokens[token]:
                    tf_norm_token = 1 + log(1 + log(self.tf_tokens[token][doc_id]))
                    idf_token = 1 + log(self.number_doc / (1 + self.idf_tokens[token]))
                    token_score += tf_norm_token * idf_token
                else:
                    token_score += 0

            entity_score = 0
            for entity in query_splits[i]['entities']:
                if entity in self.tf_entities and doc_id in self.tf_entities[entity]:
                    tf_norm_entity = 1 + log(self.tf_entities[entity][doc_id])
                    idf_entities = 1 + log(self.number_doc / (1 + self.idf_entities[entity]))
                    entity_score += tf_norm_entity * idf_entities
                else:
                    entity_score += 0
            score_dict[i] = self.lambd * token_score + entity_score
            if score_dict[i] > max_score:
                max_score = score_dict[i]
                max_index = i
        return(max_score, query_splits[max_index])
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
