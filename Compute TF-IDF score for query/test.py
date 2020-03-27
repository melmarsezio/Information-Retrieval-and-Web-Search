import pickle
import project_part1 as project_part1
#import icream_project_part1 as project_part1
from pprint import pprint

#fname = './Data/sample_documents.pickle'
#documents = pickle.load(open(fname,"rb"))
# documents = {1: 'According to Los Angeles Times, The Boston Globe will be experiencing another recession in 2020. However, The Boston Globe decales it a hoax.',
#              2: 'The Washington Post declines the shares of George Washington.',
#              3: 'According to Los Angeles Times, the UNSW COMP6714 students should be able to finish project part-1 now.'}

documents = pickle.load(open('test_500docs.pickle',"rb"))
pprint(documents)

index = project_part1.InvertedIndex()
index.index_documents(documents)


## Test cases
pprint(index.tf_tokens)
pprint(index.tf_entities)
pprint(index.idf_tokens)
pprint(index.idf_entities)

Q = 'Los The Angeles Boston Times Globe Washington Post'
DoE = {'Los Angeles Times':0, 'The Boston Globe':1,'The Washington Post':2, 'Star Tribune':3}


#Q = 'New York Times Trump travel'
#DoE = {'New York Times':0, 'New York':1,'New York City':2}#, 'Trump':3}

doc_id = 1

## 2. Split the query...
query_splits = index.split_query(Q, DoE)
# print('Possible query splits:\n')
# for split in query_splits:
#     print(split)

## 3. Compute the max-score...
# print('Score for each query split:\n')
result = index.max_score_query(query_splits, doc_id)

print('The maximum score:')
print(result)
