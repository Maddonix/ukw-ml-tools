from ..fieldnames import *

def get_test_data_ids(name, collection):
    return collection.find_one({TD_NAME: name})["ids"]



