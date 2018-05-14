# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:16:29 2018

@author: pmfin
"""
import io

from os import path

import fasttext

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return 

class FastText(object):
    def __init__(self, fasttext_lib_directory, fasttext_model_path):
        cmds = [fasttext_lib_directory, 'print-word-vectors', fasttext_model_path]  # Add '-' in the end for interactive mode, yet it didn't work for me...
        self.model = subprocess.Popen(cmds, stdout=subprocess.PIPE, stdin=subprocess.PIPE, env=os.environ.copy())

        # Test the model
        print('\nTesting the model...\nPrediction for apple: ')
        item = 'apple\n'
        item = item.encode('utf-8')
        self.model.stdin.write(item)
        result = self.model.stdout.readline()
        result = result[len(item):]
        result = np.fromstring(result, dtype=np.float32, sep=' ')
        self.vector_size = len(result)
        print('Length of word-vector is:', self.vector_size)

    def __getitem__(self, item):
        assert type(item) is str
        initial_item = item
        item = item.lower().replace('/', '').replace('-', '').replace('\\', '').replace('`', '')
        if len(item) == 0 or ' ' in item:
            raise KeyError('Could not process: ' + initial_item)

        if not item.endswith('\n'):
            item += '\n'

        item = item.encode('utf-8')
        self.model.stdin.write(item)
        self.model.stdout.flush()
        result = self.model.stdout.readline()  # Read result
        result = result[len(item):]            # Take everything but the initial item
        result = np.fromstring(result, dtype=np.float32, sep=' ')

        if len(result) != self.vector_size:
            print('Could not process: ' + item)
            raise KeyError('Could not process: ' + initial_item)
        return result


base_path = path.dirname(__file__)
file_name = "crawl-300d-2M.vec"
file_path = path.abspath(path.join(base_path, "..", "..", "..", "FastText\crawl-300d-2M.vec", file_name))

model = FastText(FastText(fasttext_lib_directory='./fastText/fasttext', fasttext_model_path='./wiki.en.bin')
print(model['machine'])

#data = load_vectors(file_path)
#print("Loaded Vectors!")
#
#print(data["mines"])
#print(data["Mines"])
#print(data["aejgejgnjawifn"])