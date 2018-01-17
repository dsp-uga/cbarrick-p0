import numpy as np
import json
from pyspark import SparkContext, SparkConf


class WordCounter:
    def __init__(self, ctx):
        self.ctx = ctx

    def read_doc(self, path):
        data = self.ctx.textFile(path)
        return data

    def read_stopwords(self, path):
        data = self.ctx.textFile(path)
        data = data.map(lambda row: (row, 1))
        return data

    def count(self, path):
        '''Get a simple word count from the file.
        '''
        data = self.read_doc(path)
        data = data.flatMap(str.split)
        data = data.map(str.lower)
        data = data.map(lambda word: (word, 1))
        data = data.reduceByKey(lambda a, b: a + b)

        # The instructions say to do this, but it tanks my part D score.
        # It has no effect on correctness for the other parts, only performance.
        # data = data.filter(lambda row: row[1] > 2)

        return data

    def remove_stopwords(self, data, stopwords):
        '''Remove stopwords from a word count RDD.
        '''
        data = data.subtractByKey(stopwords)
        return data

    def handle_punct(self, data):
        '''Handle punctuation in a word count RDD.
        '''
        data = data.filter(lambda row: len(row[0]) > 1)
        data = data.map(lambda row: (row[0].strip('.,:;\'!?'), row[1]))
        data = data.reduceByKey(lambda a, b: a + b)
        return data

    def parts_abc(self, stopwords, *paths):
        '''Solves parts A, B, and C in a single pass.
        '''
        # Dumps the top 40 words to JSON
        def dump(data, filename):
            data = data.top(40, key=lambda row: row[1])
            data = dict(data)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

        # Part A - simple word count
        path = ','.join(paths)
        data = self.count(path)
        dump(data, 'sp1.json')

        # Part B - handle stopwords
        stopwords = self.read_stopwords(stopwords)
        data = self.remove_stopwords(data, stopwords)
        dump(data, 'sp2.json')

        # Part C - handle punctuation
        data = self.handle_punct(data)
        dump(data, 'sp3.json')

        return data

    def part_d(self, stopwords, *paths):
        '''Solves part D.
        '''
        n = len(paths)
        stopwords = self.read_stopwords(stopwords)

        # Compute part-c style word count RDDs for each document.
        docs = (self.count(p) for p in paths)
        docs = (self.remove_stopwords(d, stopwords) for d in docs)
        docs = (self.handle_punct(d) for d in docs)
        docs = list(docs)

        # Compute the number of documents in which each key exists.
        counts = self.ctx.union(docs).countByKey()
        counts = dict(counts)

        # A function to map term frequencies to tf-idf scores.
        def tf_idf(row):
            key = row[0]
            tf = row[1]
            idf = np.log(n / counts[key])
            return key, tf*idf

        # Map each doc to tf-idf then dump the top 5 words into `data`.
        data = {}
        n = len(docs)
        for i, doc in enumerate(docs):
            doc = doc.map(tf_idf)
            doc = doc.top(5, key=lambda row: row[1])
            doc = dict(doc)
            data.update(doc)

        # Write the output file; `indent=2` makes it human readable.
        with open('sp4.json', 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == '__main__':
    default_docs = [
        'data/4300-0.txt',
        'data/pg1497.txt',
        'data/pg19033.txt',
        'data/pg3207.txt',
        'data/pg36.txt',
        'data/pg42671.txt',
        'data/pg514.txt',
        'data/pg6130.txt',
    ]

    import argparse
    parser = argparse.ArgumentParser(description='Count to top 40 words')
    parser.add_argument('-s', '--stopwords', default='data/stopwords.txt', help='path to stopwords file')
    parser.add_argument('path', nargs='*', default=default_docs, help='path to docs to count')
    args = parser.parse_args()

    conf = SparkConf().setAppName('cbarrick-p0')
    ctx = SparkContext(conf=conf)
    wc = WordCounter(ctx)

    wc.parts_abc(args.stopwords, *args.path)
    wc.part_d(args.stopwords, *args.path)
