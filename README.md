cbarrick-p0
==================================================
Word counting in Spark

This project is a starter tutorial on NLP with Spark. It builds word counts from a 8 classic literature books taken from Project Gutenberg.

1. The first part builds a simple word count.
2. The second part removes stopwords.
3. The third part handles punctuation.
4. And the last part applies TF-IDF.

The full instructions for this assignment can be found in `project0.pdf`.


Usage
--------------------------------------------------
You can run `main.py` as a regular python script. It outputs files `sp{1,2,3,4}.json` for each of the four parts of the assignment.

You can pass in a list of files to read and the stopwords file. By default, it finds the standard files in the data directory.

```
usage: main.py [-h] [-s STOPWORDS] [path [path ...]]

Count to top 40 words

positional arguments:
  path                  path to docs to count

optional arguments:
  -h, --help            show this help message and exit
  -s STOPWORDS, --stopwords STOPWORDS
                        path to stopwords file
```


Comments
--------------------------------------------------
My score for part C is abysmal, but my score for part B is perfect. As far as I can tell, the difference in parts C and B should only be three lines: after removing stopwords, part C should:

1. Filter out words with length 1.
2. Strip periods (.), commas (,), colons (:), semicolons (;), apostrophes ('), exclamation marks (!), and question marks (?) from the left and right sides of tokens.
3. Re-reduce to combine the newly formed keys with the existing ones.

I can't figure out why I don't get 100% on parts A, B, and C at least. The core functions to do these parts are all straight-forward and I cannot find a correctness issue. My code is very straight forward:

```python
sc = pyspark.SparkContext()

def count(path):
	'''Get a simple word count from the file.
	'''
	data = sc.textFile(path)
	data = data.flatMap(str.split)
	data = data.map(str.lower)
	data = data.map(lambda word: (word, 1))
	data = data.reduceByKey(lambda a, b: a + b)
	data = data.filter(lambda row: row[1] > 2)
	return data

def remove_stopwords(data, stopwords):
	'''Remove stopwords from a word count RDD.
	'''
	data = data.subtractByKey(stopwords)
	return data

def handle_punct(data):
	'''Handle punctuation in a word count RDD.
	'''
	data = data.filter(lambda row: len(row[0]) > 1)
	data = data.map(lambda row: (row[0].strip('.,:;\'!?'), row[1]))
	data = data.reduceByKey(lambda a, b: a + b)
	return data
```

I'm frustrated that my scores are so low for such a simple problem. These solutions take less than 100 lines of code total, and I'm tired of pouring over the same 100 lines looking for correctness issues.
