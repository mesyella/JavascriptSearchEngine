# Javascript Snippet Code Search Engine
Search engine for javascript snippet code. Information Retrieval implementation using TF-IDF and cosine-similarity.

## Database
The data used for this search engine is from 30 seconds of code which consist of 474 javascript snippet code.

## Method
1. Preprocessing text
Preprocessing the data by lowering, remove punctuation and number, and remove stopwords(from the TFIDFVectorizer).
2. TF-IDF
Calculate the Term frequency–inverse document frequency from the data and the input.
3. Cosine Similarity
Find the distance of the input and the data by using cosine similarity and show the shortest distance(nearest).

## Programming Language
The programming language used for this search engine is python. Using some library such as re, string, numpy, and scikit learn.




