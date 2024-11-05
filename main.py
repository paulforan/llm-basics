import nltk
from nltk.tokenize import word_tokenize

text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Language models are trained on large datasets.",
    "Python is a popular programming language."
]

# Tokenize the text
tokenized_data = [word_tokenize(sentence.lower()) for sentence in text_data]

print(tokenized_data)
