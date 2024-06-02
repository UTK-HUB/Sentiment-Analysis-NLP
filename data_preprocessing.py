import contractions
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer

# remove newlines
def remove_lines(data):
    clean_text = data.replace('\n',' ').replace('\\n',' ').replace('\t',' ')
    return clean_text

# Contraction mapping
def expand_text(data):
    exp = contractions.fix(data)
    return exp

# Handle accented char
def Handle_accented_chars(data):
    text = unidecode(data)
    return text

#Tokenize
def tokenizer(data):
    tokens = word_tokenize(data)
    return tokens

# Remove stopwords
stopwords_list = stopwords.words('english')
stopwords_list.remove('not')
stopwords_list.remove('no')
stopwords_list.remove('nor')
def remove_stopwords(data):
    filtered_text = [word.lower() for word in data if word.lower() not in stopwords_list]
    return filtered_text

# Clean data
def clean_data(data):
    cleaning = [word for word in data if word not in punctuation and len(word)>2 and word.isalpha()]
    return cleaning

# Lemmatization
def lemmmatize_data(data):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in data]
    return lemmatized_text

def joining_words(data):
    return ' '.join(data)