import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.util import ngrams

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()


def importfiles(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.read()
        
    return file, lines

def exportfiles(filename, varname):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(varname)
        
    return filed

def extract_nouns(txt_file):
    file, lines = importfiles(txt_file)
    is_noun = lambda pos: pos[:2] == 'NN'
    
    tokenized = nltk.word_tokenize(lines)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
                
    return nouns

def lemmatizing(word, pos=wordnet.NOUN):
    return lemmatizer.lemmatize(word, pos)

def lemmatizator(words):
    lemmatized_words = [lemmatizing(word, pos=wordnet.VERB) for word in words]
    return lemmatized_words

def lemmatizator_from_file(txt_file):
    nouns = extract_nouns(txt_file)
    return lemmatizator(nouns)

def toSentence(list_words):
    sentence = ' '.join(word for word in list_words)
    return sentence

def ngrammer(txt_file, n):
    nouns = extract_nouns(txt_file)
    n_grams = list(ngrams(nouns, n))
    
    ngramarr = [list(gram) for gram in n_grams]
    
    return ngramarr



if __name__ == "__main__":
    txt_target = "nlpwiki.txt"
    nouns = extract_nouns(txt_target)
    # print(nouns)
    
    # print(lemmatizator(nouns))
    
    # print(ngrammer(txt_target, 3))