import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')



# import these modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()



#Wortschatz eingeben und vorbereiten
wortschatz = ["hello", "here", "how", "are", "cool", "organization", "you"]


stemmed_wortschatz = []

for w in wortschatz:
     stemmed_wortschatz.append(ps.stem(w))

einzigartig = set(stemmed_wortschatz)

einzigartig_stemmed_wortschatz = list(einzigartig)


#Eingabesatz
satz = "hello how are you"

tokenized_satz = nltk.word_tokenize(satz)

stemmed_satz = []

for s in tokenized_satz:
     stemmed_satz.append(ps.stem(s))

einzigartig = set(stemmed_satz)
einzigartig_stemmed_satz = list(einzigartig)


#bag of word erzeugen
bag_of_words = []
for w in einzigartig_stemmed_wortschatz:
    if w in einzigartig_stemmed_satz:
          bag_of_words.append(1)
    else:
          bag_of_words.append(0)


print('Wortschatz:', einzigartig_stemmed_wortschatz)
print('Satz:', satz)
print('Bag_of_words:', bag_of_words)


#I. Training==================================================
#1. Take a list of sentences with tags
#2. Tokenize the sentences
#3. Stem the words in the sentences
#4. Turn the sentences into bags of words
#5. Each bag of words goes with a label
#6. All bags of words together is the training data
#II. Chatting==================================================
#1. Get a sentence
#2. Tokenize the sentence
#3. Stem the sentence
#4. Make a bag of words
#5. Classify bag of words
#6. Choose random from response list


#EXTRACTIVE TEXT SUMMARY ---------------------------
# 
#                    1. Tokenize the text -> List of tokens
#                    
#                    
#                    2. Get rid of punctuation and stop words
#                    
#                    
#                    3. Count the number of times a word is used
#                    
#                    
#                    4. Normalize the count with the highest count appearing (optional)
#                    
#                    
#                    5. Take the text and for each sentence calculate
#                       the normalized count 
#                       
#                       
#                    6. Extract a percentage of the highest ranked sentences:
#                       These function as the summary of the text
                       