import re
import random

class Ngram:

    def __init__(self, n):
       self.n = n
       if n not in [1, 2]:
           raise ValueError("Can only support unigram and bigram.")

    
    def split_words(self, texts):
        regex = r"\w+|[^\w\s]"
        splitted_words = re.findall(regex, texts)
        return splitted_words
    
    def probabilites(self):
        self.probabilites_dictionary = {}
        for word in self.unique_words_dictionary:
            self.probabilites_dictionary[word] = {}
            total_count = sum(self.unique_words_dictionary[word].values())
            for next_word, count in self.unique_words_dictionary[word].items():
                probability = count/total_count
                self.probabilites_dictionary[word][next_word] = probability
        return self.probabilites_dictionary
    
    def train(self, data):
        words = self.split_words(data)
        lowercase = [word.lower() for word in words]
        #converting all the words to lowercase to ensure uniqueness in words
        self.unique_words = set(lowercase)
        self.unique_words_dictionary = {}

      
        if (self.n == 1):
            self.unique_words_dictionary = {word: {} for word in self.unique_words}
            for i in range(len(lowercase) -1):
                current_word = lowercase[i]
                next_word = lowercase[i+1]
                if next_word not in self.unique_words_dictionary[current_word]:
                    self.unique_words_dictionary[current_word][next_word] = 1
                else:
                    self.unique_words_dictionary[current_word][next_word] += 1
        
        if (self.n == 2):
            for i in range(len(lowercase) - 2):
                current_words = (lowercase[i], lowercase[i+1])
                next_word = lowercase[i+2]
                if current_words not in self.unique_words_dictionary:
                    self.unique_words_dictionary[current_words] = {}
                if next_word not in self.unique_words_dictionary[current_words]:
                    self.unique_words_dictionary[current_words][next_word] = 1
                else:
                    self.unique_words_dictionary[current_words][next_word] += 1   



        self.probabilites()

    def predict_next_word(self, input : tuple, determenistic : bool = False):  
        if (self.n == 1):
            if(len(input) != 1):
                print("Error: Unigram model, there should only be one input.")
                return 
            current_word = input[0].lower()
        elif (self.n == 2):
                if(len(input) != 2):
                    print("Error: Bigram model, there should be two inputs")
                    return 
                current_word = (input[0].lower(), input[1].lower())
        else:
            raise ValueError("Can only support unigram and bigram.")
        
        if current_word not in self.probabilites_dictionary:
            print(f"Error: The word'{current_word}' is not in the dictionary")
            return

        probs = self.probabilites_dictionary[current_word]

        if determenistic:
            next_word = max(probs, key= probs.get)
            return next_word
        else: 
            words = list(probs.keys())
            probabilties = list(probs.values())
            rand = random.choices(words, weights=probabilties)[0]
            return rand
