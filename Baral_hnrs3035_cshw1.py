import re


class Ngram:

    def __init__(self, n):
       self.n = n  

    def train(self, data):
        words = self.split_words(data)
        lowercase = [word.lower() for word in words]
        #converting all the words to lowercase to ensure uniqueness in words
        self.unique_words = set(lowercase)
        #Created an empty dictionary with all the unique words converted to lowercase
        self.unique_words_dictionary = {word: {} for word in self.unique_words}

        for i in range(len(lowercase) -1):
            current_word = lowercase[i]
            next_word = lowercase[i+1]
            if next_word not in self.unique_words_dictionary[current_word]:
                self.unique_words_dictionary[current_word][next_word] = 1
            else:
                self.unique_words_dictionary[current_word][next_word] += 1

        self.probabilites()

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




        







        


  
