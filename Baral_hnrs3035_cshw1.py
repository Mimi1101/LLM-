import re


class Ngram:

    def __init__(self, n):
       self.n = n  

    def train(self, data):
        words = self.split_words(data)

    def split_words(self, texts):
        regex = r"\w+|[^\w\s]"
        splitted_words = re.findall(regex, texts)



        


  
