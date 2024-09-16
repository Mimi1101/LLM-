import re
import random
import argparse
import collections
from typing import Union
import requests
import pickle

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
        lala = self.unique_words
       #self.n = 2 is not iterating thorugh unique words or i guess it is and i am tripping
      
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

        """
        if (self.n == 2):
    processed_pairs = set()  # Track unique pairs
    for i in range(len(lowercase) - 2):
        current_words = (lowercase[i], lowercase[i+1])
        next_word = lowercase[i+2]

        if current_words in processed_pairs:
            continue  # Skip if pair already processed

        processed_pairs.add(current_words)  # Mark this pair as processed

        if current_words not in self.unique_words_dictionary:
            self.unique_words_dictionary[current_words] = {}

        if next_word not in self.unique_words_dictionary[current_words]:
            self.unique_words_dictionary[current_words][next_word] = 1
        else:
            self.unique_words_dictionary[current_words][next_word] += 1

        """



        self.probabilites()

    def predict_next_word(self, input: tuple, deterministic: bool = False):

        if self.n == 1:
            current_word = input[0]
            
            if current_word not in self.probabilites_dictionary:
                print (" The word is not in the vocabulary")
            probabilities = self.probabilites_dictionary[current_wordword]
                






class BPE:
    """
    count = 1
    self.mydictionary = new dictionary

    Every time adding value to set,
    if set.doesNotContain(value)
        self.mydictionary[count] = value
        count++
    """


    def __init__(self):
        self.vocabulary = {}

    def train_bpe(self, data, k : int = 500):
        tokens = self.split_data_BPE(data)
    # Initialize vocabulary with individual tokens
        self.vocabulary = set(tokens)
        
        for _   in range(k):
            # Create pairs from consecutive tokens
            double_pairs = list(zip(tokens, tokens[1:]))
            pair_counts = collections.Counter(double_pairs)
            if not pair_counts:
                break
            most_frequent = max(pair_counts, key=pair_counts.get)
            new_token = ''.join(most_frequent)          
            # Update the vocabulary
            self.vocabulary.add(new_token)
            new_merge_tokens = []
            skipping = False
            for i in range(len(tokens) - 1):
                if skipping:
                    skipping = False
                    continue
                if(tokens[i], tokens[i+1] )== most_frequent:
                    new_merge_tokens.append(new_token)
                    skipping = True
                else:
                    new_merge_tokens.append(tokens[i])
            if not skipping:
                new_merge_tokens.append(tokens[-1])
            tokens = new_merge_tokens
        print(self.vocabulary)
        return self.vocabulary


    def tokenize(self, text):
        tokens = self.split_data_BPE(text)
        token_results = []
        token_ids = []
    
        counter = 0
        while counter < len(tokens):
            current_token = tokens[counter]
            for j in range(len(tokens), counter, -1):
                segment = ''.join(tokens[counter:j])
                if segment in self.vocabulary:
                    current_token = segment
                    counter = j - 1
                    break
            token_results.append(current_token)
            token_ids.append(list(self.vocabulary).index(current_token))
            counter += 1

        return token_results, token_ids
      

    def split_data_BPE(self, text):
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        processed_tokens = []
        for token in tokens:
            if token.isalnum():
                # Split the alphanumeric token into characters and append '</w>' at the end
                processed_tokens.extend(list(token) + ['</w>'])
            else: 
                # For punctuation, just append the token
                processed_tokens.append(token)
        #print(processed_tokens)
        return processed_tokens
    
# def train_ngram(data):
#     with open(r)   

def save_model(model: Union[Ngram, BPE], path: str) -> None:
    """Save the model to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str) -> Union[Ngram, BPE]:
    """Load the model from a file using pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def train_ngram(args):
     model = Ngram(args.n)  # Pass the 'n' argument here
     try:
            with open(args.data, 'r', encoding='utf-8') as f:  # Specify utf-8 encoding
                corpus = f.read()
            model.train(corpus)
            save_model(model, args.save)
     except UnicodeDecodeError:
            print("Error: Unable to read the file. It may be encoded in a different format.")
            print("Try opening the file in a text editor and saving it with UTF-8 encoding.")
     except IOError:
            print(f"Error: Unable to read the file at {args.data}. Please check if the file exists and you have permission to read it.")


def predict_ngram(args):
    """Load an N-gram model and use it for prediction."""
    model = load_model(args.load)
    predictions = model.predict_next_word(args.word, False)
    print(predictions)

def train_bpe(args):
    """Train a BPE tokenizer and save it."""
    model = BPE()
    with open(args.data, 'r') as f:
        corpus = f.read()
    model.train(corpus)
    save_model(model, args.save)

def tokenize(args):
    """Load a BPE tokenizer and use it for tokenization."""
    model = load_model(args.load)
    tokens = model.tokenize(args.text)
    print(' '.join(tokens))

def main():
    parser = argparse.ArgumentParser(description="NLP tools for N-gram models and BPE tokenization")
    parser.add_argument("activity", choices=["train_ngram", "predict_ngram", "train_bpe", "tokenize"],
                        help="Select which activity to perform")
    parser.add_argument("--data", help="Path to the training data corpus")
    parser.add_argument("--save", help="Path to save the trained model")
    parser.add_argument("--load", help="Path to load the trained model")
    parser.add_argument("--word", help="Initial word(s) for prediction")
    parser.add_argument("--nwords", type=int, help="Number of words to predict")
    parser.add_argument("--text", help="Text to tokenize")
    parser.add_argument("--n", type=int, choices=[1, 2], help="Order of the N-gram model")
    parser.add_argument("--d", action="store_true", help="Set deterministic flag for prediction")
    args = parser.parse_args()
        

    if args.activity == "train_ngram":
        if not args.data or not args.save or args.n is None:
            parser.error("train_ngram requires --data, --save, and --n arguments")
        train_ngram(args)
    elif args.activity == "predict_ngram":
        if not args.load or not args.word:
            parser.error("predict_ngram requires --load and --word arguments")
        predict_ngram(args)
    elif args.activity == "train_bpe":
        if not args.data or not args.save:
            parser.error("train_bpe requires --data and --save arguments")
        train_bpe(args)
    elif args.activity == "tokenize":
        if not args.load or not args.text:
            parser.error("tokenize requires --load and --text arguments")
        tokenize(args)

if __name__ == "__main__":

    # url = "https://www.gutenberg.org/files/2701/2701-0.txt"
    # response = requests.get(url)
    # if response.status_code == 200:
    # # Save the content to a .txt file
    #     with open("Moby_Dick.txt", "w", encoding='utf-8') as file:
    #         file.write(response.text)
    #     print("Moby Dick has been successfully saved as 'Moby_Dick.txt'")
    # else:
    #     print("Failed to retrieve the text. Status code:", response.status_code)

    main()
    

    