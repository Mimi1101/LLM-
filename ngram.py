import argparse
import random
import re
import pickle
from typing import Union 
from collections import Counter

class Ngram:


    def __init__(self, n):
        self.n = n
        if n not in [1, 2]:
            raise ValueError("Can only support unigram and bigram.")
        self.unique_words_dictionary = {}
        self.probabilities_dictionary = {}
        
    
    def split_words(self, texts):
        regex = r"\w+|[^\w\s]"
        splitted_words = re.findall(regex, texts)
        print(splitted_words)
        return splitted_words
    
    def train(self, data):
        data = self.split_words(data)
        word_counts = Counter(data)
        
        if self.n == 1:
            self.unique_words_dictionary = dict(word_counts)
            total_tokens = len(data)
            for word, count in word_counts.items():
                self.probabilities_dictionary[word] = count/total_tokens
        elif self.n == 2:
            bigrams = [(data[i], data[i + 1]) for i in range(len(data) - 1)]
            print("Bigrams created:", bigrams) 
            bigram_counts = Counter(bigrams)
            self.unique_words_dictionary = dict(bigram_counts)
            print("Unique Bigrams:", self.unique_words_dictionary)  
            total_bigrams = len(bigrams)
            for bigram, count in bigram_counts.items():
                self.probabilities_dictionary[bigram] = count / total_bigrams
            
            
    def predict_next_word(self, input: tuple, deterministic: bool = False):
        if self.n == 1:
            word1 = input[0]
            if word1 not in self.probabilities_dictionary:
                print("Error: The word does not exist in the vocabulary.")
                return

            words = list(self.probabilities_dictionary.keys())
            probabilities = list(self.probabilities_dictionary.values())
            if deterministic:
                max_prob_index = probabilities.index(max(probabilities))
                next_word = words[max_prob_index]
                return next_word
            else:
                next_word = random.choices(words, weights=probabilities, k=1)
                return next_word[0]

        elif self.n == 2:
            words = []
            probabilities = []

            for bigram, prob in self.probabilities_dictionary.items():
                #print("Checking bigram:", bigram) 
                if bigram[0] == input[1]:  # Check if the first word of the bigram matches the second word of the input
                    words.append(bigram[1])
                    probabilities.append(prob)
            # Check if words and probabilities are empty
            if not words or not probabilities:
                print("Error: No valid next words found for the given bigram.")
                return None

            if deterministic:
                max_prob_index = probabilities.index(max(probabilities))
                next_word = words[max_prob_index]
                return next_word
            else:
                next_word = random.choices(words, weights=probabilities, k=1)
                return next_word[0]


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
    
        # Load the model from the specified path
        with open(args.load, 'rb') as f:
            model = pickle.load(f)

        n = model.n  # The 'n' in n-gram

        # Prepare input words
        if isinstance(args.word, str):
            # If args.word is a single string, split it into words
            input_words = args.word.strip().lower().split()
        else:
            # Assume args.word is a list or tuple of words
            input_words = [word.lower() for word in args.word]

        if len(input_words) != n:
            print(f"Error: For {n}-gram, provide exactly {n} word(s).")
            return

        input_words = tuple(input_words)  # Convert to tuple

        predictions = []
        for _ in range(args.nwords):
            next_word = model.predict_next_word(input_words, deterministic=args.d)
            if next_word:
                predictions.append(next_word)
                # Update input for the next prediction (shift window)
                input_words = (*input_words[1:], next_word) if n > 1 else (next_word,)
            else:
                break  # No next word predicted

        print("Predicted words:", ' '.join(predictions))

    

   
    
    



def save_model(model: Union[Ngram], path: str) -> None:
    """Save the model to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def main():
    parser = argparse.ArgumentParser(description="NLP tools for N-gram models and BPE tokenization")
    parser.add_argument("activity", choices=["train_ngram", "predict_ngram", "train_bpe", "tokenize", "predict_bigram"],
                        help="Select which activity to perform")
    parser.add_argument("--data", help="Path to the training data corpus")
    parser.add_argument("--save", help="Path to save the trained model")
    parser.add_argument("--load", help="Path to load the trained model")
    parser.add_argument("--word", nargs='+', help="Initial word(s) for prediction")
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
    
        
if __name__ == "__main__":
    main()
