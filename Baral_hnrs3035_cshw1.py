import argparse
from itertools import pairwise
import random
import re
import pickle
from collections import Counter, OrderedDict
import tqdm

"""
This class is for predicitng next tokens using unigram and bigram
"""
class Ngram:

    #Initializing n, unique words and probbailties
    def __init__(self, n):
        self.n = n
        if n not in [1, 2]:
            raise ValueError("Can only support unigram and bigram.")
        self.unique_words_dictionary = {}
        self.probabilities_dictionary = {}
        
    # Splitting words icluding punctuations
    def split_words(self, texts):
        regex = r"\w+|[^\w\s]"
        splitted_words = re.findall(regex, texts)
        print(splitted_words)
        return splitted_words
    

    def train(self, data):
        """
        Train the n-gram model (unigram or bigram) on the provided data.

        """
        #Split the data
        data = self.split_words(data)
        #Counting the freuency for every word
        word_counts = Counter(data)
        
        #Unigram Model Logic
        if self.n == 1:
            #add unique values to the unique words dictionary along with their frequency
            self.unique_words_dictionary = dict(word_counts)
            total_tokens = len(data)
            #counting the probbaility for each word
            for word, count in word_counts.items():
                self.probabilities_dictionary[word] = count/total_tokens
        #Bigram logic
        elif self.n == 2:
            # creating pairs of adjacent words
            bigrams = [(data[i], data[i + 1]) for i in range(len(data) - 1)]
            bigram_counts = Counter(bigrams)
            # add unique pairs and their frequency to the dictionary
            self.unique_words_dictionary = dict(bigram_counts)
            # print("Unique Bigrams:", self.unique_words_dictionary)  
            total_bigrams = len(bigrams)
            # count the probability
            for bigram, count in bigram_counts.items():
                self.probabilities_dictionary[bigram] = count / total_bigrams
            
            
    def predict_next_word(self, input: tuple, deterministic: bool = False):
        """
        Predict the next word based on the trained model.
        Returns:
        - The predicted next word or None if no valid prediction is found.
        """
        #If its a unigram model
        if self.n == 1:
            word1 = input[0] # the first word in  input
            if word1 not in self.probabilities_dictionary: # error check
                print("Error: The word does not exist in the vocabulary.")
                return
            # list of all words and their probbailties
            words = list(self.probabilities_dictionary.keys())
            probabilities = list(self.probabilities_dictionary.values())

            # if true select the word with highest probs
            if deterministic:
                max_prob_index = probabilities.index(max(probabilities))
                next_word = words[max_prob_index]
                return next_word
            #otherwise select randomly based on probs
            else:
                next_word = random.choices(words, weights=probabilities, k=1)
                return next_word[0]
        # bigram logic
        elif self.n == 2:
            words = [] # storing next words
            probabilities = [] # probs

            for bigram, prob in self.probabilities_dictionary.items():
                #print("Checking bigram:", bigram) 
                if bigram[0] == input[1]:  # Check if the first word of the bigram matches the second word of the input
                    words.append(bigram[1]) # Add the second word of the bigram to the possible next words
                    probabilities.append(prob) # Add the probability of the bigram to the list

            # Check if words and probabilities are empty
            if not words or not probabilities:
                print("Error: No valid next words found for the given bigram.")
                return None
            # deterministic is true so returning the word with highest probs
            if deterministic:
                max_prob_index = probabilities.index(max(probabilities))
                next_word = words[max_prob_index]
                return next_word
            #otherwise select randomly based on probs
            else:
                next_word = random.choices(words, weights=probabilities, k=1)
                return next_word[0]




class BPE:
    """
    A class to implement Byte Pair Encoding (BPE) for text tokenization.
    """
    # initialzing the voabulary
    def __init__(self):
        self.vocabulary = OrderedDict()

    #splitting the texts into characters alsong with spaces
    def split_characters(self, text):
        return list(text)
    
    #Replace occurrences of a specific character pair with a merged version.
    def replace_pair(self, characters, pair_to_replace):
        new_characters = []
        i = 0
        while i < len(characters):
            if i < len(characters) - 1 and characters[i] + characters[i+1] == pair_to_replace:
                new_characters.append(pair_to_replace)
                i += 2
            else:
                new_characters.append(characters[i])
                i += 1
        return new_characters

    def train(self, corpus, k: int = 500):
        #Spliiitng the words into characters
        characters = self.split_characters(corpus)
        self.all_merges = []

        # Initialize vocabulary with individual unique characters 
        for char in characters:
            if char not in self.vocabulary:
                self.vocabulary[char] = len(self.vocabulary)
        
        # For k iterations along with a plot to keep track of progress
        for _ in tqdm.tqdm(range(k), desc="Training BPE", unit="iter"):

            # Create pairs of adjacent characters
            pairs = [''.join(pair) for pair in pairwise(characters)]

            # Count frequency of each pair
            pair_frequency = Counter(pairs)

            if not pair_frequency:
                break  # Exit if there are no pairs left to process


            # Find all pairs with the maximum frequency
            most_frequent_pair = pair_frequency.most_common(1)
            if not most_frequent_pair:
                break
            common_pair, freq = most_frequent_pair[0]

            # Append the most frequent pair to the vocabulary
            if common_pair not in self.vocabulary:
                self.vocabulary[common_pair] = len(self.vocabulary)
            self.all_merges.append(common_pair)

            # Replace occurrences of the most frequent pairs in the corpus
            characters = self.replace_pair(characters, common_pair)

            # print(f"Iteration {_+1}:")
            # print(f"Updated Characters: {characters}")
            # print(f"Updated Vocabulary: {self.vocabulary}")
    
    def tokenize(self, corpus):
        # splitting the corpus into characters
        characters = self.split_characters(corpus)

        # Apply merges in the same order as they were learned during training
        for merge in self.all_merges:
            characters = self.replace_pair(characters, merge)

        # Map tokens to token IDs using the pre-trained vocabulary
        token_ids = [self.vocabulary[token] for token in characters]

        #print(f"Updated Characters: {characters}")
        return characters, token_ids


#saving the model after training
def save_model(model: BPE, path: str) -> None:
    """Save the model to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
       
def train_ngram(args):
     """
     training n_gram for argparse
     """
     model = Ngram(args.n)  
     try:
            with open(args.data, 'r', encoding='utf-8') as f:  
                corpus = f.read()
            model.train(corpus)
            save_model(model, args.save)
     except IOError:
            print(f"Error: Unable to read the file at {args.data}. Please check if the file exists and you have permission to read it.")

def predict_ngram(args):
        """
        Predicting the next word 
        """
        # Load the model from the specified path
        with open(args.load, 'rb') as f:
            model = pickle.load(f)

        n = model.n  # The 'n' in n-gram

        # Prepare input words
        if isinstance(args.word, str):
            input_words = args.word.strip().lower().split()
        else:
            input_words = [word.lower() for word in args.word]


        input_words = tuple(input_words)  # Convert to tuple

        predictions = []
        for _ in range(args.nwords):
            # Predict the next word based on the current input
            next_word = model.predict_next_word(input_words, deterministic=args.d)
            if next_word:
                predictions.append(next_word)
                # For bigrams, shift the tuple by one word. For unigrams, use the new word.
                input_words = (*input_words[1:], next_word) if n > 1 else (next_word,)
            else:
                break  # No next word predicted

        print("Predicted words:", ' '.join(predictions))

def train_bpe(args):
    """Train a BPE tokenizer and save it."""
    model = BPE()
    try:
            with open(args.data, 'r', encoding='utf-8') as f:  # Specify utf-8 encoding
                corpus = f.read()
            model.train(corpus, k= args.k)
            save_model(model, args.save)
            print(f"BPE model trained and saved to {args.save}")
    except IOError:
            print(f"Error: Unable to read the file at {args.data}. Please check if the file exists and you have permission to read it.")
   

def tokenize(args):
    """Tokenize the given text using a loaded BPE model."""
    # Load the BPE model
    with open(args.load, 'rb') as f:
        model = pickle.load(f)
    
    # Tokenize the text
    tokens, token_ids = model.tokenize(args.text)
    
    # Print the results
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    print("\nTokens with their IDs:")
    for token, token_id in zip(tokens, token_ids):
        print(f"Token: {token}, ID: {token_id}")

def main():
    parser = argparse.ArgumentParser(description="NLP tools for N-gram models and BPE tokenization")
    parser.add_argument("activity", choices=["train_ngram", "predict_ngram", "train_bpe", "tokenize"],
                        help="Select which activity to perform")
    parser.add_argument("--data", help="Path to the training data corpus")
    parser.add_argument("--save", help="Path to save the trained model")
    parser.add_argument("--load", help="Path to load the trained model")
    parser.add_argument("--word", help="Initial word(s) for prediction")
    parser.add_argument("--nwords", type=int, help="Number of words to predict")
    parser.add_argument("--text", nargs='+', help="Text to tokenize")
    parser.add_argument("--n", type=int, choices=[1, 2], help="Unigram or bigram")
    parser.add_argument("--d", action="store_true", help="Set deterministic flag for prediction")
    parser.add_argument("--k", type=int, default=500, help="Number of k in BPE")
    args = parser.parse_args()
        

    if args.activity == "train_bpe":
        if not args.data or not args.save:
            parser.error("train_bpe requires --data and --save arguments")
        train_bpe(args)
    elif args.activity == "tokenize":
        if not args.load or not args.text:
            parser.error("tokenize requires --load and --text arguments")
        args.text = ' '.join(args.text) 
        tokenize(args)
    elif args.activity == "train_ngram":
        if not args.data or not args.save or args.n is None:
            parser.error("train_ngram requires --data, --save, and --n arguments")
        train_ngram(args)
    elif args.activity == "predict_ngram":
        if not args.load or not args.word:
            parser.error("predict_ngram requires --load and --word arguments")
        predict_ngram(args)

if __name__ == "__main__":
    main()