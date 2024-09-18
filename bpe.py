import argparse
from collections import Counter, OrderedDict
from itertools import pairwise
import pickle

import tqdm

class BPE:

    def __init__(self):
        self.vocabulary = OrderedDict()

    def split_characters(self, text):
        return list(text)

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
        characters = self.split_characters(corpus)
        self.all_merges = []

        # Initialize vocabulary with individual characters
        
        for char in characters:
            if char not in self.vocabulary:
                self.vocabulary[char] = len(self.vocabulary)

        for _ in tqdm.tqdm(range(k), desc="Training BPE", unit="iter"):

            # Create pairs of adjacent characters
            pairs = [''.join(pair) for pair in pairwise(characters)]

            # Count frequency of each pair
            pair_frequency = Counter(pairs)

            if not pair_frequency:
                break  # Exit if there are no pairs left to process

            # Find the maximum frequency

            # Find all pairs with the maximum frequency
            most_frequent_pair = pair_frequency.most_common(1)
            if not most_frequent_pair:
                break
            common_pair, freq = most_frequent_pair[0]

            # Append each most frequent pair to the vocabulary
            
            if common_pair not in self.vocabulary:
                self.vocabulary[common_pair] = len(self.vocabulary)
            self.all_merges.append(common_pair)

            # Replace occurrences of the most frequent pairs
            # Replace occurrences of the most frequent pair
            characters = self.replace_pair(characters, common_pair)

            # print(f"Iteration {_+1}:")
            # print(f"Updated Characters: {characters}")
            # print(f"Updated Vocabulary: {self.vocabulary}")
    
    def tokenize(self, corpus):
        characters = self.split_characters(corpus)

        # Apply merges in the same order as they were learned during training
        for merge in self.all_merges:
            characters = self.replace_pair(characters, merge)

        # Map tokens to token IDs using the pre-trained vocabulary
        token_ids = [self.vocabulary[token] for token in characters]

        print(f"Updated Characters: {characters}")
        return characters, token_ids


def save_model(model: BPE, path: str) -> None:
    """Save the model to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str) -> BPE:
    """Load the BPE model from a file using pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)
            
def train_bpe(args):
    """Train a BPE tokenizer and save it."""
    model = BPE()
    try:
            with open(args.data, 'r', encoding='utf-8') as f:  # Specify utf-8 encoding
                corpus = f.read()
            model.train(corpus, k= args.k)
            save_model(model, args.save)
            print(f"BPE model trained and saved to {args.save}")
    except UnicodeDecodeError:
            print("Error: Unable to read the file. It may be encoded in a different format.")
            print("Try opening the file in a text editor and saving it with UTF-8 encoding.")
    except IOError:
            print(f"Error: Unable to read the file at {args.data}. Please check if the file exists and you have permission to read it.")
   

def tokenize(args):
    """Tokenize the given text using a loaded BPE model."""
    # Load the BPE model
    model = load_model(args.load)
    
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
    parser.add_argument("--n", type=int, choices=[1, 2], help="Order of the N-gram model")
    parser.add_argument("--d", action="store_true", help="Set deterministic flag for prediction")
    parser.add_argument("--k", type=int, default=500, help="Number of merge operations for BPE")
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


    

