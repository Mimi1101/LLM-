import argparse
import random
import re
import pickle
from typing import Union 


class Ngram:
    def __init__(self, n):
        self.n = n
        if n not in [1, 2]:
            raise ValueError("Can only support unigram and bigram.")
    
    def split_words(self, texts):
        regex = r"\w+|[^\w\s]"
        splitted_words = re.findall(regex, texts)
        return splitted_words
    
    def calculate_probabilities(self, words):
        if self.n == 1:
            # Unigram model
            total_word_count = len(words)  # Total number of words
            self.probabilities_dictionary = {}

            for word, count in self.unique_words_dictionary.items():
                self.probabilities_dictionary[word] = count / total_word_count

        elif self.n == 2:
            # Bigram model
            self.probabilities_dictionary = {}

            for word, next_words in self.unique_words_dictionary.items():
                total_count = sum(next_words.values()) 

                self.probabilities_dictionary[word] = {}
                for next_word, count in next_words.items():
                    self.probabilities_dictionary[word][next_word] = count / total_count


    def train(self, data):
        words = self.split_words(data)
        self.unique_words_dictionary = {}
        
        if self.n == 1:
            for word in words:
                if word not in self.unique_words_dictionary:
                    self.unique_words_dictionary[word] = 1  # Initialize count
                else:
                    self.unique_words_dictionary[word] += 1  
        
        elif self.n == 2:
            for i in range(len(words) - 1):
                current_words = words[i]
                next_word = words[i+1]
                if current_words not in self.unique_words_dictionary:
                    self.unique_words_dictionary[current_words] = {} # nestin it
                if next_word not in self.unique_words_dictionary[current_words]:
                    self.unique_words_dictionary[current_words][next_word] = 1
                else:
                    self.unique_words_dictionary[current_words][next_word] += 1
        print (words)
        self.calculate_probabilities(words)

    def predict_next_word(self, input: tuple, deterministic: bool = False):
        if self.n == 1:
            current_word = input[0]
            
            if current_word not in self.probabilities_dictionary:
                print("Error: Word not found in vocabulary.")
                return None
            
            probabilities = self.probabilities_dictionary[current_word]
            
            if deterministic:
                return max(self.probabilities_dictionary, key=self.probabilities_dictionary.get)
            else:
                keys_list = list(self.probabilities_dictionary.keys())
                weights = list(self.probabilities_dictionary.values())
                return random.choices(keys_list, weights=weights)[0]
        
        elif self.n == 2:
            word1, word2 = input
            
            if word1 not in self.probabilities_dictionary:
                print("Error: Word not found in vocabulary.")
                return None
            
            if word2 not in self.probabilities_dictionary[word1]:
                print("Error: Word pair not found in vocabulary.")
                return None
            
            probabilities = self.probabilities_dictionary[word1][word2]
            
            if deterministic:
                return max(probabilities, key=probabilities.get)
            else:
                keys_list = list(probabilities.keys())
                weights = list(probabilities.values())
                return random.choices(keys_list, weights=weights)[0]


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
    # Load the saved model
    try:
        with open(args.load, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not load model from {args.load}. File not found.")
        return

    # Split the input words (assuming space-separated input for bigrams)
    input_words = args.word.split()
    
    if model.n == 1 and len(input_words) != 1:
        print("Error: For a unigram model, you should provide exactly one word.")
        return
    elif model.n == 2 and len(input_words) != 2:
        print("Error: For a bigram model, you should provide exactly two words.")
        return

    # Perform predictions
    predicted_words = []
    for _ in range(args.nwords):
        next_word = model.predict_next_word(tuple(input_words), deterministic=args.d)
        if next_word is None:
            break
        predicted_words.append(next_word)
        
        # Update the input for the next iteration in case of bigrams
        if model.n == 2:
            input_words = [input_words[-1], next_word]
        else:
            input_words = [next_word]
    
    # Output the predicted words
    print(" ".join(predicted_words))



def save_model(model: Union[Ngram], path: str) -> None:
    """Save the model to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
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
