import re
import random
import itertools
import collections

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



# # Set random seed for reproducibility
# random.seed(42)

# # Test data
# test_text = "The quick brown fox jumps over the lazy dog. The dog barks at the fox. The fox runs away quickly."

# # Test unigram model
# print("Testing Unigram Model:")
# unigram = Ngram(1)
# unigram.train(test_text)

# print("\nUnigram probabilities:")
# print(unigram.probabilites())

# print("\nPredicting next word after 'the' (deterministic):")
# print(unigram.predict_next_word(("the",), determenistic=True))

# print("\nPredicting next word after 'the' (non-deterministic):")
# for _ in range(5):
#     print(unigram.predict_next_word(("the",), determenistic=False))

# # Test bigram model
# print("\n\nTesting Bigram Model:")
# bigram = Ngram(2)
# bigram.train(test_text)

# print("\nBigram probabilities:")
# print(bigram.probabilites())

# print("\nPredicting next word after 'the fox' (deterministic):")
# print(bigram.predict_next_word(("the", "fox"), determenistic=True))

# print("\nPredicting next word after 'the fox' (non-deterministic):")
# for _ in range(5):
#     print(bigram.predict_next_word(("the", "fox"), determenistic=False))

# # Test error cases
# print("\nTesting error cases:")
# try:
#     Ngram(3)
# except ValueError as e:
#     print(f"ValueError: {e}")

# unigram.predict_next_word(("nonexistent",))
# bigram.predict_next_word(("the",))
# bigram.predict_next_word(("nonexistent", "word"))

# # Expected Output:
# """
# Testing Unigram Model:

# Unigram probabilities:
# {'the': {'quick': 0.2, 'lazy': 0.2, 'dog': 0.2, 'fox': 0.4}, 'quick': {'brown': 1.0}, 'brown': {'fox': 1.0}, 'fox': {'jumps': 0.5, 'runs': 0.5}, 'jumps': {'over': 1.0}, 'over': {'the': 1.0}, 'lazy': {'dog': 1.0}, 'dog': {'the': 0.5, 'barks': 0.5}, 'barks': {'at': 1.0}, 'at': {'the': 1.0}, 'runs': {'away': 1.0}, 'away': {'quickly': 1.0}, 'quickly': {'the': 1.0}}

# Predicting next word after 'the' (deterministic):
# fox

# Predicting next word after 'the' (non-deterministic):
# fox
# dog
# quick
# fox
# lazy


# Testing Bigram Model:

# Bigram probabilities:
# {('the', 'quick'): {'brown': 1.0}, ('quick', 'brown'): {'fox': 1.0}, ('brown', 'fox'): {'jumps': 1.0}, ('fox', 'jumps'): {'over': 1.0}, ('jumps', 'over'): {'the': 1.0}, ('over', 'the'): {'lazy': 1.0}, ('the', 'lazy'): {'dog': 1.0}, ('lazy', 'dog'): {'the': 1.0}, ('the', 'dog'): {'barks': 1.0}, ('dog', 'barks'): {'at': 1.0}, ('barks', 'at'): {'the': 1.0}, ('the', 'fox'): {'runs': 1.0}, ('fox', 'runs'): {'away': 1.0}, ('runs', 'away'): {'quickly': 1.0}}

# Predicting next word after 'the fox' (deterministic):
# runs

# Predicting next word after 'the fox' (non-deterministic):
# runs
# runs
# runs
# runs
# runs

# Testing error cases:
# ValueError: Can only support unigram and bigram.
# Error: The word'nonexistent' is not in the dictionary
# Error: Bigram model, there should be two inputs
# Error: The word('nonexistent', 'word') is not in the dictionary
# """

class BPE:
    

    def __init__(self):
        self.vocabulary = {}

    def train(self, data, k : int = 500):
        tokens = self.split_data_BPE(data)
        # Initialize vocabulary with individual tokens
        self.vocabulary = {token: i for i, token in enumerate(set(tokens))}
        
        for _ in range(k):
            double_pairs = list(itertools.pairwise(tokens))
            pair_counts = collections.Counter(double_pairs)
            if not pair_counts:
                break
            most_frequency = max(pair_counts, key=pair_counts.get)
            new_token = ''.join(most_frequency)
            self.vocabulary[new_token] = len(self.vocabulary)
            # you need to replace the merged token in the vocabulary right?
            tokens = self.merge_tokens(tokens, most_frequency, new_token)

    def tokenize(self, text):
        tokens = self.split_data_BPE(text)
        token_results = []
        token_ids = []

        counter = 0
        while counter < len(tokens):
            for j in range(len(tokens), counter, -1):
                segment = ''.join(tokens[counter:j])
                if segment in self.vocabulary:
                    token_results.append(segment)
                    token_ids.append(self.vocabulary[segment])
                    counter = j
                    break
                else:
                    token_results.append(tokens[counter])
                    token_ids.append(self.vocabulary.get(tokens[counter], len(self.vocabulary)))
                    counter += 1
        return token_results, token_ids


    def merge_tokens(self, tokens, pair, new_token):
        token_str = ''.join(tokens)
        pair_str = ''.join(pair)
        token_str = token_str.replace(pair_str, new_token)
        return token_str.split()

      

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
        return processed_tokens
    
    

# Create a BPE instance
bpe = BPE()

# Train the model
training_text = "How much wood could a woodchuck chuck?"
bpe.train(training_text, k=2)

# Tokenize the same text
tokens, token_ids = bpe.tokenize(training_text)

# Print results
print("Vocabulary:", bpe.vocabulary)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
        


    





