from collections import Counter, OrderedDict
from itertools import pairwise

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

        # Initialize vocabulary with individual characters
        i = 0
        for char in characters:
            if char not in self.vocabulary:
                self.vocabulary[char] = i
                i += 1

        for _ in range(k):
            # Create pairs of adjacent characters
            pairs = [''.join(pair) for pair in pairwise(characters)]

            # Count frequency of each pair
            pair_frequency = Counter(pairs)

            if not pair_frequency:
                break  # Exit if there are no pairs left to process

            # Find the maximum frequency
            max_frequency = max(pair_frequency.values())

            # Find all pairs with the maximum frequency
            most_frequent_pairs = []
            for pair, count in pair_frequency.items():
                if count == max_frequency:
                    most_frequent_pairs.append(pair)

            # Append each most frequent pair to the vocabulary
            for common_pair in most_frequent_pairs:
                if common_pair not in self.vocabulary:
                    self.vocabulary[common_pair] = len(self.vocabulary)

            # Replace occurrences of the most frequent pairs
            for common_pair in most_frequent_pairs:
                characters = self.replace_pair(characters, common_pair)

            # print(f"Iteration {_+1}:")
            # print(f"Updated Characters: {characters}")
            # print(f"Updated Vocabulary: {self.vocabulary}")
    
    def tokenize(self, corpus):
        characters = self.split_characters(corpus)
        
        # Loop through each token in the vocabulary in the order of insertion
        for token in self.vocabulary:
            if len(token) > 1:  # Skip single characters
                i = 0
                while i < len(characters) - 1:
                    if characters[i] + characters[i+1] == token:
                        characters[i:i+2] = [token]
                        i += 2
                    else:
                        i += 1
        
        # Add any new pairs to the vocabulary
        i = 0
        while i < len(characters) - 1:
            current_pair = characters[i] + characters[i+1]
            if current_pair not in self.vocabulary:
                self.vocabulary[current_pair] = len(self.vocabulary)
            i += 1

        # Handle unseen characters
        for char in characters:
            if char not in self.vocabulary:
                self.vocabulary[char] = len(self.vocabulary)
        
        token_ids = [self.vocabulary[token] for token in characters]
        
        print(f"Updated Characters: {characters}")
        # print(f"Token IDs: {token_ids}")
        # print(f"Updated Vocabulary: {self.vocabulary}")
        
        return characters, token_ids

            



# Example usage
lala = BPE()
haha = """hello world
hello new world
new world is exciting
world is full of wonders
"""
lala.train(haha, k=3)
lulu = lala.tokenize("""hello world
exciting new world
wonders of the world
""")
print(lulu)
