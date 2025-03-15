import os
import pickle
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

def clean_caption(caption):
    """Clean a single caption"""
    # preprocessing steps
    caption = caption.lower()  # convert to lower case
    caption = re.sub(r'[^A-Za-z ]', '', caption)  # delete everything that is not a letter
    caption = caption.replace('\s+', ' ')  # replace multiple spaces with one single space
    # add start and end tags to the caption
    caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
    return caption

def extract_vocabulary(captions_file):
    """Extract vocabulary from captions file"""
    # Read captions file
    with open(captions_file, 'r') as f:
        next(f)  # Skip header
        captions_doc = f.read()
    
    # Create mapping of image to captions
    mapping = {}
    for line in captions_doc.split("\n"):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1]
        # removing extension from image_id
        image_id = image_id.split(".")[0]
        # convert caption list to string
        caption = "".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        # store the caption
        mapping[image_id].append(caption)
    
    # Clean captions
    for key, captions in mapping.items():
        for i in range(len(captions)):
            captions[i] = clean_caption(captions[i])
    
    # Collect all captions
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    
    # Get vocabulary statistics
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)
    
    # Get word frequency
    word_counts = tokenizer.word_counts
    
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'tokenizer': tokenizer,
        'vocab_size': vocab_size,
        'max_length': max_length,
        'word_counts': word_counts,
        'sorted_words': sorted_words,
        'all_captions': all_captions
    }

def main():
    # Check if captions file exists
    captions_file = os.path.join("data", "captions.txt")
    if not os.path.exists(captions_file):
        print(f"Captions file not found at {captions_file}")
        print("Please provide the path to the captions.txt file:")
        captions_file = input("> ")
    
    # Extract vocabulary
    try:
        vocab_data = extract_vocabulary(captions_file)
        
        # Print vocabulary statistics
        print(f"Vocabulary size: {vocab_data['vocab_size']}")
        print(f"Maximum caption length: {vocab_data['max_length']}")
        
        # Print top 50 most common words
        print("\nTop 50 most common words:")
        for word, count in vocab_data['sorted_words'][:50]:
            print(f"{word}: {count}")
        
        # Save vocabulary to file
        with open("vocabulary.txt", "w") as f:
            f.write(f"Vocabulary size: {vocab_data['vocab_size']}\n")
            f.write(f"Maximum caption length: {vocab_data['max_length']}\n\n")
            f.write("Word frequency (sorted by count):\n")
            for word, count in vocab_data['sorted_words']:
                f.write(f"{word}: {count}\n")
        
        print(f"\nFull vocabulary saved to vocabulary.txt")
        
        # Save tokenizer
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(vocab_data['tokenizer'], f)
        
        print("Tokenizer saved to tokenizer.pkl")
        
    except Exception as e:
        print(f"Error extracting vocabulary: {str(e)}")

if __name__ == "__main__":
    main()
