import json
import re
import spacy
import pandas as pd
from collections import Counter
from nltk import bigrams as nltk_bigrams
from fuzzywuzzy import fuzz
import emoji
import numpy as np

# Load spacy model
nlp = spacy.load('en_core_web_sm')

# Decode strings
def decode_message(message):
    try:
        return message.encode('latin1').decode('utf-8')
    except UnicodeDecodeError:
        return message 

# Extract messages
def extract_messages_from_file(file_path, sender_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    messages = [decode_message(message['content']) for message in data['messages'] 
                if message.get('sender_name') == sender_name and 'content' in message]
    
    return messages

# Filter messages
def filter_messages(messages):
    return [msg for msg in messages if msg != "You sent an attachment." and not msg.startswith("Du hast das")]

# Remove tokens with numbers
def remove_tokens_with_numbers(token_list):
    return [token for token in token_list if not re.search(r'\d', token)]

# Process and filter messages
def process_and_filter_messages(messages):
    all_tokens = []
    
    for message in messages:
        doc = nlp(message)
        all_tokens.extend([token.text for token in doc])
    
    return remove_tokens_with_numbers(all_tokens)

# Replace emojis with name
def replace_emojis_with_names(text):
    return emoji.demojize(text, language='en')

# Normalize tokens with fuzzy matching
def normalize_tokens_with_fuzzy_matching(tokens, threshold=50):
    normalized_tokens = []
    stopwords = set(nlp.Defaults.stop_words)  
    
    for token in tokens:
        token = replace_emojis_with_names(token)
        
        if token.strip() == "" or token in ["(", ")", "[", "]", "{", "}", ":", ";", "!", "?", ",", ".", "<", "#", "%", '"', "*", "’", "-", '…', "'", "/", "uuuuu", "u", "you're", "urs", "uuuu"]:
            continue
        
        if ":(" in token:
            normalized_tokens.append('unhappy_face')
            continue
        
        if ":-(" in token:
            normalized_tokens.append('unhappy_face')
            continue
        
        if ":)" in token:
            normalized_tokens.append('happy_face')
            continue
        
        if re.search(r'heart', token.lower()):
            normalized_tokens.append('heart_emoji')
            continue
        
        if re.search(r'^[ðâ]|https', token) or token.lower() in stopwords:
            continue
        
        if re.search(r'^(baby|bebe|babe|bubu|bb|bebi)', token.lower()):
            normalized_tokens.append('baby')
            continue
        
        if re.search(r'((ha){2,}|(he){2,})', token.lower()):
            normalized_tokens.append('haha')
            continue
        
        found_match = False
        
        for normalized_token in normalized_tokens:
            if fuzz.ratio(token, normalized_token) >= threshold:
                found_match = True
                break
        
        if not found_match:
            normalized_tokens.append(token)
    
    return normalized_tokens

# Names and DJs
name_dict = {"niko", "mandy", "camila", "sedef", "doris", "edolovati", "luna", "aysi", "kutchi", "birgit", "s.e.d.e.f", "patrick", "darina", "ays", "andy", "oli",
             "zoya", "natascha", "vinnie", "maria", "ayse", "gigi", "hadid", "edo", "conxi", "fiona", "mandybitch", "fio", "alinas", "alina", "ramy", "moses"}

dj_dict = {"rhadoo", "vera", "locky", "ricardo", "sonja", "z@p", "shonky", "roza", "helmut"}

# Lemmatize and normalize names and names of DJs so no information is lost
def lemmatize_and_normalize_names(tokens):
    lemmatized_tokens = []
    
    doc = nlp(" ".join(tokens))
    
    for token in doc:
        token_lower = token.text.lower()
        
        if token_lower in dj_dict:
            lemmatized_tokens.append("DJ")
        elif token.ent_type_ == 'PERSON' or token_lower in name_dict:
            lemmatized_tokens.append("PERSON")
        else:
            lemmatized_tokens.append(token.lemma_)
    
    return lemmatized_tokens

# Generate bigrams from messages
def generate_bigrams_from_messages(messages):
    all_bigrams = []
    
    for message in messages:
        tokens = process_and_filter_messages([message])
        lemmatized_tokens = lemmatize_and_normalize_names(tokens)
        normalized_tokens = normalize_tokens_with_fuzzy_matching(lemmatized_tokens)
        
        normalized_tokens = [token for token in normalized_tokens if token.strip() != ""]
        message_bigrams = list(nltk_bigrams(normalized_tokens))
        all_bigrams.extend(message_bigrams)
    
    return all_bigrams

# Create co-occurrence matrix
def create_co_occurrence_matrix(bigrams):
    unique_words = set(word for bigram in bigrams for word in bigram)
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    
    co_occurrence_matrix = np.zeros((len(unique_words), len(unique_words)))
    
    for word1, word2 in bigrams:
        i = word_index[word1]
        j = word_index[word2]
        co_occurrence_matrix[i, j] += 1
        co_occurrence_matrix[j, i] += 1  
    
    return co_occurrence_matrix, unique_words

# Create co-occurrence edges
def create_co_occurrence_edges(co_occurrence_matrix, unique_words, threshold=1):
    edges = []
    nodes = list(unique_words)
    
    node_weights = {node: 0 for node in nodes}
    
    for i in range(co_occurrence_matrix.shape[0]):
        for j in range(i + 1, co_occurrence_matrix.shape[1]): 
            weight = co_occurrence_matrix[i, j]
            if weight >= threshold: 
                edges.append({
                    'source': nodes[i],  
                    'target': nodes[j],  
                    'weight': weight
                })
                
                node_weights[nodes[i]] += weight
                node_weights[nodes[j]] += weight  
    
    nodes_df = pd.DataFrame({'id': list(nodes)})
    edges_df = pd.DataFrame(edges)
    
    # Add weights to nodes_df
    nodes_df['weight'] = nodes_df['id'].map(node_weights)
    
    return nodes_df, edges_df

# List of file paths to process
file_paths = [
    #'messages_edo_ig.json',
    'message_camila.json',
    'messages_camila2.json',
    'messages_mandy_ig.json',
    'messages_paula_ig.json'
]

# Process messages
fiona_messages = []

for file_path in file_paths:
    messages = extract_messages_from_file(file_path, 'Fiona Elena Cosima')
    filtered = filter_messages(messages)
    fiona_messages.extend(filtered)
    
fiona_bigrams = generate_bigrams_from_messages(fiona_messages)
co_occurrence_matrix, unique_words = create_co_occurrence_matrix(fiona_bigrams)
nodes_df, edges_df = create_co_occurrence_edges(co_occurrence_matrix, unique_words)


nodes_df.to_csv('nodes_co_occurrence_friends.csv', index=False)
edges_df.to_csv('edges_co_occurrence_friends.csv', index=False)

# Convert nodes & save
nodes_json = nodes_df.to_dict(orient='records')
edges_json = edges_df.to_dict(orient='records')

graph_data = {
    "nodes": nodes_json,
    "links": edges_json
}

with open('graph_data_friends.json', 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, ensure_ascii=False)

print("Graph data saved to 'graph_data.json'")
