import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

def extract_text(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        article_text = '\n'.join([p.get_text() for p in soup.find_all('p')])
        return article_text
    else:
        print(f"Error: Unable to fetch data from {url}")
        return None
def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file]
    return words

positive_words = load_dictionary('positive-words.txt')
negative_words = load_dictionary('negative-words.txt')

def read_custom_stopwords(file_path):
    with open(file_path, 'r') as file:
        custom_stopwords = [line.strip() for line in file]
    return custom_stopwords

stopwords_auditor_path= 'StopWords_Auditor.txt'
stopwords_auditor= read_custom_stopwords(stopwords_auditor_path)

stopwords_currencies_path='StopWords_Currencies.txt'
stopwords_currencies= read_custom_stopwords(stopwords_currencies_path)

stopwords_generic_path='StopWords_Generic.txt'
stopwords_generic= read_custom_stopwords(stopwords_generic_path)

stopwords_datenum_path='StopWords_DatesandNumbers.txt'
stopwords_datenum= read_custom_stopwords(stopwords_datenum_path)

stopwords_genericlong_path= 'StopWords_GenericLong.txt'
stopwords_genericlong= read_custom_stopwords(stopwords_genericlong_path)

stopwords_geographic_path='StopWords_Geographic.txt'
stopwords_geographic= read_custom_stopwords(stopwords_geographic_path)

stopwords_name_path= 'StopWords_Auditor.txt'
stopwords_name= read_custom_stopwords(stopwords_name_path)

stop_words= set(stopwords.words('english') + stopwords_auditor + stopwords_currencies+ stopwords_datenum+stopwords_genericlong+stopwords_geographic + stopwords_name)

def analyze_text(article_text):

    words = word_tokenize(article_text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)

    # Polarity Score
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

    # Subjectivity Score
    
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)

    # Named Entity Recognition (spaCy example)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(article_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Keyword Extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([article_text])
    keywords = vectorizer.get_feature_names_out()

    # Additional variables calculations
    avg_sentence_length = len(words) / len(sent_tokenize(article_text))
    
    percentage_complex_words = sum(1 for word in words if len(word) > 2) / len(words)
    
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    avg_number_of_words_per_sentence = len(words) / len(sent_tokenize(article_text))
    
    complex_word_count = sum(1 for word in words if len(word) > 2)
    
    word_count = len(words)
    
    syllable_per_word = calculate_syllable_per_word(words)
    
    personal_pronouns = count_personal_pronouns(article_text)
    
    avg_word_length = sum(len(word) for word in words) / len(words)

    return {
        "POSITIVE SCORE": positive_score,
        "NEGATIVE SCORE": negative_score,
        "POLARITY SCORE": polarity_score,
        "SUBJECTIVE SCORE": subjectivity_score,
        "AVG SENTENCE LENGTH": avg_sentence_length,
        "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words,
        "FOG INDEX": fog_index,
        "AVG NUMBER OF WORDS PER SENTENCE": avg_number_of_words_per_sentence,
        "COMPLEX WORD COUNT": complex_word_count,
        "WORD COUNT": word_count,
        "SYLLABLE PER WORD": syllable_per_word,
        "PERSONAL PRONOUNS": personal_pronouns,
        "AVG WORD LENGTH": avg_word_length
    }

# Function to calculate syllables per word
def calculate_syllable_per_word(words):
    syllable_count = 0
    
    for word in words:
        # Adjust for exceptions like words ending with "es" or "ed"
        if word.endswith(('es', 'ed')):
            continue
        
        vowels = 'aeiouy'
        count = 0
        
        # Count the number of vowels in the word
        for char in word:
            if char.lower() in vowels:
                count += 1
        
        # Adjust for words with no vowels
        if count == 0:
            count = 1
        
        syllable_count += count
    
    return syllable_count / len(words)

# Function to count personal pronouns
def count_personal_pronouns(text):
    personal_pronoun_regex = re.compile(r'\b(?:I|we|my|ours|us)\b', flags=re.IGNORECASE)
    return len(re.findall(personal_pronoun_regex, text))

# Read URLs from an input Excel file
input_data = pd.read_excel('Input.xlsx')

# Iterate through each row in the input data
for index, row in input_data.iterrows():
    url = row['URL']
    
    # Extract text from the URL
    article_text = extract_text(url)
    
    if article_text:
        # Perform text analysis
        analysis_result = analyze_text(article_text)
        
        # Output the analysis result
        print(f"URL: {url}")
        print("Analysis Result:", analysis_result)
        print("\n")

        # Save the analysis result to an output file (modify as needed)
        output_data = pd.DataFrame([row.tolist() + list(analysis_result.values())], columns=input_data.columns.tolist() + list(analysis_result.keys()))
        output_data.to_excel(f"Output_{row['URL_ID']}.xlsx", index=False)
import os
input_data = pd.read_excel('Input.xlsx')
merged_results = pd.DataFrame()
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    output_file = f"Output_{url_id}.xlsx"
    if os.path.exists(output_file):
        output_data = pd.read_excel(output_file)
        output_data['URL_ID'] = url_id
        merged_results = merged_results.append(output_data, ignore_index=True)
    else:
        print(f"Output file {output_file} not found.")
merged_results.to_excel('Output Data Structure.xlsx', index=False)

import pandas as pd
file_path = 'Output Data Structure.xlsx'
results = pd.read_excel(file_path)

with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:

    results.to_excel(writer, sheet_name='Sheet1', index=False)

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    for i, col in enumerate(results.columns):
        max_len = max(results[col].astype(str).apply(len).max(), len(col)) + 2
        worksheet.set_column(i, i, max_len)
        
    header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center'})
    for col_num, value in enumerate(results.columns.values):
        worksheet.write(0, col_num, value, header_format)