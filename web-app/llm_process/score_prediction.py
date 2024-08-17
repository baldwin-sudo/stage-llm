import re
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Load pre-trained BERT model and tokenizer for multiple languages
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Load datasets
questions = pd.read_csv('./questions.csv')
dataset = pd.read_csv('./scoring.csv')
dataset = dataset.reset_index(drop=True)

# French stop words
french_stopwords = stopwords.words('french')

def preprocess_text(text):
    '''
    Preprocess the input text by lowercasing, removing punctuation, and extra whitespace.
    '''
    if not isinstance(text, str):
        text = str(text)  # Convert non-string inputs to string
    
    text = text.lower()  # Lowercase the text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def preprocess(text):
    '''
    Preprocess text by capturing negations and removing stop words.
    '''
    tokens = ['negation' if word == 'ne' or word == "n'" else word for word in text.split()]
    tokens = [word for word in tokens if word.lower() not in french_stopwords]
    return ' '.join(tokens)

def get_synonym_RO(word, reference_vocab):
    '''
    Get a synonym for a word from the reference vocabulary using WordNet.
    '''
    if word in reference_vocab:
        return word
    synonyms = []
    for syn in wordnet.synsets(word, lang='fra'):
        for lemma in syn.lemma_names("fra"):
            if lemma in reference_vocab:
                synonyms.append(lemma)
    return synonyms[0] if synonyms else word

def replace_with_synonyms(text, reference_vocab):
    '''
    Replace terms in the text with their synonyms from the reference vocabulary.
    '''
    tokens = text.split()
    new_tokens = []
    for token in tokens:
        synonym = get_synonym_RO(token, reference_vocab)
        new_tokens.append(synonym)
    return ' '.join(new_tokens)

def encode_sentence(sentence):
    '''
    Encode a sentence using BERT embeddings.
    '''
    sentence = preprocess_text(sentence)
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embeddings

def compute_tfidf_similarity(texts, sentence1, sentence2):
    '''
    Compute the semantic similarity between two sentences using TF-IDF embeddings.
    '''
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    tfidf_matrix = vectorizer.fit_transform(texts + [sentence1, sentence2])
    sim_matrix = cosine_similarity(tfidf_matrix[-2:], tfidf_matrix[:-2])
    return sim_matrix[0, 1]

def calculate_score(user_answer, question_id, use_tfidf=False, use_lsa=False, questions_df=questions, answers_df=dataset):
    '''
    Calculate the score for the user answer based on reference answers.
    '''
    try:
        question_rows = questions_df[questions_df["ID"] == question_id]
        if question_rows.empty:
            raise ValueError(f"Question ID: '{question_id}' not found in 'questions' DataFrame.")
        
        question_text = question_rows["question"].values
        if len(question_text) == 0:
            raise ValueError(f"No question text found for ID: '{question_id}'.")
        
        question_text = question_text[0]
        
        reference_answers = answers_df[answers_df["ID"] == question_id]["Reponse"].tolist()
        reference_scores = answers_df[answers_df["ID"] == question_id]["Score"].tolist()
        
        if not reference_answers or not reference_scores:
            raise ValueError("No reference answers or scores found for the given question ID.")
        
        preprocessed_references = [preprocess(ans) for ans in reference_answers]
        reference_vocab = list(set(word_tokenize(' '.join(preprocessed_references))))
        
        if use_tfidf:
            similarity_scores = [compute_tfidf_similarity(reference_answers, user_answer, answer) for answer in reference_answers]
            highest_similarity_index = similarity_scores.index(max(similarity_scores))
            score = reference_scores[highest_similarity_index]
        elif use_lsa:
            vectorizer = TfidfVectorizer(preprocessor=preprocess)
            tfidf_matrix = vectorizer.fit_transform(preprocessed_references)
            svd = TruncatedSVD(n_components=2)  # Adjust n_components based on your needs
            X_lsa = svd.fit_transform(tfidf_matrix)
            
            preprocessed_user_answer = preprocess(user_answer)
            user_answer = replace_with_synonyms(preprocessed_user_answer, reference_vocab)
            user_tfidf = vectorizer.transform([user_answer])
            user_lsa = svd.transform(user_tfidf)
            
            similarity_scores = cosine_similarity(user_lsa, X_lsa)[0]
            highest_similarity_index = similarity_scores.argmax()
            score = reference_scores[highest_similarity_index]
        else:
            embedded_user_answer = encode_sentence(user_answer)
            similarity_scores = []
            
            for answer in reference_answers:
                embedded_reference_answer = encode_sentence(answer)
                answer_sim_score = cosine_similarity(embedded_user_answer, embedded_reference_answer)[0][0]
                similarity_scores.append(answer_sim_score)
            
            if len(similarity_scores) < 2:
                raise ValueError("Not enough reference answers for comparison.")
            
            similarity_scores_and_scores = list(zip(similarity_scores, reference_scores))
            highest_similarity, second_highest_similarity = sorted(similarity_scores_and_scores, key=lambda x: x[0], reverse=True)[:2]
            
            normalized_sim1 = highest_similarity[0] / (highest_similarity[0] + second_highest_similarity[0])
            normalized_sim2 = second_highest_similarity[0] / (highest_similarity[0] + second_highest_similarity[0])
            
            score = highest_similarity[1] * normalized_sim1
            
            if highest_similarity[1] > second_highest_similarity[1]:
                score -= second_highest_similarity[1] * normalized_sim2 / 2
            else:
                score += second_highest_similarity[1] * normalized_sim2 / 2
        
        return score
    except Exception as e:
        print(f"Error: {e}")
        print(f"Question ID: '{question_id}' not found in the dataset!")
        return None

if __name__ == "__main__":
    # Example usage
    user_answer = "pas Beaucoup"
    
    question_id = 10  # question :Avez-vous des technologies ou des équipements qui doivent être mis à jour ?
    print("Question : Avez-vous des technologies ou des équipements qui doivent être mis à jour ?")
    print("---------------------------------")
    print("Réponse de reference")
    print("Beaucoup,0")
    print("Moyenne,1")
    print("Peu,2")
    print("Tres peu,3") 
    print("---------------------------------")
    print("Exemple de réponse d'utilisateur : "+user_answer)
    score_tf = calculate_score(user_answer, question_id, use_tfidf=True, use_lsa=False)
    
    score_lsa = calculate_score(user_answer, question_id, use_tfidf=False, use_lsa=True)

    score_bert = calculate_score(user_answer, question_id, use_tfidf=False, use_lsa=False)
    print("---------------------------------")
   
    print(f"Calculated Score TF-IDF: {score_tf}")
    print(f"Calculated Score LSA: {score_lsa}")
    print(f"Calculated Score BERT: {score_bert}")
