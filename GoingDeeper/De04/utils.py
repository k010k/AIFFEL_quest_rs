import re
import tensorflow as tf
import MeCab
import nltk

def read_data(path_to_data):
    with open(path_to_data, "r") as f:
        data = f.read().splitlines()
    return data


def preprocess_eng(sentence):
    sentence = sentence.lower().strip()
    
    sentence = re.sub(r"\([^)]*\)", "", sentence)  # 괄호로 닫힌 문자열 제거
    sentence = re.sub(r'["\,.]', "", sentence)  # ",.- 제거
    sentence = re.sub(r"([?!])", r" \1 ", sentence)  # ?! 앞뒤에 공백 추가
    sentence = re.sub(r"\s+", " ", sentence)  # 연속된 공백을 하나의 공백으로 변환
    
    # 특수기호 변경
    sentence = re.sub(r"\$", "dollars ", sentence)  # $ → dollars
    sentence = re.sub(r"%", " percents", sentence)  # % → percent
    sentence = re.sub(r"&", "and", sentence)  # & → and
    
    sentence = re.sub(r"'s\b", "", sentence)  # 소유격 제거
    
    # 알파벳과 ?! 제외한 모든 문자 공백 치환
    sentence = re.sub(r"[^a-zA-Z0-9?!]+", " ", sentence) 
    
    sentence = sentence.strip()
    
    sentence = '<start> ' + sentence + ' <end>'  # 문장 앞뒤에 start, end token 추가
    
    return sentence


def preprocess_kor(sentence):
    sentence = sentence.strip()

    sentence = re.sub(r"\([^)]*\)", "", sentence)  # 괄호로 닫힌 문자열 제거
    sentence = re.sub(r'["\,.]', "", sentence)  # ",.- 제거
    sentence = re.sub(r"([?!])", r" \1 ", sentence)  # ?! 앞뒤에 공백 추가
    sentence = re.sub(r"\s+", " ", sentence)  # 연속된 공백을 하나의 공백으로 변환
    
    sentence = re.sub(r"%", "퍼센트", sentence)  # % → 퍼센트
   
    # 한글, 문장부호(?!), 숫자만 남기고 나머지는 제거
    sentence = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ0-9?!]+", " ", sentence)
    
    sentence = sentence.strip()
    
    return sentence


def tokenize_eng(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(corpus)

    tensor = tokenizer.texts_to_sequences(corpus)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer



def tokenize_kor(corpus):
    mecab = MeCab.Tagger()
    
    # Tokenizing each sentence
    tokenized_corpus = []
    for sentence in corpus:
        parsed_text = mecab.parse(sentence)  # Parse the sentence
        tokens = [line.split("\t")[0] for line in parsed_text.split("\n") if line and "\t" in line]  # Extract words
        tokenized_corpus.append(tokens)

    # Keras Tokenizer to convert tokens to integers
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(tokenized_corpus)

    tensor = tokenizer.texts_to_sequences(tokenized_corpus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer