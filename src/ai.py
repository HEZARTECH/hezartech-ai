#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: ai.py
"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch

from nltk.tokenize import sent_tokenize
from transformers import pipeline

from flair.data import Sentence
from flair.models import SequenceTagger

import re

# Noktalama işaretleri tokenizer fonksiyonlarının import edildiğinden emin oluyoruz.
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk; nltk.download('punkt', quiet=True)

# Firma tespiti için modeli yükleme (Named Entity Recognition)
ner_recognizer = SequenceTagger.load('flair/ner-english-large')

# Duygu analizi için Huggingface transformers modeli yükleme
sentiment_analysis = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

#data_path = '../model/heartech_ai/'

# Model ve tokenizer'ı yükle.
loaded_model = BertForSequenceClassification.from_pretrained(data_path)
loaded_tokenizer = BertTokenizer.from_pretrained(data_path)

# Modeli değerlendirme moduna al.
loaded_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli GPU'ya aktarma.
loaded_model.to(device)

# Etiketlerin ve sayısal eşlemelerin tanımlanması.
label_mapping: dict[int, str] = {
    0: "notr",
    1: "positive",
    2: "negative",
    3: "positive|negative",
}

# Result Vectors
firm_list: list[str] = []
results: list[dict[str, str]] = []

ner_recognizer: SequenceTagger = SequenceTagger.load('flair/ner-english-large')

def print_result_vectors():
    __import__('pprint').pprint({
        "entity_list": firm_list,
        "results": results
    })

def split_sentences(sentence: str) -> list[str]:
    return sent_tokenize(sentence, language='turkish')

sentiment_analysis = pipeline("sentiment-analysis", model=data_path)

def remove_word(sentence: str, word: str) -> str:
    words = word_tokenize(sentence)
    filtered_words = [w for w in words if w != word]
    return " ".join(filtered_words)

def separate_sentences_via_conjunctions(sentence: str) -> list[str]:
    words: list[str] = word_tokenize(sentence)

    negative_conjunctions: list[str] = [
        "ama", "fakat", "lakin",
        "ancak", "oysa", "halbuki",
        "gerçi"
    ]

    split_sentences = []
    current_sentence = []

    for word in words:
        if word in negative_conjunctions:
            split_sentences.append(" ".join(current_sentence))
            current_sentence = []
        current_sentence.append(word)

    if current_sentence:
        split_sentences.append(" ".join(current_sentence))

    filtered_split_sentences: list[str] = []

    for _sentence in split_sentences:
        for conjunction in negative_conjunctions:
            if conjunction in _sentence:
                _sentence = remove_word(_sentence, conjunction)
        filtered_split_sentences.append(_sentence.strip())

    return filtered_split_sentences

def reset_result_vectors() -> None:
    global firm_list, results
    firm_list = []
    results = []

def cut_after_apostroph(line: str) -> str:
    return re.sub(r"'.*$", "", line)

def sentiment_analyzer(sentence_i: str, firm: str) -> str:
    firm_list.append(firm)

    sentiment_result = sentiment_analysis(sentence_i)
    sentiment = sentiment_result[0]['label']

    if sentiment == "LABEL_1":
        results.append({"entity": firm, "sentiment": "Pozitif"})

    elif sentiment == "LABEL_0":
        results.append({"entity": firm, "sentiment": "Nötr"})

    elif sentiment == "LABEL_2":
        results.append({"entity": firm, "sentiment": "Negatif"})

    elif sentiment == "LABEL_3":
        results.append({"entity": firm, "sentiment": "Pozitif"})
        results.append({"entity": firm, "sentiment": "Negatif"})

    return sentiment

def analyze_sentences(_input: str) -> None:
    sentences = split_sentences(_input)

    for sentence in sentences:
        all_entities = []
        firms = []
        firmcount = 0
        text = Sentence(sentence)
        ner_recognizer.predict(text)

        for entity in text.get_spans('ner'):
            all_entities.append({
                "entity_name": entity.text,
                "type": entity.get_label('ner').value
            })

        # Firma olarak kategorize edilenleri çekiyoruz.
        for entity in all_entities:
            if entity['type'] == 'ORG':

                firms.append(cut_after_apostroph(entity['entity_name']))
                firmcount += 1
        # Tekrarlayan firma adlarının tek bir tanesini aldı.
        firms = list(set(firms))
        firmcount = len(firms)

        if firmcount == 1:
            for firm in firms:
                if firm in sentence:
                    sentiment = sentiment_analyzer(sentence, firm)
                    print(f"Cümle:{sentence}\nSentiment:{sentiment}")
        else:
            print("Birden fazla firma adı var. Karşıtlık bağlacı aranacak.")
            inside_sentences = separate_sentences_via_conjunctions(sentence)
            if len(inside_sentences) == 1:
                print(f"Cümle:{inside_sentences}")
                print("Cümlede karşıtlık bağlacı yoktur. Bu cümleye sentiment atandı.")
                for firm in firms:
                    if firm in inside_sentences[0]:
                        sentiment = sentiment_analyzer(inside_sentences[0], firm)
            else:
                print(f"Cümle:{inside_sentences}")
                print("Karşıtlık bağlacı bulundu. Bu aralıktan cümle bölündü.")

                for _sentence in inside_sentences:
                    for firm in firms:
                        if firm in _sentence:
                            sentiment = sentiment_analyzer(_sentence, firm)


def make_predict(comment: str) -> None:
    analyze_sentences(comment)
    print_result_vectors()
    reset_result_vectors()

if __name__ == '__main__':
    # Örnek yorum
    comment = "@TurkcellHizmet ya vereceÄŸiniz hizmeti... ya bilader adada bile iyi Ã§ekerken, ÅŸehir merkezinde nasÄ±l Ã§ekmiyor bu"

    analyze_sentences(comment)
    print_result_vectors()
    reset_result_vectors()

