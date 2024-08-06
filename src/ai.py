#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: ai.py
@author: Yiğit GÜMÜŞ
@date: 2024-08-06 13:33:16
"""
from nltk.tokenize import sent_tokenize # Sentence splitter/tokenizer.
from nltk.tokenize import word_tokenize # Word tokenizer.
from transformers import pipeline

from flair.data import Sentence
from flair.models import SequenceTagger

# Noktalama işaretleri tokenizer modellerinin indiğinden emin oluyoruz.
import nltk; nltk.download('punkt', quiet=True)

# Firma tespiti için modeli yükleme (Named Entity Recognition)
ner_recognizer = SequenceTagger.load('flair/ner-english-large')

# Duygu analizi için Huggingface transformers modeli yükleme
sentiment_analysis = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

def get_sentiment(text):
    prediction = sentiment_analysis(text)[0]['label']
    if prediction == 'LABEL_0':
        return 'Olumsuz'
    elif prediction == 'LABEL_1':
        return 'Olumlu'
    elif prediction == 'LABEL_2':
        return 'Olumlu|Olumsuz'
    elif prediction == None:
        return 'Tarafsiz'

def remove_word(sentence, word):
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


def absa_analyzer(comment: str): # Aspect-Based Sentiment Analyzer
    sentence = Sentence(comment)
    ner_recognizer.predict(sentence)


    all_entities = []
    for entity in sentence.get_spans('ner'):
        print(f'Entity: {entity.text}, Label: {entity.get_label("ner").value}')
        all_entities.append({
            "entity_name": entity.text,
            "type": entity.get_label('ner').value
        })

    # Varlık tanıma (Firma adlarını filtreleme)
    firms = []

    for entity in all_entities:
        if entity['type'] == 'ORG':
            firms.append(entity['entity_name'])

    # Cümle bölme
    sentences = sent_tokenize(comment, language="turkish")

    # Duygu analizi ve etiketleme
    results = []
    for sentence in sentences:
        internal_sentences = separate_sentences_via_conjunctions(sentences)

        for internal_of_internal_sentences in internal_sentences:
            for firm in firms:
                if firm in internal_of_internal_sentences.split():
                    sentiment_analysis(internal_of_internal_sentences)

    return {"entity_list": firms, "results": results}


if __name__ == '__main__':
    # Örnek yorum
    comment = "@TurkcellHizmet ya vereceÄŸiniz hizmeti... ya bilader adada bile iyi Ã§ekerken, ÅŸehir merkezinde nasÄ±l Ã§ekmiyor bu"

