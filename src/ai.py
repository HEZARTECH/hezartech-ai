#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: ai.py
@author: Yiğit GÜMÜŞ
@date: 2024-08-06 13:33:16
"""
from nltk.tokenize import sent_tokenize # Sentence splitter/tokenizer.
from transformers import pipeline

from flair.data import Sentence
from flair.models import SequenceTagger

import nltk; nltk.download('punkt', quiet=True)
import zeyrek


analyzer = zeyrek.MorphAnalyzer()
kelime = 'ama'
print(analyzer.analyze(kelime))

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

def seperate_sentences_via_stopwords(sentence: str) -> list[str]:
    analyzer = zeyrek.MorphAnalyzer()
    conjuctions_in_sentence = []

    for parsed_words in analyzer.analyze(sentence):
        for formatted in parsed_words:
            if formatted == 'Conj':
                conjuctions_in_sentence.append(formatted)

    conjuctions_in_sentence = list(set(conjuctions_in_sentence))

    #@TODO: bulduğun bağlaçları import et
    #@TODO: Implement here.
    return ""

def analyze(comment: str):
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
        internal_sentences = seperate_sentences_via_stopwords(sentences)

        for internal_of_internal_sentences in internal_sentences:
            for firm in firms:
                if firm in internal_of_internal_sentences.split():
                    sentiment_analysis(internal_of_internal_sentences)

    return {"entity_list": firms, "results": results}


if __name__ == '__main__':
    # Örnek yorum
    comment = "@TurkcellHizmet ya vereceÄŸiniz hizmeti... ya bilader adada bile iyi Ã§ekerken, ÅŸehir merkezinde nasÄ±l Ã§ekmiyor bu"

