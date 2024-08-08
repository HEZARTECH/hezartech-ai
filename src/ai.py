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
from collections import OrderedDict
import json

from typing import Any

# Noktalama işaretleri tokenizer fonksiyonlarının import edildiğinden emin oluyoruz.
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk; nltk.download('punkt', quiet=True)

# Firma tespiti için modeli yükleme (Named Entity Recognition)
ner_recognizer = SequenceTagger.load('flair/ner-english-large')

# Duygu analizi için Huggingface transformers modeli yükleme
data_path = '..\model\hezartech_ai'

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

# Sonuç Vektörleri
firm_list: list[str] = []
results: list[dict[str, str]] = []

ner_recognizer: SequenceTagger = SequenceTagger.load('flair/ner-english-large')

def split_sentences(text: str) -> list[str]:
    '''
        Bu fonksiyon cümleleri ayırmak için kullanılıyor.

        text: str (cümle)

        Returns:
            list[str]
    '''

    # Cümleleri nokta ve soru işaretleri ile ayırmak
    sentences = sent_tokenize(text, language='turkish')

    def handle_commas(sentence: str) -> list[str]:
        '''
            Virgülle ayrılan cümleleri tespit ve ayırmak için bir işlev
        '''
        parts = sentence.split(',')
        new_parts = []
        for part in parts:
            if part.strip():
                new_parts.append(part.strip())
        return new_parts


    # Her bir cümleyi virgülle ayırmak
    final_sentences = []
    for sentence in sentences:
        # Cümleyi virgüllerle ayırmak
        parts = handle_commas(sentence)
        final_sentences.extend(parts)

    return final_sentences

sentiment_analysis = pipeline("sentiment-analysis", model=data_path)

def remove_word(sentence: str, word: str) -> str:
    '''
        Bu fonksiyon cümleden belirli bir kelimeyi silmeye yarıyor.
        (Bunu gereksiz bağlaçları silmek için kullandık.)

        sentence: str (Cümle)
        word: str (Cümleden silinecek belirli kelime)

        Returns:
            str
    '''

    words = word_tokenize(sentence)
    filtered_words = [w for w in words if w != word]
    return " ".join(filtered_words)

def separate_sentences_via_conjunctions(sentence: str) -> list[str]:
    '''
        Cümleler alınıp negatif/karşılaştırma bağlaçları sayesinde ayrılıp cümle listelerine dönüştürlüyor.
        (Firmaları almamızda yardımcı olan bir mekanizma.)

        sentence: str (cümle)

        Returns:
            list[str] [`sentence` değişkeninin (cümlenin) negatif/karşıt anlam bağlaçları ile ayrılmış versiyonu.]
    '''

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
    '''
        Sonuç vektörlerini sıfırlamak için kullanılıyor.
    '''
    global firm_list, results
    firm_list = []
    results = []

def print_result_vectors():
    '''
        Sonuç vektörlerini ekrana bastırmak için kullanılıyor.
    '''
    __import__('pprint').pprint({
        "entity_list": firm_list,
        "results": results
    })

# Bu değişkenleri (Bu değişkenler asıl kullanılan ve sonuç döndürmede önemli yerler.)
firm_list = []
dandan = []
results = []

def cut_after_apostroph(line: str):
    '''
        Bu fonksiyon eklerden sonra gelen kesme işaretlerini ayırmada kullanılıyor.
        (Firma adını temizlemek için kullanılıyor.)

        line: str (text for remove apostroph)

        Returns:
            str [RegEx'den dönen yazı(firma) versiyonu]
    '''
    return re.sub(r"'.*$", "", line)

def sentiment_analyzer(sentence_i: str, firm: str):
    """
        Bu fonksiyon duygu analizi hesaplamalarında kullanılıyor.
    """

    firm_list.append(firm)
    sentiment_result = sentiment_analysis(sentence_i)
    sentiment = sentiment_result[0]['label']

    if sentiment == "LABEL_1":
        results.append({"entity": firm, "sentiment": "Olumlu"})
    elif sentiment == "LABEL_0":
        results.append({"entity": firm, "sentiment": "Nötr"})
    elif sentiment == "LABEL_2":
        results.append({"entity": firm, "sentiment": "Olumsuz"})
    elif sentiment == "LABEL_3":
        results.append({"entity": firm, "sentiment": "Olumlu"})
        results.append({"entity": firm, "sentiment": "Olumsuz"})

    return sentiment

def analyze_sentences(sentences: list[str], text: str):
    """
        Bu fonksiyon cümle analizi ve duygu-firma eşleşmesi analizi için kullanılıyor.

        text: str
        sentences: list[str] (split_sentences(text) 'a denk geliyor)

    """

    firm_list = []
    pattern = r'@(\w+(\_\w+)*)'
    matches = re.findall(pattern, text)
    result = [match[0] for match in matches]
    dandan.append(result)
    for i in dandan:
        for j in i:
            firm_list.append(j)
    for sentence in sentences:
        all_entities = []
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

                firm_list.append(cut_after_apostroph(entity['entity_name']))
                firmcount += 1
        # Tekrarlayan firma adlarının tek bir tanesini aldı.
        firm_list = list(set(firm_list))
        firmcount = len(firm_list)
        if firmcount == 1:
            for firm in firm_list:
                if firm in sentence:
                    sentiment = sentiment_analyzer(sentence, firm)
        else:
            inside_sentences = separate_sentences_via_conjunctions(sentence)
            if len(inside_sentences) == 1:
                for _sentence in inside_sentences:
                    for firm in firm_list:
                        if firm in _sentence:
                            sentiment = sentiment_analyzer(_sentence, firm)
            else:
              for i in inside_sentences :
                for _sentence in inside_sentences:
                    for firm in firm_list:
                        if firm in _sentence:
                          print(_sentence)
                          sentiment = sentiment_analyzer(_sentence, firm)

    unique_results = list(OrderedDict.fromkeys(tuple(sorted(d.items())) for d in results))
    unique_dict_results = [dict(t) for t in unique_results]

    response = {"entity_list": firm_list, "results": unique_dict_results}
    firm_list = []
    reset_result_vectors()

    with open('output.json', 'w', encoding='utf-8') as json_file:
        json.dump(response, json_file, ensure_ascii=False, indent=4)

    return response

def make_predict(text: str) -> dict[str, Any]:
    """
        Model'den çıktı almak için her şeyi tek fonksiyonda kolaylaştırmak amacıyla
        oluşturduk.

        text: str

        Returns:
            dict[str, Any] (Sonuçlar)

        Example:
        >>> make_predict('Turcell çok iyi bir şirket. İyi ki varsın @Turkcell.')
    """
    return analyze_sentences(split_sentences(text), text)

if __name__ == '__main__':
    while True:
        print(make_predict(input("Input: ")))
