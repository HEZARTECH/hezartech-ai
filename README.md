![LOGO_PNG](https://raw.githubusercontent.com/HEZARTECH/.github/main/profile/assets/HEZARTECH_LOGO.png)

# HEZARTECH-AI ~ Teknofest Türkçe Doğal Dil İşleme

A natural language processing (NLP) project created via Python. (for Teknofest).
We attend with scenario type competetion. We uploaded our model to huggingface: 
https://huggingface.co/hezartech/hezartech-ai-teknofest-tddi-scenario

## Table of Contents
1. [Project Description](#project-description)
2. [Installation Instructions](#installation-instructions)
3. [Usage Instructions](#usage-instructions)
4. [Features](#features)
5. [Contributing](#contributing)
6. [License](#license)
7. [Authors](#authors)
8. [Acknowledgments](#acknowledgments)

## Project Description

HEZARTECH is a project designed to connect sentiment with firm names in input text. It analyzes the sentiment of the text and associates it with the mentioned firm names, providing valuable insights into public perception and sentiment towards specific companies. We made firm detection with Flair (DL-NER Model) and RegEx. Also we finetune BERTurk-128k-cased version with our 80K custom free dataset. And we connect these datas into together with a sentence matcher algorithm (which developed by us).

## Installation Instructions

To install the necessary dependencies for this project, run the following command:

```bash
$ pip3 install -r requirements.txt
```

## Usage Instructions

Ensure you have installed all the dependencies using the installation instructions above.
Run the main script to analyze sentiment and connect it with firm names in your input text.

```bash
$ python3 setup.py #(hit enter until program finish.)
```

```bash
$ cd src
$ python3 main.py
```

## Features

* Sentiment analysis of input text
* Association of sentiment with firm names
* Detailed output of sentiment scores and associated firms

## Contributing

We welcome contributions to improve HEZARTECH.AI.

## License
This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE.md) file for more details.

## Authors

* Yiğit GÜMÜŞ
* Burak Erdoğan
* Yusuf Hasan Onkun
* Yasemin Serçe

## Acknowledgments

We would like to thank everyone who made this competetion available.
Special thanks to Teknofest, Turkcell and Bilişim Vadisi. 😊
