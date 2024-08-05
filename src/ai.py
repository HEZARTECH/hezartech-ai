"""from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import pandas as pd

from torch.utils.data import Dataset
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

model_name: str = "berturk-teknofest-tddi-sceneario-finetune"

df: pd.DataFrame = pd.read_csv('../model/dataset.csv')

label_mapping: dict[str | None, int] = {
    None: 0,
    "positive": 1,
    "negative": 2,
    "positive|negative": 3,
}

# Etiketleri güncelle
df['label'] = df['value'].map({
    1: 'positive',
    3: 'positive|negative',
    2: 'negative',
})

# Sadece gerekli sütunları tutun
df = df[['text', 'label']]

tokenizer = BertTokenizer.from_pretrained(f'HEZARTECH/{model_name}')
model = BertForSequenceClassification.from_pretrained(
    f'HEZARTECH/{model_name}',
    num_labels = len(label_mapping)
)

sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

class HezartechDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.dataframe.set_index('index', inplace=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, id):
        return self.dataframe.loc[id]


def get_sentiment(text):
    prediction = sentiment_analysis(text)[0]['label']
    if prediction == 'LABEL_0':
        return 'Olumsuz'
    elif prediction == 'LABEL_1':
        return 'Olumlu'
    elif prediction == None:
        return 'Tarafsiz'
"""
"""
def sentiment_analysis(text: str = "Turkcell çok iyi. TurkTelekom tam bir rezaletti. Ama iyi ki değiştirdim."):
    return {
        'entity_list': ['Turkcell', 'TurkTelekom'],
        'results': [
            {
                'entity': 'Turkcell',
                'sentiment': 'Olumlu'
            },
            {
                'entity': 'TurkTelekom',
                'sentiment': 'Olumsuz'
            }
        ]
    }

"""
