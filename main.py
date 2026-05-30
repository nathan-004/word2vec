# https://becominghuman.ai/how-does-word2vecs-skip-gram-work-f92e0525def4

import string
from typing import Optional

import torch
import torch.nn as nn

def get_phrases(text) -> list[str]:
    """Renvoie une liste du texte contenu dans chaque phrases"""
    start = 0
    result = []
    
    for idx, car in enumerate(text):
        if car in string.punctuation:
            result.append(text[start:idx].strip())
            start = idx + 1
    
    return result

assert get_phrases("Test.test.") == ["Test", "test"]
assert get_phrases("Test?Test!") == ["Test", "Test"]
assert get_phrases("Test,Test!") == ["Test", "Test"]

def get_training_samples(text: str, window: int = 1) -> list[tuple[str, str]]:
    """
    Parameters
    ----------
    text: str Le corpus de texte d'entraînement
    window: int Nombre de mots représentant le contexte d'un mot
    
    Return
    ------
    list[tuple[str, str]] Liste de tuple sous la forme : [(mot1, mot2), ...]
    """
    text = text.lower()
    result = []
    
    for phrase in get_phrases(text):
        phrase = phrase.split(" ")
        for idx, word in enumerate(phrase):
            for add_idx in range(-window, window+1):
                if add_idx == 0 or not(0 <= idx + add_idx < len(phrase)):
                    continue
                result.append((word, phrase[idx + add_idx]))
    
    return result

assert get_training_samples("Ceci est un test.") == [("ceci", "est"), ("est", "ceci"), ("est", "un"), ("un", "est"), ("un", "test"), ("test", "un")]

def get_vocabulary(training_data: list[tuple[str, str]]) -> list:
    """Renvoie la liste des mots mélangés dans les données d'entraînement"""
    return list(set(
        [pair[0] for pair in training_data]
    ))

text_corpus = """Je ne connaîtrai pas la peur car la peur tue l’esprit.
La peur est la petite mort qui conduit à l’oblitération totale. 
J’affronterai ma peur. Je lui permettrai de passer sur moi, au travers de moi. Et lorsqu’elle sera passée, je tournerai mon œil intérieur sur son chemin. Et là où elle sera passée, il n’y aura plus rien. Rien que moi."""
VOCABULARY = get_vocabulary(get_training_samples(text_corpus))
print(VOCABULARY)

pairs = get_training_samples(text_corpus)

def get_vector(word: str, vocab: Optional[list] = None) -> torch.Tensor:
    """Retourne un vecteur 1, len(vocab) avec toutes les valeurs nulles sauf celle correspondant au mot donné"""
    if vocab is None:
        vocab = VOCABULARY

    return torch.tensor(
        [int(val == word) for val in vocab],
        dtype=torch.float
    )

class SkipGram(nn.Module):
    def __init__(self, embedding_dim: int, vocabulary: Optional[list] = None):
        super().__init__()

        if vocabulary is None:
            vocabulary = VOCABULARY

        self.vocabulary = vocabulary
        n = len(vocabulary)

        self.embeddings = nn.Embedding(n, embedding_dim)
        self.linear = nn.Linear(embedding_dim, n)

    def forward(self, x: list | torch.Tensor):
        x = self.embeddings(x)
        x = self.linear(x)

        return x

X = torch.tensor([VOCABULARY.index(p[0]) for p in pairs])
Y = torch.tensor([VOCABULARY.index(p[1]) for p in pairs])

model = SkipGram(10)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    logits = model(X)

    loss = loss_fn(logits, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item(), end="\r")
    
print()