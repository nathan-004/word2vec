# https://becominghuman.ai/how-does-word2vecs-skip-gram-work-f92e0525def4

import string
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

text_corpus = """Le chat mange une souris.
Le chien mange de la viande.
Le chat dort sur le canapé.
Le chien dort dans le jardin.
L'oiseau vole dans le ciel.
L'oiseau chante le matin.
Le poisson nage dans l'eau.
L'homme mange une pomme.
La femme mange une banane.
L'enfant mange du pain.
Le chat regarde le chien.
Le chien regarde le chat.
La voiture roule sur la route.
Le bus roule en ville.
Le vélo roule rapidement.
Le soleil brille dans le ciel.
La pluie tombe sur la ville.
Le vent souffle fort.
Le professeur enseigne aux élèves.
L'élève apprend la programmation.
L'élève écrit du code.
Le robot exécute un programme.
Le robot apprend à parler.
Le programme calcule un résultat.
Le chien court dans le parc.
Le chat court après la souris.
Le poisson mange des algues.
L'oiseau mange des graines.
La femme lit un livre.
L'homme lit un journal.
L'enfant lit une histoire.
Le chat est petit.
Le chien est grand.
Le poisson est rapide."""

VOCABULARY = get_vocabulary(get_training_samples(text_corpus, window=2))
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

model = SkipGram(100)

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

def most_similar(word, k=5, vocabulary: Optional[list] = None):
    if vocabulary is None:
        vocabulary = VOCABULARY

    idx = vocabulary.index(word)
    v = model.embeddings.weight[idx]

    sims = F.cosine_similarity(v.unsqueeze(0), model.embeddings.weight)

    best = torch.topk(sims, k+1).indices[1:]
    return [vocabulary[i] for i in best]

print(most_similar("homme"))