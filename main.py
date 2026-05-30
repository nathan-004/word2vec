# https://becominghuman.ai/how-does-word2vecs-skip-gram-work-f92e0525def4

import string
from typing import Optional

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

text_corpus = "Ceci est un test."
VOCABULARY = get_vocabulary(get_training_samples(text_corpus))
print(VOCABULARY)

def get_vector(word: str, vocab: Optional[list] = None) -> list:
    """Retourne un vecteur 1, len(vocab) avec toutes les valeurs nulles sauf celle correspondant au mot donné"""
    if vocab is None:
        vocab = VOCABULARY

    return [int(val == word) for val in vocab]

print(get_vector("test"))