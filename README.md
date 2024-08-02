# Descriptif du répertoire 

Dans le cadre de mon stage, le but à été de combiner des données tabulaires et textuelles afin de pouvoir 
classifier ou prédire une variable Y et de pouvoir justifier la prédiction en langage naturel.

Ce projet a pour objectif de construire un modèle qui a 2 objectifs: 

- Classifier les overall ratings de 0 à 5
- Justifier cette prédiction à travers une génération de texte (positive, négative ou neutre).

Le code a été inspiré de cet [article](https://aclanthology.org/P19-1560.pdf).

# Organisation du répertoire 

## Le répertoire est composé de 4 dossiers:

- Datasets: Les données _Train_, _Validation_, _Test_ prétraitées. (Si vous voulez plus de détails sur le prétraitement voir rapport de stage). Ces données proviennent de [l'article](https://aclanthology.org/P19-1560.pdf), appelée _PCMAG_. Cela contient une liste de produits technologiques, avec des notes, et des commentaires positifs negatifs et neutres. 

- Programs: Dossier avec les différents programmes python. Ils sont classés dans l'ordre chronologique (01 le premier, et le 04 le dernier).

- _ids_attention_masks: Dossier contenant les 'ids' et les 'attention masks' des données train, val et test afin d'éviter de les charger à chaque lancement. 

- Tokenizer: Ce dossier contient le tokenizer _bert uncased_ (vocab.txt, tokenizer config, special token map), cela peut permettre notamment de le charger en local au lieu de le requeter avec _hugging face_.

- Les 2 fichiers restants sont le _classifier préentrainé finetuné_ sur nos données, et le _requirements.txt_ qui permet de ne pas avoir de problème de librairies et de versions.


## Prérequis:

- Python version: 3.12+

- CUDA version: 12.2+

## Programs/04_LLMGen:

Le code n'est pas terminé pour ce fichier, pour le poursuivre voici le [code](https://medium.com/@mohitdulani/fine-tune-any-llm-using-your-custom-dataset-f5e712eb6836) 
source dont je me suis inspiré. 

## Installation:

```bash
git clone https://gitlab.pleiade.edf.fr/gamme-conso-client/nle
```

```bash
pip install -r requirements.txt
```

## Entrainement des modèles: 

1) Autoriser l'accès à internet 

```bash
source oproxy2.sh
```
2) Lancer l'entrainement 

```bash
exemple: python 03_CVAE.py
```










