# one-class_classifier

Tecniche di classificazione mono-classe per problemi di medicina legale.

## mltools.py

Questo modulo permette di effettuare delle Nested Cross-Validation sul dataset run-over, utilizzando diversi algoritmi: `OneClassSVM`, `LocalOutlierFactor` e `IsolationForest`, e utilizzando diverse tecniche per la riduzione delle dimensionalità: `PCA`, `TruncatedSVD`, `FastICA` e `EncoderMLP` (una classe per la riduzione presente nel modulo basata su un `MLPRegressor`).

## Prerequisiti

Assicurati di avere installato [Anaconda](https://www.anaconda.com/products/individual) sul tuo sistema. Anaconda è una distribuzione di Python che include una serie di pacchetti scientifici e strumenti di gestione degli ambienti virtuali.

## Configurazione dell'Ambiente

1. **Creazione di un Ambiente Virtuale:**

   Apri il terminale (o Anaconda Prompt su Windows) e crea un nuovo ambiente virtuale con Python 3.8 (o altra versione supportata):

   ```sh
   conda create --name nested_cv_env python=3.8

2. **Attivazione dell'Ambiente Virtuale**

   ```sh
   conda activate nested_cv_env

3. **Installazione delle dipendenze**

   Apri il terminale e installa le dipendenze con [pip](https://pypi.org/project/pip/):

   ```sh
   pip install numpy
   pip install pandas
   pip install sklearn

## Utilizzo del modulo

1. **Importa il modulo**

   Assicurati che il modulo importato sia nella tua directory di lavoro:

   ```python
   from mltools import load_csv

2. **Caricamento del Dataset**

   Carica il dataset relativo:

   ```python
   from mltools import load_csv

   X = load_csv(reduced=True)
   ```

3. **Esecuzione della Nested Cross-Validation**
   
   Esegui la Nested Cross-Validation specificando i parametri necessari:

   ```python
   from mltools import load_csv, nested_cv_svm, get_random_seed
   from sklearn.decomposition import PCA

   X = load_csv(reduced=True)
   random_seed = get_random_seed()

   nested_cv_svm(X, random_seed, PCA, 'nested_cv_svm_pca.log', n_components=85)
   ```

