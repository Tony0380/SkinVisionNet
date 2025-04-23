# SkinVisionNet: Deep Learning Techniques for Accurate Skin Lesion Classification

## Introduzione

Negli ultimi anni, la diagnosi assistita da intelligenza artificiale ha mostrato un enorme potenziale in ambito medico, in particolare nella dermatologia digitale. La classificazione automatica delle lesioni cutanee rappresenta una sfida cruciale per la prevenzione e la diagnosi precoce di malattie gravi come il melanoma, che, se individuato tempestivamente, può essere trattato con successo. Tuttavia, distinguere tra melanoma, nevi benigni e cheratosi seborroica richiede un'analisi accurata di dettagli visivi sottili, spesso non immediatamente riconoscibili.

Da questa esigenza nasce l’idea progettuale di **SkinVisionNet**, un sistema intelligente di supporto alla diagnosi dermatologica basato su tecniche di deep learning di ultima generazione. L'obiettivo è quello di realizzare un classificatore di immagini dermatoscopiche in grado di distinguere in modo affidabile tra le principali tipologie di lesioni cutanee, con un'architettura efficiente e scalabile.

Il cuore del sistema sarà costituito dal **Swin Transformer**, un modello particolarmente adatto per analizzare immagini ad alta risoluzione e con struttura visiva complessa. Grazie alla sua architettura gerarchica e all’uso di finestre scorrevoli (shifted windows), il modello riesce a catturare sia pattern locali (come la struttura dei bordi o le variazioni di pigmentazione) sia pattern globali (come la simmetria e la distribuzione del colore), risultando particolarmente efficace in ambito medico.

Per allenare il modello sarà utilizzato il dataset pubblico [Skin Lesion Analysis Toward Melanoma Detection](https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection), che fornisce immagini classificate in tre classi clinicamente rilevanti. Considerata la presenza di uno sbilanciamento tra le classi, verranno adottate tecniche specifiche di gestione dello squilibrio come il *weighted sampling* e la *focal loss*. Inoltre, il sistema sarà potenziato con strategie di *data augmentation* per aumentare la robustezza e la capacità di generalizzazione del modello.

A completamento del lavoro, il progetto prevede la valutazione delle prestazioni del sistema mediante metriche appropriate (accuracy, recall, F1-score, AUC) e l’eventuale integrazione di tecniche di explainability per rendere il modello interpretabile anche in contesti clinici.

Di seguito, la descrizione dettagliata del progetto, delle tecnologie scelte e delle motivazioni che guidano la nostra proposta.

## Obiettivi del progetto

- Sviluppare un classificatore automatico in grado di distinguere tra **melanoma**, **nevus** e **cheratosi seborroica** su immagini dermatoscopiche.
- Adottare tecniche avanzate di **fine-tuning** su modelli pre-addestrati con architetture transformer.
- Mitigare lo sbilanciamento tra classi attraverso l'uso combinato di *data augmentation*, *weighted sampler* e *focal loss*.
- Valutare le prestazioni mediante metriche robuste (accuracy, F1, recall, AUC).
- Esplorare possibili estensioni future verso explainability e integrazione clinica.

## Dataset

### Struttura Dataset

|                                | Training | Validation | Test |
| :----------------------------: | -------- | ---------- | ---- |
|       **Melanoma**       | 374      | 30         | 117  |
|        **Nevus**        | 1372     | 78         | 393  |
| **Seborrheic keratosis** | 254      | 42         | 90   |

[Link Dataset Kaggle](https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection/)

> ? [possibile secondo dataset](https://challenge.isic-archive.com/data/)

#### Normalizzazione

Standard ImageNet, perfetta per modelli pre-addestrati

### Tecniche per mitigare lo sbilanciamento

| Tecnica                | Serve per...                              | Quando usarla                        |
|------------------------|-------------------------------------------|--------------------------------------|
| **Data Augmentation**  | Aumentare la varietà e robustezza         | Sempre, su dataset piccoli/medi     |
| **Weighted Sampler**   | Bilanciare il numero di esempi per classe | Quando il dataset è sbilanciato     |
| **Focal Loss**         | Penalizzare di più gli errori gravi       | Quando il modello ignora classi minori |

#### Link utili

[Link Tutorial Utile](https://github.com/sara-kassani/Medical-Image-Processing/blob/master/2.%20Kaggle-Full%20Preprocessing%20Tutorial.ipynb)
[Link Tool Dataset](https://github.com/dvschultz/dataset-tools)

## Algoritmi di Deep Learning

Fine tuning su [Swin Transformer](https://github.com/microsoft/Swin-Transformer) piuttosto che [ViT-base-patch16](https://hiya31.medium.com/vision-transformer-vs-swin-transformer-a-conceptual-comparison-6502d9b949f2)

### Perché Swin Transformer?

| Caratteristica                         | ViT-base-patch16-224                            | Swin Transformer                                     |
|---------------------------------------|--------------------------------------------------|------------------------------------------------------|
| **Tipo di patch**                     | Patch fisse 16x16                                | Finestra scorrevole (shifted windows)               |
| **Architettura**                      | Transformer standard                             | Gerarchica (multi-scala)                            |
| **Gestione della scala**              | Limitata (non gerarchica)                        | Ottima per immagini con dettagli locali             |
| **Prestazioni su immagini mediche**  | Molto buone                                      | Spesso superiori nei task con dettagli fini         |
| **Efficienza computazionale**         | Buona, ma meno efficiente                        | Più efficiente a parità di prestazioni              |
| **Robustezza su piccoli dataset**     | Discreta, con buon fine-tuning                   | Alta, migliore generalizzazione                     |
| **Facilità d'uso per inizio progetto**| Semplice e veloce da usare                       | Richiede più configurazione                         |
| **Ideale per...**                     | Progetti con risorse limitate o dataset medi     | Classificazione con immagini complesse o dettagliate|

### In sintesi

| Fattore                             | Considerazione                           |
|------------------------------------|------------------------------------------|
|  **Numero totale immagini**       | ~2660 immagini:  OK                     |
|  **Sbilanciamento**               | Presente (ma gestibile)                  |
|  **Compito visivo complesso**     | Sì, serve riconoscere dettagli           |
|  **Necessità di contesto globale**| Sì, pattern e bordi nei/lesioni          |
|  **Potenza computazionale richiesta** | Media-alta                             |

## Sviluppi futuri

- Integrazione di tecniche di **explainability** (Grad-CAM, LIME) per rendere interpretabile la decisione del modello.
- Estensione a classificazione multi-label o segmentazione delle lesioni.
- Validazione clinica con esperti dermatologi su nuovi set di immagini reali.
- Realizzazione di una **web app prototipale** o un'interfaccia mobile di supporto alla diagnosi.

## Studenti

[Antonio Colamartino](https://github.com/Tony0380) | a.colamartino6@studenti.uniba.it

[Claudio De Candia](https://github.com/ClaudideCandia) | c.decandia15@studenti.uniba.it
