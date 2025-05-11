# SkinVisionNet: Deep Learning Techniques for Accurate Pigmented Skin Lesion Classification

## Executive Summary

SkinVisionNet è un progetto di deep learning che utilizza tecnologie all'avanguardia per classificare con precisione le lesioni pigmentate della pelle. L'implementazione utilizza un'architettura di Swin Transformer per affrontare un problema di classificazione multi-classe di immagini dermatologiche, con particolare attenzione alla gestione dello sbilanciamento delle classi. Questo report descrive dettagliatamente le componenti tecniche dell'implementazione.

## 1. Setup e Importazione dei Package

Il progetto utilizza diversi framework e librerie Python, tra cui:

- **PyTorch** e **torchvision**: Framework principale per la creazione e addestramento di reti neurali
- **timm** (PyTorch Image Models): Libreria per accedere a modelli pre-addestrati come Swin Transformer
- **numpy** e **pandas**: Per l'elaborazione e la manipolazione dei dati
- **matplotlib**: Per la visualizzazione dei risultati
- **sklearn**: Per le metriche di valutazione
- **tqdm**: Per monitorare l'avanzamento dell'addestramento

Il codice verifica anche la disponibilità di CUDA per accelerare l'addestramento su GPU, impostando automaticamente il dispositivo appropriato (GPU o CPU).

## 2. Configurazione del Dataset

Il dataset è organizzato in una struttura di directory standard:
- `skin-lesions/train`: Immagini per l'addestramento
- `skin-lesions/valid`: Immagini per la validazione
- `skin-lesions/test`: Immagini per il test

Il codice verifica l'esistenza di queste directory per garantire che il dataset sia accessibile prima di procedere.

## 3. Data Augmentation e Preprocessing

Per migliorare la generalizzazione del modello e prevenire l'overfitting, il progetto implementa diverse tecniche di data augmentation:

**Per il training set:**
- Ridimensionamento a 224x224 pixel (standard per molti modelli pre-addestrati)
- Flip orizzontali e verticali casuali (probabilità 50%)
- Rotazioni casuali (fino a 20 gradi)
- Modifica casuale di luminosità, contrasto, saturazione e tonalità
- Trasformazioni affini casuali (traslazione e scala)
- Normalizzazione utilizzando media e deviazione standard di ImageNet

**Per validation e test set:**
- Solo ridimensionamento e normalizzazione per mantenere la coerenza

Il preprocessing include anche la configurazione dei dataset utilizzando `ImageFolder` di torchvision, che automaticamente organizza le immagini in classi basate sulla struttura delle directory.

## 4. Gestione dello Sbilanciamento delle Classi

Per affrontare il problema dello sbilanciamento delle classi (alcune condizioni dermatologiche sono più rare di altre), vengono implementate due strategie principali:

1. **Weighted Random Sampler**: Questo approccio assegna probabilità di campionamento inversamente proporzionali alla frequenza delle classi, garantendo che durante l'addestramento tutte le classi siano rappresentate equamente indipendentemente dalla loro frequenza nel dataset originale.

2. **Calcolo dei pesi delle classi**: Viene calcolata una distribuzione di pesi inversamente proporzionale al conteggio delle classi, che verrà utilizzata nella funzione di loss per penalizzare maggiormente gli errori sulle classi meno rappresentate.

## 5. Preparazione dei Data Loaders

I data loader sono configurati per gestire il caricamento efficiente dei dati durante l'addestramento:

- **Batch size**: 32 immagini per batch
- **Weighted sampler**: Applicato al training loader per bilanciare le classi
- **Numero di worker**: Adattato automaticamente al sistema operativo (ridotto o disattivato per Windows)

Questa configurazione garantisce un'efficiente alimentazione dei dati alla GPU durante l'addestramento, mantenendo al contempo un bilanciamento tra le diverse classi.

## 6. Visualizzazione di Immagini di Esempio

Il codice include funzionalità per visualizzare campioni dal dataset, permettendo di:
- Verificare visivamente le immagini caricate
- Visualizzare la denormalizzazione applicata per riportare le immagini a valori RGB standard
- Confermare le etichette associate a ciascuna immagine

Questa fase è fondamentale per il debugging e per comprendere meglio le caratteristiche del dataset.

## 7. Implementazione della Focal Loss

Per migliorare ulteriormente la gestione dello sbilanciamento delle classi, viene implementata la Focal Loss, una funzione di loss avanzata che:

- Assegna un peso maggiore agli esempi difficili da classificare
- Riduce l'impatto degli esempi facilmente classificati sul gradiente
- Include un parametro gamma che modula l'effetto di downweighting degli esempi facili
- Supporta anche pesi per classe per gestire lo sbilanciamento

La Focal Loss è particolarmente efficace nei dataset sbilanciati in quanto si concentra automaticamente sugli esempi più informativi per l'apprendimento.

## 8. Definizione del Modello

Il progetto utilizza un'architettura Swin Transformer ("swin_tiny_patch4_window7_224") per la classificazione delle lesioni cutanee:

- **Swin Transformer**: Un'architettura all'avanguardia basata sui transformer che utilizza finestre di attenzione scorrevoli gerarchiche
- **Transfer Learning**: Il modello viene pre-caricato con pesi pre-addestrati su ImageNet
- **Fine-tuning**: L'ultimo livello viene adattato al numero specifico di classi nel dataset dermatologico

L'uso di un modello pre-addestrato consente di sfruttare le caratteristiche generali apprese su grandi dataset e applicarle al dominio più specifico delle immagini dermatologiche.

## 9. Configurazione del Training

L'addestramento è configurato con i seguenti iperparametri:

- **Funzione di loss**: Focal Loss con gamma=2.0 e pesi delle classi calcolati
- **Ottimizzatore**: AdamW con learning rate di 3e-4 e weight decay di 1e-5
- **Scheduler**: CosineAnnealingLR che riduce gradualmente il learning rate secondo una curva cosinusoidale
- **Numero epoche**: 15

Questi parametri sono stati selezionati per bilanciare la velocità di convergenza, la stabilità dell'addestramento e la prevenzione dell'overfitting.

## 10. Funzioni di Training e Valutazione

Il codice implementa due funzioni principali per gestire l'addestramento e la valutazione:

**`train_one_epoch`**: 
- Imposta il modello in modalità training
- Esegue forward pass, calcolo della loss, backward pass e ottimizzazione per ogni batch
- Traccia e restituisce loss e accuratezza

**`evaluate`**: 
- Imposta il modello in modalità valutazione
- Calcola loss e accuratezza senza aggiornare i pesi
- Raccoglie tutte le predizioni e le etichette vere per un'analisi dettagliata
- Restituisce metriche di performance

Entrambe le funzioni includono barre di avanzamento con tqdm per monitorare il progresso in tempo reale.

## 11. Training Loop

Il ciclo di addestramento implementa una serie di best practice:

- **Early stopping implicito**: Salva il modello solo quando migliora l'accuratezza di validazione
- **Monitoraggio delle metriche**: Traccia loss e accuratezza sia su train che validation set
- **Gestione degli errori**: Include gestione delle eccezioni e interruzione manuale
- **Visualizzazione in tempo reale**: Mostra statistiche per ogni epoca

Questo approccio garantisce un addestramento efficiente e monitorabile, con la possibilità di rilevare e correggere problemi durante l'esecuzione.

## 12. Visualizzazione dei Risultati del Training

Il progetto include strumenti di visualizzazione per analizzare l'andamento dell'addestramento:

- **Grafici di loss**: Mostra l'andamento della funzione di loss su training e validation set
- **Grafici di accuratezza**: Visualizza l'evoluzione dell'accuratezza per entrambi i set
- **Confronto train/val**: Permette di identificare visivamente eventuali segni di overfitting

Questi grafici sono essenziali per comprendere il comportamento del modello durante l'addestramento e per identificare potenziali problemi.

## 13. Valutazione sul Test Set

La fase finale valuta il modello migliore sul test set, fornendo una valutazione completa delle prestazioni:

- **Metriche generali**: Loss e accuratezza
- **Classification report**: Precisione, recall e F1-score per ogni classe
- **Matrice di confusione**: Visualizzazione dettagliata di previsioni corrette e errate per classe

Queste metriche offrono una visione completa dell'efficacia del modello su dati mai visti, con particolare attenzione alla performance per ciascuna classe individuale.

## Conclusioni

SkinVisionNet rappresenta un'implementazione completa e robusta per la classificazione delle lesioni cutanee, utilizzando tecniche moderne di deep learning come:

1. **Architettura avanzata**: Swin Transformer per una migliore estrazione delle caratteristiche
2. **Gestione dello sbilanciamento**: Weighted sampling e Focal Loss
3. **Augmentation estensiva**: Tecniche multiple per migliorare la generalizzazione
4. **Monitoraggio completo**: Valutazione continua con metriche dettagliate

Il modello è stato progettato per affrontare le sfide specifiche della classificazione dermatologica, con particolare attenzione all'accuratezza diagnostica e alla capacità di generalizzazione su diverse condizioni cliniche.
