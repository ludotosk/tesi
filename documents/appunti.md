# Papers da citare



- Network Intrusion Detection Based on Explainable Artificial Intelligence
  
  - citazione [4] dice che il 67% degli esperti non si fida dall'ai
  
  - spiegano per filo e per segno tutto dalla pulizia dei dati a come hanno fatto il modello, però non fanno vedere una rappresentazione grafica della spiegazione del modello
  
  - usano kdd9 e poi spiegano con shap ma non usano tutti gli shap
  
  - usano kernel shap

- ROULETTE
  
  - gdpr
  
  - parlano della mancanza di visualizzazione e di come molti studi si basano solo sull'aumento delle performance
  
  - ci sono studi su come usare modelli surrugati per spiegare le scelte di NID
  
  - c'è uno studio che usa shap per spiegare il risultato
  
  - ci sono esempi di adv attack per confondere il NID
  
  - convertono il vettore con le informazioni in una matrice stile quella delle immagini
  
  - anche questo paper spiega nel dettaglio come ha misurato i risultati e come ha fatto il tuning della rete

- PAIRED è quello che tiene conto della velocità dell'algoritmo
  
  - non mi serve perché parla di android

- Realguard parla di perché ha senso fare un ids per l'iot
  
  - citano il fatto che siano capaci di processare 10600 pacchetti al secondo con un raspberry
  
  - nell'intro cita due studi che spiegano come i device iot saranno importantissimi
  
  - parlando dnn citando altri studi dove però non viene fatto fine tuning
  
  - hanno pubblicato il docker (no link provided, su richiesta dicono)
  
  - hanno creato un mvp
  
  - in questo studio parlano dei pcap di cicids 2017, cosa che negli altri dataset manca e apre alla possibilità di fare mvp come hanno fatto loro
  
  - fanno under sampling e usano k-fold
  
  - mancano tutti i web attack, nessuna spiegazione fornita
  
  - fa anche i test su un raspberry con una rete simulata con i pcap, sono tra i pochissimi a farlo
  
  - e fa vedere come far una piccola CNN con 5 layers di modo che utilizzi poche performance
  
  - lamentano la mancanza di altri dataset ma c'è hikari

- A systematic literature review of methods and datasets for anomaly-based network intrusion detection (in realtà non mi serve)
  
  - seguono la metodologia SLR, systematic literature review
  - dicono che il loro è l'unico lavoro di literature review riproducibile
  - spiega perché hanno usato scopus
  - cita tutte le strategie usate dai vari papers sul tema, per esempio come fare feature selection ecc.

- Generating Network Intrusion Detection Dataset Based on Real and Encrypted Synthetic Attack Traffic (questo è il paper che presenta hikair-2021)
  
  - comincia facendo notare come la maggior parte dei datasets non tiene conto che oggi i dati sono quasi tutti cifrati
  
  - dice che il payload non è utile ma comunque loro lo hanno messo perché in futuro potrebbe rivelarsi utile, in ogni caso è cifrato quindi non va preso in considerazione
  
  - si concetrano su attacchi a livello 7 dello stack iso/osi cioè livello applicazione perché 80% degli attacchi arrivano li. Per farlo hanno preso i tre CMS che valgono il 50% del market share.
  
  - l'unica cosa che manca è l'utilizzo dei plugin più famosi di queste piattaforme
  
  - il traffico di background è stato anonimizzato e potrebbe contentere attacchi
  
  - Hanno fatto una tabella in cui testano il dataset con KNN, MLP, SVM e RF però manca tutto il procedimento e manca anche la spiegabilità. Inoltre manca anche il peso nell'eseguire gli algoritmi.
  
  - ha le stesse feature di CICIDS-2017 ma ne sono state aggiunte altre 6, bisogna controllare quali attacchi sono stati fatti in CICIDS ed eventualmente vedere se l'ids fatto con HIKARI funziona con CICIDS
  
  - c'è un riassunto di tutto il lavoro svolto nella conclusione
  
  - cercare zeek per vedere cosa significhino quelle feature extra
  
  - dicono che kdd9 è obsoleto
  
  - dicono che hanno usato 80 feature da CICIDS
  
  - il traffico di anonimizzato è quello classificato come di backgroud
  
  - tutti gli ip esterni, cioè quelli della lista ALEXA sono benigni
  
  - spiegano anche come abbiano fatto a fare le labels

- CICIDS-2017 Dataset Feature Analysis With Information Gain for Anomaly Detection
  
  - dicono che information gain è il metodo più usato per fare feature selection
  
  - dicono che i risultati sono migliorati facendo feature selection
  
  - sottolineano come manchi il tempo computazionale nei vari test
  
  - parlano di FMIFS che è un medodo di fare feature selection testato da un altro paper. Ci altri papers che hanno utilizzato altre tecniche
  
  - enfatizzano il fatto che manchino ancora studi su dataset di grosse dimensioni
  
  - spiega che alcune feature che ci sono qui mancano altrove e per questo questo data set è meglio
  
  - parlano di rimuovere features ridondanti (in realtà è una che è finita lì per errore)
  
  - usano solo il 20% dei dati del dataset
  
  - dicono che il miglior modo per fare feature selection è quello di usare l'entropia, per ipotesi si potrebbe fare lo stesso usando i modelli ad albero
  
  - mostrano il loro setup hardware e software per eseguire il tutto usano un programma basato su java
  
  - usano delle quantità predefinite di peso di ciascuna feature per decidere quante tenerne. questo potrebbe velocizzare di tanto il mio processo
  
  - dicono che fare RFCE è svantaggioso perché richiede un sacco di tempo
  
  - usano 10-fold
  
  - fanno vedere come RF e RT funzionino meglio degli altri modelli testati
  
  - dicono che i modelli ad albero sono i migliori
  
  - fanno notare come gli infiltatrion attack non vengono rilevati, essendo solo 36 è normale. Quindi andrebbero tolti o comunque maggiorati con smote e poi andrebbe fatto un sampling stratificato.

- Evaluating Unbalanced Network Data for Attack Detection
  
  - qui citano l'hardware usato, citano come hanoo fatto a fare il tune del modello e in più come hanno selezionato le feature
  
  - qui in apertura fanno un confronto con cicids e hikari ma non testano l'uno con l'altro, però fanno training con gli stessi parametri in entrambi i dataset
  
  - parlando anche di come fare lo split e come fare undersampling dei dati
  
  - il problema grosso è qui fanno la classificazione sui singoli attacchi, che va bene per identificare un attacco specifico ma se vuoi fare un ids devi proteggerti da attacchi mai visti e questa cosa non viene considerata

- Network Intrusion Detection Packet Classification with the HIKARI-2021 Dataset: a study on ML Algorithms
  
  - leggere parte critiche
  - usano chi-squared per rimuovere le feature

- A detailed analysis of CICIDS2017 dataset for designing Intrusion Detection Systems
  
  - nel intro spiega perché questo dataset sia valido
  
  - creano un subset del dataset e fanno anche relabeling (qui bisogna fare attenzione)
  
  - il paper più citato che analizza CICIDS2017
  
  - loro parlano di perché sia importante fare il merge di tutti i papers
  
  - bisogna eliminare qualche riga con i dati mancanti
  
  - fanno notare come avere un dataset così sbilanciato sia un male perché durante la fase di training potrebbe non essere preso in cosiderazione qualche dato
  
  - rispetto a hikari mancano 3 features, una è la label binaria e le altre sono la porta di partenza e gli ip
  
  - cita un paper che fa una valutazione di ids del passato

- Improving AdaBoost-based Intrusion Detection System (IDS) Performance on CIC IDS 2017 Dataset (191)
  
  - parlano di smote come possibile soluzione (questo dataset ha una minority class più piccola di hikari)
  
  - Usano la pca che non va bene per il mio caso perché si perde la parte di explainability e poi  ci va di più ad elaborare il dato
  
  - parlano del fatto che si possa migliorare il risultato ottenuto nel paper originale di cic ids, potrei farlo meglio col mio metodo di sampling
  
  - loro usano ensemble feature selection, io però userò shap
  
  - Fanno l'analisi solo sugli attacchi ddos
  
  - cita un paper a caso per giustificare l'utilizzo di una confusion matrix
  
  - descrivono il setup e anche la configurazione usata
  
  - raddoppiano la minority class del 200%
  
  - non hanno fatto fit una volta e poi transform per con la pca
  
  - cita un paper che dice quali sono le caratteristiche di un buon paper per fare gli ids
  
  - citano le feature selezionate e come le hanno selezionate
  
  - hanno fatto la cross validation 5 volte per far si che fosse più precisa

- Benchmarking of Machine Learning for Anomaly Based Intrusion Detection Systems in the CICIDS2017 Dataset
  
  - nell'introduzione dice quali sono i migliori modelli e dicono di averne usati 31
  
  - si parla di AIDS cioè i gli ids fatti con i modelli di machine learning
  
  - parlano di multi class
  
  - c'è un grafico dei dataset più usati fanno vedere che cic ids 2017 è tra questi
  
  - dicono che il dataset darpa è stato usato nel 42% degli studi e quello kdd cup nel 28%
  
  - spiega perché usare un dataset più aggiornato dei due precedentemente menzionati sia meglio
  
  - spiegano la differenza tra multi classification e quella binary
  
  - spiegano che accuracy non va sempre bene siccome i dataset sono sbilanciati
  
  - cita papers che guardano anche ai tempi di classificazione
  
  - c'è un grafico che fa vedere quanto vengano usati i dataset in letteratura
  
  - spiegano nel dettaglio come funzioni una random forest
  
  - introducono cicids2017
  
  - usano z-score per normalizzare i dati
  
  - nel confronto si vede come la mlp che chiamano ann funzioni meglio e sia anche più veloce

- Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization
  
  - cominciano citando perché i vecchi dataset non sono più affidabili
  
  - citano perché avere dei dataset aggiornati sia importante
  
  - citano cicflowmeter
  
  - hanno usato random forest per far vedere quali siano le feature più importanti
  
  - provano a far girare dei modelli per vedere come vanno
  
  - calcolano anche il tempo d'esecuzione
  
  - mostrano come random forest sia il più veloce
  
  - dicono che facendo il compromesso tra i gli score ottenuti e il tempo d'esecuzione random forest ne esca vincitore

- A Unified Approach to Interpreting Model Predictions
  Scott
  
  - creano un approccio unico per sei differenti metodi esistenti, in realtà poi sono aumentati
  
  - si spiega cos'è lime cioè una funzione lineare
  
  - come deep lift possa essere trasformato in funzione lineare
  
  - parlando di layer-wise relevance propagation che è simile a deep lift
  
  - presentano i classic shapley value, che per funzionare devono ogni volta fare il train di un modello togliendo una feature
  
  - indroducono kernel shap, cioè lime + shapley values
  
  - il paper è famoso per aver introdotto kernel shap, che è più veloce di classic shap
  
  - introducono dei model specific explainer perché sono più veloci di kernel shap
  
  - la figura 3 mostra come kernel shap sia più fedele si sampling shap e di lime rispetto agli shapley value

- Consistent Individualized Feature Attribution for Tree Ensembles
  
  - introduzione di tree shap, un algoritmo efficient in grado di spiegare l'intero dataset
  
  - illustrano pure i alcuni plot

- From Local Explanations to Global Understanding with Explainable AI for Trees
  
  - parte spiegando perché fare reduction usando metodi come la pca implica una perdita in termini di spiegabilità (cui bisogna agganciarsi al paper dove dicono che molti esperti non si fidano degli ids)
  
  - dicono che i modelli ad albero sono i non lineari più popolari
  
  - manca letteratura sulle local explanation
  
  - sappiamo come i modelli fanno le predizioni
  
  - dice che gli altri modelli proposti fin'ora ci mettono anni e quindi ha senso concentrarsi su quelli ad albero
  
  - dice come sommare diverse spiegazioni locali aiuti a spiegare tutto il modello
  
  - i modelli ad albero vano meglio per dataset tabulari
  
  - dicono come si possa spiegare un albero solo ma non un ensemble e poi parlano di un altro algoritmo saabas che provano essere inferiore a shap.
  
  - dice che il punteggio non va solo alle singole features ma anche alle paia di features
  
  - dicono come mettere assieme diverse spiegazioni sia il metodo migliore per fare una spiegazione globale
  
  - parla di come le tech companies sbaglino
  
  - hanno usato 200 background data per avere low variance
  
  - nella figura 3 fanno vedere come sia meglio tree shap di altre soluzioni compreso kernel shap
  
  - confronto tra usare shap per determinare l'importanza delle feature e altri metodi
  
  - questo paper lo posso aggiornare con fast tree shap

- Fast TreeSHAP: Accelerating SHAP Value Computation for Trees
  Jilei
  
  - fast treeshap v1 1,5 volte più veloce a parità di memoria
  
  - fast treeshap v2 2,5 volte più veloce ma con più memoria
  
  - spiega perché shap è il più raccomandato
  
  - dice che l'algoritmo originale patisce sopra le milioni di osservazioni e con alberi di profondità maggiore di 10
  
  - dice che l'alternative esistono ma si concentrano nel parallelizzare i calcoli e non semplificare

- Classification and Explanation for Intrusion Detection System Based on Ensemble Trees and SHAP Method
  
  - citano un paper dove si parla del fatto che la mancanza di spiegazioni è un problema [9]
  
  - usano una libreria no code per lo sviluppo
  
  - citano tutte le volte che una random forest è stata applicata per lavorare gli ids
  
  - dicono che andrebbe usato tutto il dataset, solo che così facendo i risultati peggiorano
  
  - cita i paper che usando xai hanno migliorato gli ids
  
  - spiega che la random forest fa la media dei risultati degli alberi
  
  - dicono hanno automatizzato il processo di fine tuning
  
  - fanno vedere come la random forest sia la migliore
  
  - fanno vedere le specs del pc

- True to the Model or True to the Data?
  
  - spiega come interventional sia true to the model e observational true to the data
  
  - il vantaggio di quello observational è che se anche sbaglia con le features in realtà è capace di gestire anche dati mai visti che sono esegerati
  
  - è il paper dove presentano linear shap
  
  - dicono che generalmente true to the model è meglio perché di solito è il modello che si vuole spiegare
  
  - spiega che invece se vogliamo capire i dati true to the data sia meglio
  
  - se le features non sono correlate tra loro allora il risultato è lo stesso

- Feature Drift Aware for Intrusion Detection System Using Developed Variable Length Particle Swarm Optimization in Data Stream
  
  - sottolineano come con la feature selection si possano migliorare le performance
  
  - parla degli ensemble

- Understanding variable importances in forests of randomized trees
  
  - mda per essere usato come feature importance potrebbe avere dei bias ma non tutti gli studi sono concordi
  
  - il mdi in un singolo albero è calcolato con una sommattoria dell'impurità di una singola variabile su tutti i nodi

- Anomaly process detection using negative selection algorithm and classification techniques
  
  - usano logistic regression, k-nn, rf e dt
  
  - fanno feature selection con weka (java)
  
  - introducono due diversi tipi di ids
  
  - fanno notare come fare feature selection sia importante
  
  - selezionano le feature che sono più correllate con gli attacchi
  
  - in sostanza non fanno hyper tuning, ma solo feature selection e poi fanno under sampling con quel algoritmo chiamato nsa

- Effectiveness of Machine Learning based Intrusion Detection Systems
  
  - provano diverse metriche tra cui il tempo di training ecc
  - fanno notare come in quasi tutti gli studi su cicids non vengano usati tutti gli attacchi
  - dice che due colonne hanno valori nan o infiniti
  - hanno fatto feature selection usando la random forest
  - fanno vedere che la knn è la migliore ma è lenta quindi inadatta al mondo reale

- Dual-IDS: A bagging-based gradient boosting decision tree model for network anomaly intrusion detection system (42 citazioni il più citato su 32 di hikari)
  
  - usano tre dataset
  
  - non usano hikari completo
  
  - hanno usato hold out 80/20 per testare il dataset senza stratificazione
  
  - giustificano la scelta di tenere il dataset sbilanciato usano una formula chiamata mcc
  
  - forniscono i parametri per il fine tuning (provare a usare stessi parametri)
  
  - non fanno feature selection
  
  - da prendere in esempio per comparare i modelli
  
  - dicono che i falsi positivi sono da ridurre
  
  - dicono che alcuni progressi sono stati fatti dalle neural network per riuscire a battere i metodi ensemble

- Intrusion detection system for wireless mesh network using multiple support vector machine classifiers with genetic-algorithm-based feature selection (195 citazioni)
  
  - sfruttano i pcap di cicids ma non il csv quindi non è paragonabile al mio

- A fast network intrusion detection system using adaptive synthetic oversampling and LightGBM (113)
  
  - usa lightgbm e adasyn per fare classificazion e oversampling
  
  - non notano le colonne duplicate o vuote e non parlano delle righe duplicate
  
  - fanno over sampling su Web Attack, Bot, Infiltration and Heartbleed
  
  - mancano auc, f1 e mcc

# Critiche ai vari papers

- Evaluating Unbalanced Network Data for Attack Detection
  
  - non fanno il fine tuning della random forest e così facendo ho ottenuto un punteggio più alto del loro, non solo ma questo mi ha permesso di riuscire a ottenere i risultati che loro non sono stati in grado di ottenere
  - bisogna ricordare di menzionare il rapport 1:1 tra attacco e non attacco altrimenti l'f1 sale ma l'auc scende a causa dei false positive che sono più bassi

- Network Intrusion Detection Packet Classification with the HIKARI-2021 Dataset: a study on ML Algorithms
  
  - anche qui fanno l'errore di fare la classificazione multipla e dicono che l'undersampling non aiuta ma ho già mostrato come non sia vero
  
  - il grafico che mostrano delle correlazione è sbagliato, o quanto meno è diverso alla luce dell'aggiornamento
  
  - usano chi-square per fare undersampling però come dice l'altro paper shap è meglio
  
  - parlano di 83 feature ma non è possibile, sono 81 ma tenendo conto di traffic_category e di Label di cui delle due ne va usata una. Lui ha usato anche 'Unnamed: 0.1', 'Unnamed: 0' prova il grafico e il paper successivo
  
  - non dicono come hanno fatto il fine tuning
  
  - hanno fatto random sampling anziché stratificato

- From explanations to feature selection: assessing SHAP values as feature selection mechanism
  
  - qui loro fanno un confronto in termini di perfomance tra SHAP e altri algoritmi però non tengono conto del nuovo algoritmo proposto da LinkedIn. Quindi potrei eventualmente portare questo test.
  - introducono prima kernel shap e poi tree shap ma fanno confusione e non spiegano la differenza tra i due
  - non parlano nemmeno della differenza tra true to data e true to the model, il che potrebbe portare a un miglioramento del loro paper

- Benchmarking of Machine Learning for Anomaly Based Intrusion Detection Systems in the CICIDS2017 Dataset
  
  - parlano del fatto che la f1 è utile, ma in realtà ho visto con l'altro dataset che se la classe più grande è positiva può essere che f1 score sbagli e sia maggiore
  - prendono in cosiderazione solo 4 classi per giunta neanche le più grandi così che il risultato sia completamente sbilanciato
  - non hanno fatto undersampling

- Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization
  
  - manca chiarezza sulla versione di scikit learn
  
  - fanno girare dei modelli ma non ci sono dettagli su come abbiano fatto il tuning
  
  - cicflowmeter 3 a me ha dato qualche problema con gli ip

- Generating Network Intrusion Detection Dataset Based on Real and Encrypted Synthetic Attack Traffic (questo è il paper che presenta hikair-2021)
  
  - non hanno publicato il flowmeter, quindi provando a usare cicflowmeter ottengo un risultato diverso
  
  - hanno messo le label senza usare il timestamp che crea un po' di conflitti
  
  - non hanno pubblicato il timestamp
  
  - non è vero che le loro feature sono le stesse di cicids, inoltre il numero di righe non fa il match con zeek e con cicflowmeter

- Classification and Explanation for Intrusion Detection System Based on Ensemble Trees and SHAP Method
  
  - usano una no code
  
  - non misurano il tempo impiegato per farle il tutto
  
  - mancano le specifiche dell'albero

- Feature Drift Aware for Intrusion Detection System Using Developed Variable Length Particle Swarm Optimization in Data Stream
  
  - eliminano alcuni attacchi per avere solo quelli bruteforce, così da avere le metriche al massimo

- A detailed analysis of CICIDS2017 dataset for designing Intrusion Detection Systems
  
  - Il problema è che non controllano se questi attacchi sono simili tra loro e poi li mettono assieme. Per esempio xss e brute force non ha senso che stiano assieme.
  - nello stesso paper di introduzione del dataset fanno feature selection e le feature selezionate non sono le stesse per tutti gli attacchi ddos per esempio

- Improving AdaBoost-based Intrusion Detection System (IDS) Performance on CIC IDS 2017 Dataset
  
  - cita un altro studio dove dicono che adaboost ha pochi false negative e pochi false positive, ma non è vero nel mio caso
  
  - loro dicono di aver usato monday working hours ma quel dataset contiene solo dati benigni

- CICIDS-2017 Dataset Feature Analysis With Information Gain for Anomaly Detection
  
  - parlano di fare feature selection ma volendo lavorare con modelli ad albero si potrebbe optare per fare il pruning

- Effectiveness of Machine Learning based Intrusion Detection Systems
  
  - non fanno undersampling ne hyper tuning, risultato la knn è la migliore perché non ha problemi con il dataset sbilanciato assumendo che i vari cluster siano ben distanti
  
  - manca la spiegazione su cosa hanno usato per fare feature importance

- Anomaly-Based Intrusion Detection From
  Network Flow Features Using Variational
  Autoencoder (200 citazioni)
  
  - allenano il modello usando un metodo semi supervised, si allena il modello senza label ma si testa con label
  
  - non possono fare cross validation siccome il training è fatto su dati non anomali
  
  - fanno hyper tuning e dopo dicono che hanno usato la rule of thumb per avere i parametri giusti
  
  - fanno notare come il loro metodo funzioni anche con delle percentuali di dati in training ridicole

- Dual-IDS: A bagging-based gradient boosting decision tree model for network anomaly intrusion detection system
  
  - non parlano delle feature da eliminare eppure ci sono

# Perché il mio studio è diverso

- dopo aver dimsostrato che con il mio metodo il mio ids riesce a riconoscere alcuni attacchi mai visti bisogna vedere se unendo i dataset si può fare di meglio

- ci sarà la parte di xai che manca in molti studi per fare feature selection

- farò il confronto tra HIKARI-2021 e CICIDS-2017 usando il modello fatto per uno con l'altro dataset per testare gli zero day

- nessuno parla del tempo impiegato nel fare la predizione

- devo fornire anche i pacchetti dettagliati

- come ho fatto il tune dei modelli nessuno lo dice

- pure il peso in memoria non è stato riportato

- ho mostrato come l'undersampling funzioni a differenza del altro paper

- potrei eventualmente pensare al over sampling

- devo spiegare perché non usare la pca siccome fa perdere di spiegabilità

- dovrei testare diverse configurazioni dell'albero

- posso anche stampare gli alberi

- spigare come funziona hikari, hanno usato zeek 3 ma non combicia con il 6

- spiegare il bug cicflowmeter 4 e quello precedente

- fornire schema di indirizzi ip e quali file contengono cosa in hikari

- spigare che hikari ha le label diverse

- spiegare come ho ottenuto i risultati e farlo per ogni attacco, nessuno lo fa

- spiegare che il traffico di background non è così tanto valido perché è possibile che dello streaming assomogli di più a un bruteforce

- spiegare come ho fatto a creare quel test degli attacchi zero day

- spiegare come in realtà avere il dataset sbilanciato faccia ottenere un punteggio migliore con l'f1 score

- spiegare che per misuare i tempi ho fatto fare un giro di warmup

- [Rational for KernelExplainer sampling · Issue #375 · shap/shap · GitHub](https://github.com/shap/shap/issues/375)

- utilizzo di kmeans per fare selezione dei punti da analizzare

- puntare tutto sugli alberi, non solo sono più veloci e funzionano meglio ma tree shap analizza il modello stesso invece di fare un surrugato quindi sono perfetti

- siccome faccio pruning dei modelli ensemble come risultato ho che non tutte le features vengano usate, questo fa si che a livello di zero day funzionino diversamente perché nuovi attacchi possono usare features ancora inutilizzate

- stressare sul fatto che molti paper non parlano di quale versione del dataset abbiano usato

- bisogna spiegare che non ho selezionato gli attacchi ad arte e le poche volte che l'ho fatto era perché il numero di osservazioni era davvero esiguo

- non ha senso fare come fanno alcuni studi che prendono in cosiderazione solo alcuni attacchi

# Cose fatte

- build di cicflowmeter usando un wm con ubuntu 20.04 e openjdk-8
- Testati pytorch e tensorflow, tensorflow andava lento così ho scelto pytorch. Con tensorflow la libreria shap funziona e deep lift shap funziona senza usare troppa memoria, mentre per pytoch serve captum che prende più memoria. A questo punto ci sono due direzioni siccome pure il kernel shap prende troppa memoria, una è usare sampling shap ma è lento oppure si può fare sampling dei dati. Per fare sampling si possono prendere dati a caso, oppure si può usare k medoids o k means come consigliato dagli autori di shap. Quindi userò k medoids per fare sampling e sceglierò il numero di cluster in base a elbow method e silhouette per avere un numero sufficientemente alto ma non troppo che rappresenti la distribuzione.
- confrontati risultati di kernel shap vs linear vs tree e avere più background non fa troppa differenza
- notato che più semplifico il modello e più è capace di vedere gli zero day
- controllato la convergenza degli shapley value e che la varianza non aumentasse
- utilizzare il metodo interventional per spiegare i modelli ad albero
- provato ad aggiungere dei dati con smote per aumentare la class miner ma la crossvalidation non è migliorata
- lightgbm migliora nella feature selection con il metodo interventional, perché i grafici sono più coerenti
- confrontare le variabili usate da adaboost con quelle che il kernel mi ha dato indietro
- libreria con cross validation custom per avere una cross validation stratificata
- far notare come questo studio possa avere dei grossi bias in base a come si fa la cross validation, al posto di quello di imblearn
- scrivere di come non pubblicare i propri notebook renda opaco tutto lo studio e citare il caso del tipo che ha usato l'index di pandas come feature
- notato che con più osservazioni il modello va peggio, ma facendo un po' di feature selection a volte migliora
- dato più peso a alla classe negativa per avere precision e recall uguali
- devo dire che ho usato la z-score con i modelli che ne necessitavano

# Da fare

- ricordarsi di far vedere al prof la questione del singolo albero per ciascun attacco

- contare tempo di scaling e di trasformazione dei dati in tensor, misurare a parte giusto per avere un'idea

- bisogna provare a fare feature selection dei modelli al albero usando l'information gian assieme a shap

- mettere dei limiti di importanza da usare per fare feature selection, per farlo bisogna standardizzare l'importanza

- provare a usare self organizing map per vedere se gli attacchi sono tutti simili o meno

- guardare quanta memoria richiede un normale dataset contro un dataset di float

- bisogna usare bagging sui miei modelli

# Issue utili shap

-  [Question about TreeExplainer data parameter &amp; train/test subsets. · Issue #1366 · shap/shap · GitHub](https://github.com/shap/shap/issues/1366) qui dice che è meglio usare interventional
-  [question about kmean · Issue #107 · shap/shap · GitHub](https://github.com/shap/shap/issues/107) qui dice che kmeans ha senso come background data
- [Should global shap values depend on the variance of the feature? · Issue #696 · shap/shap · GitHub](https://github.com/shap/shap/issues/696) qui dice che per gli alberi è meglio usare loss
- [which method is best for subsampling? · Issue #1018 · shap/shap · GitHub](https://github.com/shap/shap/issues/1018) fa il confronto tra kmeans e random sample
- [Question: Background and explaining set sizes for TreeExplainer&#39;ing large data sets · Issue #955 · shap/shap · GitHub](https://github.com/shap/shap/issues/955) come capire quando ho abbastanza dati
- https://github.com/shap/shap/issues/1098 spiega la differenza tra true to the model e true to the data, interventional è true to the model quindi sarebbe meglio per feature selection

# Paper letti

- Network Intrusion Detection Based on Explainable Artificial Intelligence, fa semplimente vedere quale modello sia il migliore in termini di risultato per fare un IDS. Parla di tutti i passaggi, pulizia training ecc ma non si sofferma più di tanto sulla spiegabilità.

- Analysis and Optimization of Convolutional Neural Network Architectures, è una tesi su come ottimizzare un CNN potrebbe essere utile nel caso volessi seguire questa strada

- Network Intrusion Detection for IoT Security Based on Learning Techniques
  
  - parla in generale degli ids per iot, non è specifico sui modelli da scegliere
  - fa notare come un modello di ml in generale è meglio di un confronto delle firme che richiede avere un database enorme e molta cpu per cifrare
  - cita un paper che descrive come ha fatto una mlp per allenare la rete
  - parla di papers che usano un dataset per il training e un altro per il test. potrebbe essere utile per dimostare come il modello funzioni con nuovi attacchi.

- CNN-LSTM-based IDS on Precision Farming for IIoT data
  
  - qui fanno un bello schema che rende comprensibile come abbiano fatto a fare il training dei dati
  
  - fanno pure vedere una tabella con le feature selezionate

- Dual-IDS: A bagging-based gradient boosting decision tree model for network anomaly intrusion detection system
  Maya
  
  - non usano f1 il che potrebbe minare il risultato ottenuto dove spiegano perché il loro modello è migliore

- The Many Shapley Values for Model Explanation Mukund
  
  - qui citano tutte le volte che gli shapley values vennero usati e si nota come quelli classici vennero usati prima dell'introduzione di shap
  
  - non riguarda i miei studi

# Paper non letti

- Hybrid Intrusion Detection System for DDoS Attacks (qui parlano di come classificare in maniera veloce i dati sia importante)
- GAN-based imbalanced data intrusion detection system (qui parlano di come classificare in maniera veloce i dati sia importante)
- Ant colony optimization and feature selection for intrusion detection (parla di come fare feature selection sia un bene)
- Toward credible evaluation of anomaly-based intrusion-detection methods (qui parla del fatto che molti paper non siano credibili)
- Challenges in Experimenting with Botnet Detection Systems (parla della difficoltà nel comparare alcuni paper con altri)
- Characterization of tor traffic using time based features (paper che introduce CICFlowMeter)
- An Efficient Explanation of Individual Classificationsusing Game Theory (paper dal quale è stato ispirato sampling explainer)
- Cybersecurity attacks: which dataset
  should be used to evaluate an intrusion
  detection system?
- Anchors: High-Precision Model-Agnostic Explanations (parla di come sommare le spiegazioni locali ne crei una globale accurata)
- Rules  of  Machine Learning: Best Practices for ML Engineering (parlano di come fanno le cose in google è un paper sulle best practice)
- The many shapley values for model explanation (parla di come alcune approssimazioni di shap siano diverse)
- random forest
- Feature relevance quantification in explainable AI: A causal problem (parlano di alcuni problemi nell'interpretazione di shap)
- Understanding variable importances in Forests of randomized trees (parla di come interpretare il mdi per la random forest)
- paper sugli ids xai
  - Explaining Network Intrusion Detection System Using Explainable AI Framework
  - An Explainable Machine Learning Framework for Intrusion Detection Systems
  - Explainable Artificial Intelligence (XAI) to Enhance Trust Management in Intrusion Detection Systems Using Decision Tree Model
  - Achieving Explainability of Intrusion Detection System by Hybrid Oracle- Explainer Approach
  - An Explainable Machine Learning-based Network Intrusion Detection System for Enabling Generalisability in Securing IoT Networks
  - Evaluating Standard Feature Sets Towards Increased Generalisability and Explainability of ML-based Network Intrusion Detection
- paper su hikari (32 citazioni 11/01/2024)
  - PeerAmbush: Multi-Layer Perceptron to Detect Peer-to-Peer Botnet
- paper su cicids (829 citazioni 11/01/2024 senza parola survey nel titolo)
  - Effective network intrusion detection using stacking-based ensembleapproach (rifanno il dataset in modo da ottenere risultati migliori)

# Making dataset

- anzitutto ho usato cicflowmeter per rifare quello fatto con hikari

- ho notato un bug in cicflowmeter per cui le porte e gli ip a volte erano invertiti

- gli autori dicono che hanno fatto il dataset guardando ip e porte, allora sto facendo la stessa cosa prendendo gli ip e le porte dal dataset originale e mettendole su quello nuovo

- ho provato a rifare il data set usando le stesse connessioni ma non ha funzionato

- allora ho fatto un data set bilanciato partendo dall'originale dove ho creato alcune variabili booleane per ip e porte e poi ho chiesto all'albero di spiegare

- così facendo ho scoperto quali sono le informazioni più importanti per l'albero
