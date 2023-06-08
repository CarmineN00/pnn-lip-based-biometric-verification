# Descrizione files ".py"

- **pnn.py**: contiene i metodi e la struttura della Probabilistic Neural Network (utilizzata). Contiene inoltre tanti metodi quanti sono i kernel disponibili all'uso.

- **read_data.py**: contiene i metodi per generare i file csv a partire dai video. Contiene funzioni parametriche per specificare il metodo di sampling, quale configurazione utilizzare, quanti frames prelevare, quale metrica utilizzare, quale strategia seguire.

- **sparse_coding_babele.py**: contiene i metodi per generare, a partire dai video del dataset BABELE, il dataset csv ottenuto tramite l'applicazione di tecniche di sparse coding ai video.

- **sparse_coding_babele.py**: contiene i metodi per generare, a partire dai video del dataset VidTimit, il dataset csv ottenuto tramite l'applicazione di tecniche di sparse coding ai video.

- **verify.py**: contiene i metodi di generazione dei dataset di verifica e richiama i metodi di pnn.py per effettuare il forecasting dei risultati.

- **one_shot_tester.py**: permette di definire tutti i parametri del forecasting che si desidera effettuare. Ricerca un dataset csv che gode degli esatti parametri, altrimenti lo crea a partire dai video di BABELE richiamando i metodi di read_data.py. Esegue il forecasting dei risultati stampando su console sia la configurazione che le percentuali ottenute, richiamando i metodi di pnn.py 

- **LipLandmarks.py**: frozenset, contiene le coppie di landmarks che determinano i segmenti della configurazione BASE

- **LipLandmarksDynamic.py**: frozenset, contiene le coppie di landmarks che determinano i segmenti della configurazione DYNAMIC

- **LipLandmarksFullMesh.py**: frozenset, contiene le coppie di landmarks che determinano i segmenti della configurazione FULL-MESH

- **delaunay.py**: contiene il codice per richiamare la creazione del dataset con la configurazione Delaunay, richiama inoltre i metodi di pnn.py per il forecasting dei risultati.

- **fullmesh.py**: contiene un main in cui è stata testata un'istanza di forecasting che fa uso della configurazione FULL-MESH

- **print_landmarks.py**: contiene il codice per generare le immagini esemplificative delle diverse configurazioni (le immagini .jpg presenti nella repository)

- **test_all.py**: contiene il codice che permette di testare, specificando entro quali parametri variare, più configurazioni una dopo l'altra, facendo l'uso di cicli e iterazioni.

- **video_generator.py**: permette di generare i video di esempio mostrati nella presentazione.

- **classifiers.py**: usato per effetuare il forecasting su dataset csv già esistenti utilizzando però altri classificatori, come SVD, Random Forest, etc... Mostra i risultati del forecasting su console

- **playground.py**: area di playground per lo sviluppo ed il testing di "patch" da importare