# Nuclear Transition Models

Repository di supporto per i modelli Python del progetto "Storia Futura".
Include i due script principali richiesti, con istruzioni per esecuzione e dipendenze.

## Contenuto

- `codice_5_FIXED  (1).py`
  - Modello nucleare e scenari geopolitici per Cina / USA / UE.
- `codice_6_comparison (1).py`
  - Script di confronto tra scenari o metriche basate sul modello.
- `requirements.txt`
  - Elenco delle librerie Python necessarie.
- `.gitignore`
  - File di esclusione standard per Python.

## Requisiti

- Python 3.10+ (consigliato)
- Librerie Python:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `scipy`

## Installazione

1. Apri un terminale nella cartella `nuclear-transition-models`
2. Crea un ambiente virtuale (consigliato):

```powershell
python -m venv .venv
```

3. Attiva l'ambiente virtuale:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Installa le dipendenze:

```powershell
pip install -r requirements.txt
```

## Esecuzione

Usa i nomi dei file tra virgolette a causa degli spazi e delle parentesi:

```powershell
python "codice_5_FIXED  (1).py"
python "codice_6_comparison (1).py"
```

Se vuoi eseguire uno script specifico, sostituisci il nome del file con quello desiderato.

## Note

- I file mantengono il nome originale con spazi e parentesi, quindi è importante usare le virgolette.
- Questa repository è pronta per essere collegata a un remote GitHub.
  Se vuoi aggiungere il remote, usa:

```powershell
git remote add origin https://github.com/<tuo-utente>/<tuo-repo>.git
git branch -M main
git add .
git commit -m "Initial commit"
git push -u origin main
```

## Limitazioni

- L'ambiente locale in uso non contiene l'eseguibile `git`, quindi la repository è stata preparata ma non inizializzata con Git automaticamente.
