# Projekt koncowy NLP: PolEval SMIGIEL (Constrained)
### Autorzy: Franciszek Trzeciak i Szymon Pawlak

Projekt realizowany w ramach kursu Przetwarzania Jezyka Naturalnego.
Celem jest klasyfikacja tekstow w jezyku polskim jako napisanych przez czlowieka lub wygenerowanych przez modele LLM. (PolEval 2025 Task 1 (Constrained))

## Struktura plikow
```text
data/      # Folder na zbior danych
outputs/   # Wyniki naszego fine-tuningu, logi TensorBoard i checkpointy
config.py  # Konfiguracja modelow, hiperparametrow
data.py    # Skrypty do ladowania i procesowania danych
model.py   # Definicja architektury i konfiguracja LoRA
trainer.py # Klasa Trainer (petla ucząca, ewaluacja)
train.py   # Glowny skrypt uruchamiający trening na roznych modelach
predict.py # Skrypt do generowania predykcji na zbiorze testowym
```

## Wymagania
Projekt wymaga Python 3.10+, maszyne z CUDA oraz srodowiska z bibliotekami wymienionymi w pliku requirements.txt

## Trening
W sklad projektu wchodza 3 modele. Aby wytrenowac je nalezy pojedynczo uruchomic:
!UWAGA! Modele sa juz wytrenowane w */outputs*, dlatego uruchomienie ponizszych skryptow bedzie skutkowalo usunieciem poprzednich modeli. Aby sprobowac trening zmien tymczasowo nazwe katalogu */outputs* na np. */output-temp*.
```
python train.py --model [deberta | herbert | polish-roberta]
```
Lub po wytrenowaniu deberta, herbert i polish roberta zastosowac optymizacje ensemble.
```
python ensemble.py
```
Mozemy monitorowac postepy za pomoca tensorboard
```
tensorboard --logdir [deberta | herbert | polish-roberta]
```

## Ewaluacja na zbiorze testowym (predykcje tylko Out Of Fold, poniewaz PolEval zamkniete)
Aby szczegolowo ewaluowac pojedynczy model, lub model ensemble (accuracy, precision, recall, f1)
```
python evaluate.py [deberta | herbert | polish-roberta | ensemble]
```
