# Rozpoznawanie Ręcznie Pisanych Cyfr

To prosty projekt w Pythonie, który stworzyłem do rozpoznawania ręcznie pisanych cyfr.  
Wykorzystuje TensorFlow oraz niewielką sieć neuronową do przewidywania cyfr od 0 do 9.

Aplikacja pozwala narysować cyfrę na obszarze roboczym (canvasie), a następnie próbuje odgadnąć, jaka to liczba.

## Wykorzystane technologie

- Python  
- TensorFlow / Keras  
- TensorFlow Datasets  
- NumPy  
- Pillow  
- Tkinter  

## Funkcjonalność

- Trenuje model na zbiorach danych MNIST i EMNIST  
- Umożliwia rysowanie cyfry w prostej aplikacji desktopowej  
- Przewiduje narysowaną liczbę  
- Wyświetla poziom pewności dla każdej predykcji  
- Zapisuje wytrenowany model, aby można było go użyć ponownie  

## Jak to działa

1. Model jest trenowany na obrazach cyfr ze zbiorów MNIST i EMNIST.  
2. Obrazy EMNIST są przed treningiem korygowane pod względem orientacji.  
3. Rysujesz cyfrę na obszarze Tkintera.  
4. Rysunek jest skalowany i oczyszczany przed predykcją.  
5. Model zwraca przewidywanie oraz poziomy pewności.

## Pierwsze uruchomienie

Jeśli nie zostanie znaleziony zapisany model, program automatycznie go wytrenuje.  
Może to potrwać kilka minut w zależności od wydajności systemu.

## Pliki projektu

- `data.py` — wczytywanie i przygotowanie danych  
- `model.py` — budowa, trenowanie, zapisywanie i wczytywanie modelu  
- `gui.py` — interfejs graficzny do rysowania cyfr  
- `main.py` — punkt wejścia aplikacji  
- `digit_model.keras` — zapisany model  

## Jak uruchomić

Sklonuj projekt:

```bash
git clone https://github.com/your-username/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
python main.py