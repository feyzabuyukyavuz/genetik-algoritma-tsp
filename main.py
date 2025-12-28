import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# --- 1. AYARLAR VE HİPERPARAMETRELER (Rapor Tablosu İçin) ---
FILE_PATH = 'Distance_matrix.xlsx'
POP_SIZE = 100  # Popülasyondaki rota sayısı
GEN_COUNT = 100  # İterasyon sayısı (100'e düşürüldü)
MUTATION_RATE = 0.05  # Mutasyon olasılığı (%5)
ELITE_SIZE = 5  # Her nesilde en iyi 5 rota korunur


# --- 2. VERİ YÜKLEME ---
def load_data(path):
    try:
        # Excel'deki ilk sütun ve satır index olduğu için index_col=0
        df = pd.read_excel(path, index_col=0)
        print("Mesafe matrisi başarıyla yüklendi.\n")
        return df.values
    except Exception as e:
        print(f"Hata: Excel dosyası okunamadı! {e}")
        return None


# --- 3. GENETİK OPERATÖRLER ---

def calculate_distance(route, matrix):
    """Rotanın toplam uzunluğunu hesaplar."""
    total = 0
    for i in range(len(route) - 1):
        total += matrix[route[i]][route[i + 1]]
    return total


def create_random_route():
    """0'da başlayıp 0'da biten rastgele rota oluşturur."""
    nodes = list(range(1, 21))
    random.shuffle(nodes)
    return [0] + nodes + [0]


def ordered_crossover(p1, p2):
    """Şehir tekrarını önleyen özel çaprazlama (OX)."""
    p1_mid = p1[1:-1]
    p2_mid = p2[1:-1]
    size = len(p1_mid)
    start, end = sorted(random.sample(range(size), 2))

    child_mid = [None] * size
    child_mid[start:end] = p1_mid[start:end]

    p2_ptr = 0
    for i in range(size):
        if child_mid[i] is None:
            while p2_mid[p2_ptr] in child_mid:
                p2_ptr += 1
            child_mid[i] = p2_mid[p2_ptr]
    return [0] + child_mid + [0]


def mutate(route):
    """Rastgele iki şehrin yerini değiştirir."""
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(1, 21), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


# --- 4. ANA DÖNGÜ ---

def run_genetic_algorithm():
    matrix = load_data(FILE_PATH)
    if matrix is None: return

    # Başlangıç popülasyonu oluşturma
    population = [create_random_route() for _ in range(POP_SIZE)]

    best_dist_history = []
    global_best_route = None
    global_best_dist = float('inf')

    for gen in range(1, GEN_COUNT + 1):
        # Fitness hesapla ve popülasyonu en iyiden en kötüye sırala
        population = sorted(population, key=lambda r: calculate_distance(r, matrix))

        current_best_route = population[0]
        current_best_dist = calculate_distance(current_best_route, matrix)

        # Global en iyiyi güncelle
        if current_best_dist < global_best_dist:
            global_best_dist = current_best_dist
            global_best_route = current_best_route

        best_dist_history.append(global_best_dist)

        # KONSOL ÇIKTISI (Her İterasyon Tek Tek Yazdırılır)
        print(f"Iteration {gen} → Best Distance: {int(global_best_dist)}")
        print(f"Best Route: {global_best_route}\n")

        # Yeni nesil (popülasyon) oluşturma
        new_gen = population[:ELITE_SIZE]  # En iyileri koru

        while len(new_gen) < POP_SIZE:
            # En iyi 20 içinden 2 ebeveyn seçip çaprazla ve mutasyona uğrat
            parents = random.sample(population[:20], 2)
            child = ordered_crossover(parents[0], parents[1])
            child = mutate(child)
            new_gen.append(child)

        population = new_gen

    # --- 5. SONUÇLARI GÖRSELLEŞTİRME ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_dist_history, color='blue', marker='o', markersize=3)
    plt.title("Genetik Algoritma - Mesafe Gelişimi (100 İterasyon)")
    plt.xlabel("İterasyon (Nesil)")
    plt.ylabel("En İyi Mesafe")
    plt.grid(True)
    plt.show()

    print("--- ALGORİTMA TAMAMLANDI ---")
    print(f"Final En İyi Rota: {global_best_route}")
    print(f"Final En Kısa Mesafe: {int(global_best_dist)}")


if __name__ == "__main__":
    run_genetic_algorithm()