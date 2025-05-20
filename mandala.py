import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw
import math
import matplotlib.colors as mcolors
import os
import json

# ---------------------------
# Matrix Generation Functions
# ---------------------------

def digital_value(n):
    """Calculate the digital root of a number."""
    while n >= 10:
        n = sum(int(digit) for digit in str(n))
    return n

def generate_random_square(size, personal_numbers=None):
    """Generate a random magic square, optionally embedding personal numbers."""
    all_numbers = list(range(1, size**2 + 1))
    
    if personal_numbers:
        valid_personal_numbers = []
        for num in personal_numbers:
            if num in all_numbers and num not in valid_personal_numbers:
                valid_personal_numbers.append(num)
                all_numbers.remove(num)
        
        random.shuffle(all_numbers)
        square = np.array(all_numbers + valid_personal_numbers).reshape(size, size)
    else:
        random.shuffle(all_numbers)
        square = np.array(all_numbers).reshape(size, size)
    
    return square

def compute_fitness(square, personal_numbers=None):
    """Compute fitness of a magic square."""
    size = square.shape[0]
    target_sum = sum(range(1, size**2 + 1)) // size
    fitness = 0
    
    for i in range(size):
        row_sum = sum(square[i, :])
        col_sum = sum(square[:, i])
        fitness += abs(row_sum - target_sum) + abs(col_sum - target_sum)
    
    diag_sum1 = sum(square[i, i] for i in range(size))
    diag_sum2 = sum(square[i, size - i - 1] for i in range(size))
    fitness += abs(diag_sum1 - target_sum) + abs(diag_sum2 - target_sum)

    if personal_numbers:
        for num in personal_numbers:
            if num not in square:
                fitness += 10
    
    return fitness

def boosting_step(square, size, learning_rate=0.1):
    """Improve the square through small adjustments."""
    new_square = square.copy()
    fitness = compute_fitness(square)
    
    for i in range(size):
        for j in range(size):
            for change in [-1, 1]:
                new_square[i, j] += change
                new_fitness = compute_fitness(new_square)
                
                if new_fitness < fitness:
                    fitness = new_fitness
                else:
                    new_square[i, j] -= change
    
    return new_square

def evaluate_square(square, personal_numbers, size, learning_rate):
    new_square = boosting_step(square, size, learning_rate)
    fitness = compute_fitness(new_square, personal_numbers)
    return new_square, fitness

def xgboost_like_algorithm(personal_numbers, size=3, population_size=100, generations=500, learning_rate=0.1):
    """Evolutionary algorithm to find optimized magic squares."""
    population = [generate_random_square(size, personal_numbers) for _ in range(population_size)]
    best_square = None
    best_fitness = float('inf')

    pool = Pool(processes=cpu_count()-4)

    for generation in range(generations):
        results = pool.starmap(
            evaluate_square,
            [(square, personal_numbers, size, learning_rate) for square in population]
        )

        population = [result[0] for result in results]
        fitness_scores = [result[1] for result in results]
        
        min_fitness = min(fitness_scores)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_square = population[fitness_scores.index(min_fitness)]

        if best_fitness == 0:
            break
    
    pool.close()
    pool.join()
    
    return best_square

def get_unique_consonants(name):
    consonants = [c for c in name.upper() if c.isalpha() and c not in "AEIOU"]
    return list(dict.fromkeys(consonants))

def get_wirth_base_consonants(name):
    base_map = {
        'B': 'B', 'P': 'B', 'F': 'B', 'V': 'B',
        'D': 'D', 'T': 'D', 'TH': 'D',
        'G': 'G', 'K': 'G', 'Q': 'G', 'C': 'G',
        'L': 'L', 'R': 'L', 'N': 'L', 'M': 'L',
        'S': 'S', 'Z': 'S', 'X': 'S', 'H': 'H', 'J': 'J', 'Y': 'Y', 'W': 'W'
    }
    name = name.upper()
    result = []
    for c in name:
        if c in base_map and base_map[c] not in result:
            result.append(base_map[c])
    return result

def generate_panmagic_square(n):
    """Generate a panmagic square of odd order n using Strachey's method."""
    if n % 2 == 0:
        raise ValueError("Panmagic square generation implemented only for odd n.")
    square = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            square[i, j] = (n * ((i + j + (n // 2)) % n) + ((i + 2 * j) % n) + 1)
    return square

def generate_normal_magic_square(n):
    """Generate a normal magic square of odd order n using the Siamese method."""
    if n % 2 == 0:
        raise ValueError("Normal magic square generation implemented only for odd n.")
    square = np.zeros((n, n), dtype=int)
    num = 1
    i, j = 0, n // 2
    while num <= n * n:
        square[i, j] = num
        num += 1
        newi, newj = (i - 1) % n, (j + 1) % n
        if square[newi, newj]:
            i += 1
        else:
            i, j = newi, newj
    return square

def generate_associative_magic_square(n):
    """Generate an associative magic square (for odd n, same as normal)."""
    return generate_normal_magic_square(n)

# --------------------------
# Mandala Creation Functions
# --------------------------

def create_angelic_mandala(magic_squares, personal_numbers, name):
    """Generate sacred meditation mandala from matrices."""
    palette = matrix_to_palette(magic_squares[-1])
    cmap = LinearSegmentedColormap.from_list("matrix_palette", palette)
    
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

    current_radius = 0
    band_height = 2
    
    for idx, matrix in enumerate(magic_squares):
        if matrix is None:
            continue
            
        N = matrix.shape[0]
        matrix = matrix.astype(float)
        
        min_val, max_val = matrix.min(), matrix.max()
        normalized = (matrix - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else matrix * 0.5
            
        band_start = current_radius
        current_radius += band_height
        layer_thickness = band_height / N

        for i in range(N):
            for j in range(N):
                theta = 2 * np.pi * j / N
                width = 2 * np.pi / N
                alpha = normalized[i,j] * 0.8 + 0.2
                
                for glow in [1.0, 0.7, 0.4]:
                    r = band_start + i * layer_thickness
                    ax.bar(theta, height=layer_thickness*glow, width=width*glow,
                          bottom=r, color=cmap(normalized[i,j]), 
                          alpha=alpha*glow*0.3, edgecolor='none')

                ax.bar(theta, height=layer_thickness, width=width,
                      bottom=band_start + i * layer_thickness,
                      color=cmap(normalized[i,j]), edgecolor='white',
                      linewidth=0.3, alpha=alpha)

        if idx > 0:
            circle = plt.Circle((0,0), band_start, color='white', 
                              fill=False, linewidth=0.7, alpha=0.5)
            ax.add_artist(circle)

    draw_angelic_symbols(ax, magic_squares[-1], current_radius, personal_numbers)
    
    plt.savefig(f"{name}_mandala.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    add_final_touches(name)

def draw_angelic_symbols(ax, main_matrix, radius, personal_numbers):
    if main_matrix is None:
        return
    
    N = main_matrix.shape[0]
    symbol_radius = radius * 1.05
    
    for i in range(N):
        for j in range(N):
            if main_matrix[i,j] in personal_numbers:
                theta = 2 * np.pi * j/N - np.pi/N
                ax.text(theta, symbol_radius, "▲", fontsize=18, color='gold', ha='center', va='center', alpha=0.8)
                ax.text(theta, symbol_radius, "✦", fontsize=14, color='white', ha='center', va='center', alpha=1)

def add_final_touches(name):
    img = Image.open(f"{name}_mandala.png")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    for _ in range(50):
        angle = np.random.random() * 2 * math.pi
        dist = np.random.random() * width/2
        x = width/2 + math.cos(angle) * dist
        y = height/2 + math.sin(angle) * dist
        draw.ellipse([x-3, y-3, x+3, y+3], fill=(255, 255, 200, 50))
    
    border_width = 20
    for i in range(border_width):
        rect = [i, i, width-i, height-i]
        draw.rectangle(rect, outline=(255, 215, 0, 100 - 5*i), width=1)
    
    img.save(f"{name}_mandala_final.png", "PNG")

# --- Palette generation from matrix values ---
def matrix_to_palette(matrix, num_colors=5):
    """Generate a palette from matrix values by mapping them to a color space."""
    flat = np.unique(matrix.flatten())
    # Select evenly spaced values from the sorted unique values
    if len(flat) < num_colors:
        values = np.linspace(flat.min(), flat.max(), num_colors)
    else:
        values = np.interp(np.linspace(0, len(flat)-1, num_colors), np.arange(len(flat)), flat)
    # Normalize to [0, 1]
    normed = (values - flat.min()) / (flat.max() - flat.min()) if flat.max() > flat.min() else values * 0.5
    # Map to colors (e.g., using HSV colormap)
    palette = [mcolors.hsv_to_rgb([hue, 0.7, 0.9]) for hue in normed]
    return palette

# --- Hebrew Gematria mapping (basic example for demonstration) ---
HEBREW_GEMATRIA = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ך': 20, 'ל': 30, 'מ': 40, 'ם': 40, 'נ': 50, 'ן': 50, 'ס': 60, 'ע': 70,
    'פ': 80, 'ף': 80, 'צ': 90, 'ץ': 90, 'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400
}

def get_hebrew_gematria(name):
    """Calculate the sum of Hebrew letter values (Gematria)."""
    return sum(HEBREW_GEMATRIA.get(c, 0) for c in name)

def get_wirth_number(name):
    """Calculate a Wirth-based number from mapped consonants."""
    consonants = get_wirth_base_consonants(name)
    # Assign a simple value: A=1, B=2, ..., Z=26 for mapped consonants
    values = [ord(c) - ord('A') + 1 for c in consonants if 'A' <= c <= 'Z']
    return digital_value(sum(values)) if values else 0

# -----------------
# Main Execution
# -----------------

if __name__ == "__main__":
    # === INPUT YOUR DETAILS HERE ===
    name = "Eduard Samokhvalov"        # Replace with your name
    date_of_birth = "08/07/1984"  # Replace with your birth date
    
    # Calculate personal numbers for Angelic
    expression_num = digital_value(sum(ord(char) for char in name if char.isalpha()))
    soul_urge_num = digital_value(sum(ord(char) for char in name if char in "AEIOUaeiou"))
    day, month, year = map(int, date_of_birth.split('/'))
    personal_numbers_angel = [expression_num, soul_urge_num, 
                             digital_value(day), digital_value(month), digital_value(year)]
    
    # Calculate personal numbers for Hebrew (if name is in Hebrew)
    personal_numbers_hebrew = [digital_value(get_hebrew_gematria(name)),
                               digital_value(get_hebrew_gematria(date_of_birth)),
                               digital_value(day), digital_value(month), digital_value(year)]
    
    # Calculate personal numbers for Wirth
    wirth_num = get_wirth_number(name)
    personal_numbers_wirth = [wirth_num, digital_value(day), digital_value(month), digital_value(year)]
    
    # Generate or load magic squares
    sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    systems = [
        ("angelic", personal_numbers_angel),
        ("hebrew", personal_numbers_hebrew),
        ("wirth", personal_numbers_wirth)
    ]
    
    for system_name, personal_numbers in systems:
        magic_squares_normal = []
        magic_squares_panmagic = []
        magic_squares_associative = []
        cache_filename = f"magic_squares_{system_name}_{name.replace(' ', '_')}_{date_of_birth.replace('/', '-')}.json"
        regenerate = False
        if os.path.exists(cache_filename):
            print(f"Loading pre-generated magic squares for {system_name} from cache...")
            with open(cache_filename, 'r') as f:
                data = json.load(f)
                magic_squares_normal = [np.array(ms) for ms in data['normal']]
                magic_squares_panmagic = [np.array(ms) for ms in data['panmagic']]
                magic_squares_associative = [np.array(ms) for ms in data['associative']]
            expected_count = len(sizes)
            if (len(magic_squares_normal) != expected_count or
                len(magic_squares_panmagic) != expected_count or
                len(magic_squares_associative) != expected_count):
                print(f"Cache incomplete or inconsistent for {system_name}, regenerating matrices...")
                regenerate = True
        else:
            regenerate = True
        if regenerate:
            print(f"Generating sacred matrices for {system_name}...")
            magic_squares_normal = []
            magic_squares_panmagic = []
            magic_squares_associative = []
            for size in sizes:
                magic_squares_normal.append(generate_normal_magic_square(size))
                magic_squares_panmagic.append(generate_panmagic_square(size))
                magic_squares_associative.append(generate_associative_magic_square(size))
            # Save to cache
            with open(cache_filename, 'w') as f:
                json.dump({
                    'normal': [ms.tolist() for ms in magic_squares_normal],
                    'panmagic': [ms.tolist() for ms in magic_squares_panmagic],
                    'associative': [ms.tolist() for ms in magic_squares_associative]
                }, f)
        # Create meditation mandala for each system
        print(f"Creating angelic mandala for {system_name}...")
        create_angelic_mandala(magic_squares_normal, personal_numbers, name.replace(" ", "_") + f"_{system_name}_normal")
        create_angelic_mandala(magic_squares_panmagic, personal_numbers, name.replace(" ", "_") + f"_{system_name}_panmagic")
        create_angelic_mandala(magic_squares_associative, personal_numbers, name.replace(" ", "_") + f"_{system_name}_associative")
    print(f"Complete! Check these files for all systems and types.")