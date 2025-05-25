import pandas as pd
from PIL import Image
from features import FeatureImage
import csv
from math import sqrt
import os


def dist(vector1, vector2):
    assert len(vector1) == len(vector2)
    return sqrt(sum((coord1 - coord2) ** 2 for coord1, coord2 in zip(vector1, vector2)))


def proximity(vector1, vector2):
    return 1 / (1 + dist(vector1, vector2))


feature_names = ['relative_' + name for name in
                 ['weight_I', 'weight_II', 'weight_III', 'weight_IV',
                  'center_x', 'center_y', 'inertia_x', 'inertia_y']]
target = '𐎓𐎆𐎐𐎚𐎋𐎘𐎚𐎁𐎗𐎎'


def print_mismatches(target_str, recognized_str):
    mismatches = []
    for i, (expected, actual) in enumerate(zip(target_str, recognized_str)):
        if expected != actual:
            mismatches.append((i, expected, actual))

    if mismatches:
        print("\nНесовпадающие символы:")
        print("Номер | Ожидаемый | Распознанный")
        print("------|-----------|--------------")
        for idx, expected, actual in mismatches:
            print(f"{idx + 1:4d}  | {expected:9s} | {actual:11s}")
    else:
        print("\nВсе символы распознаны верно")


def classify_symbols(symbols_path, features_path, output_csv):
    features = pd.read_csv(features_path)
    results = []
    sentence = ""
    mismatch_indices = []

    for i in range(len(target)):
        try:
            symbol = FeatureImage(Image.open(f'{symbols_path}/letter_{i}.png'), invert=True)
            feature_vector = [
                symbol.relative_weight_I(),
                symbol.relative_weight_II(),
                symbol.relative_weight_III(),
                symbol.relative_weight_IV(),
                symbol.relative_center(1),
                symbol.relative_center(0),
                symbol.relative_inertia(1),
                symbol.relative_inertia(0)
            ]

            proximities = features.apply(
                lambda row: (row['letter'], proximity(feature_vector, row[feature_names])),
                axis=1
            )

            proximities = sorted(proximities, key=lambda x: x[1], reverse=True)
            results.append(proximities)
            recognized_char = proximities[0][0] if proximities else "?"
            sentence += recognized_char

            if recognized_char != target[i]:
                mismatch_indices.append((i, target[i], recognized_char))

        except FileNotFoundError:
            print(f"Ошибка: файл символа {i} не найден")
            recognized_char = "?"
            sentence += recognized_char
            mismatch_indices.append((i, target[i], "?"))
            continue

    os.makedirs('results', exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Symbol Index', 'Hypotheses (Letter, Proximity)'])
        for i, hypotheses in enumerate(results):
            writer.writerow([i, hypotheses])

    correct = sum(1 for expected, actual in zip(target, sentence) if expected == actual)
    accuracy = correct / len(target) * 100

    return sentence, accuracy, mismatch_indices, results


if __name__ == '__main__':
    original_sentence, original_accuracy, original_mismatches, original_results = classify_symbols(
        symbols_path='../lab6/results/symbols',
        features_path='../lab5/results/features.csv',
        output_csv='results/original_classification.csv'
    )

    print("\nРезультаты классификации (52pt):")
    print("=" * 50)
    print(f"Ожидаемая строка: {target}")
    print(f"Распознанная строка: {original_sentence}")
    print(f"\nТочность распознавания: {original_accuracy:.2f}%")

    print_mismatches(target, original_sentence)

    print("\nЛучшие гипотезы для каждого символа:")
    for i, hypotheses in enumerate(original_results):
        best = hypotheses[0] if hypotheses else ("?", 0.0)
        status = "✓" if best[0] == target[i] else "✗"
        print(f"Символ {i + 1:2d}: {status} {best[0]} (сходство: {best[1]:.4f})")

    bigger_sentence, bigger_accuracy, bigger_mismatches, bigger_results = classify_symbols(
        symbols_path='../lab6/results/symbols_bigger',
        features_path='../lab5/results/features.csv',
        output_csv='results/bigger_classification.csv'
    )

    print("\n\nРезультаты классификации (104pt):")
    print("=" * 50)
    print(f"Ожидаемая строка: {target}")
    print(f"Распознанная строка: {bigger_sentence}")
    print(f"\nТочность распознавания: {bigger_accuracy:.2f}%")

    print_mismatches(target, bigger_sentence)

    print("\nЛучшие гипотезы для каждого символа:")
    for i, hypotheses in enumerate(bigger_results):
        best = hypotheses[0] if hypotheses else ("?", 0.0)
        status = "✓" if best[0] == target[i] else "✗"
        print(f"Символ {i + 1:2d}: {status} {best[0]} (сходство: {best[1]:.4f})")

    print("\n\nСравнение результатов:")
    print(f"Изменение точности: {bigger_accuracy - original_accuracy:.2f}%")
    if bigger_accuracy > original_accuracy:
        print("Увеличение размера шрифта улучшило точность.")
    elif bigger_accuracy < original_accuracy:
        print("Увеличение размера шрифта ухудшило точность.")
    else:
        print("Точность не изменилась.")