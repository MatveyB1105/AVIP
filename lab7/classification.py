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
        print("-------|-----------|------------")
        for idx, expected, actual in mismatches:
            print(f"{idx + 1:6d} | {expected:9s} | {actual:11s}")
    else:
        print("\nВсе символы распознаны верно")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    features = pd.read_csv('../lab5/results/features.csv')
    results = []
    sentence = ""
    mismatch_indices = []

    for i in range(len(target)):
        try:
            symbol = FeatureImage(Image.open(f'../lab6/results/symbols/letter_{i}.png'), invert=True)
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

            # Запоминаем несовпадения
            if recognized_char != target[i]:
                mismatch_indices.append((i, target[i], recognized_char))

        except FileNotFoundError:
            print(f"Ошибка: файл символа {i} не найден")
            recognized_char = "?"
            sentence += recognized_char
            mismatch_indices.append((i, target[i], "?"))
            continue


    with open('results/classification_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Symbol Index', 'Hypotheses (Letter, Proximity)'])
        for i, hypotheses in enumerate(results):
            writer.writerow([i, hypotheses])

    correct = sum(1 for expected, actual in zip(target, sentence) if expected == actual)
    accuracy = correct / len(target) * 100

    print("\nРезультаты классификации:")
    print("=" * 50)
    print(f"Ожидаемая строка: {target}")
    print(f"Распознанная строка: {sentence}")
    print(f"\nТочность распознавания: {accuracy:.2f}%")

    print_mismatches(target, sentence)

    print("\nЛучшие гипотезы для каждого символа:")
    for i, hypotheses in enumerate(results):
        best = hypotheses[0] if hypotheses else ("?", 0.0)
        status = "✓" if best[0] == target[i] else "✗"
        print(f"Символ {i + 1:2d}: {status} {best[0]} (сходство: {best[1]:.4f})")