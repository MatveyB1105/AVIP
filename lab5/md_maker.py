import pandas as pd

# Считываем данные из Excel
df = pd.read_excel('results/Base.xlsx', sheet_name='features')

# Шаблон для нового файла
template = """# Лабораторная работа №5. Выделение признаков символов.
Лабораторная работа проделывалась на грузинском алфавите. Тип букв -- обычные прописные,
шрифт NotoSansGeorgian-Regular, размер 52. В качестве демонстрации были выбраны 
5 символов, представляющих собой наибольший интерес. Квадраты для расчета веса черного расположены
следующим образом:

![](table.jpg)

{letters}
"""

# Шаблон для каждой буквы
letter_template = """
## Символ {letter} ({name})
Прямое и инвертированное сгенерированные изображения:

![](alphabet/direct/letter_{index:02d}.png)
![](alphabet/inverse/letter_{index:02d}.png)

Профили буквы:

![](results/profiles/x/letter_{index:02d}.png "Профиль по Х")
![](results/profiles/y/letter_{index:02d}.png "Профиль по Y")

Признаки:
1. Вес первого квадрата: {weight_I}
2. Нормированный(на четверть площади) вес черного: {relative_weight_I}
3. Вес второго квадрата: {weight_II}
4. Нормированный(на четверть площади) вес черного: {relative_weight_II}
5. Вес третьего квадрата: {weight_III}
6. Нормированный(на четверть площади) вес черного: {relative_weight_III}
7. Вес четвертого квадрата: {weight_IV}
8. Нормированный(на четверть площади) вес черного: {relative_weight_IV}
9. Центр масс: ({center_x}, {center_y})
10. Нормированный центр масс: ({relative_center_x}, {relative_center_y})
11. Моменты инерции: ({inertia_x}, {inertia_y})
12. Нормированные моменты инерции: ({relative_inertia_x}, {relative_inertia_y})
"""

# Список букв и их названий
letter_names = {
    "ბ": "ба́ни",
    "ზ": "зéни",
    "პ": "па́ри",
    "ქ": "ка́ни",
    "ჭ": "ча́ри"
}

# Создаем список букв
letters_content = []

# Проходим по каждой строке в таблице с использованием enumerate
for idx, (_, row) in enumerate(df.iterrows(), start=1):
    if row["letter"] in letter_names:
        letter_data = {
            "letter": row["letter"],
            "name": letter_names[row["letter"]],  # Используем правильное имя для каждой буквы
            "index": idx,  # Используем порядковый номер строки
            "weight_I": row["weight_I"],
            "relative_weight_I": row["relative_weight_I"],
            "weight_II": row["weight_II"],
            "relative_weight_II": row["relative_weight_II"],
            "weight_III": row["weight_III"],
            "relative_weight_III": row["relative_weight_III"],
            "weight_IV": row["weight_IV"],
            "relative_weight_IV": row["relative_weight_IV"],
            "center_x": row["center_x"],
            "center_y": row["center_y"],
            "relative_center_x": row["relative_center_x"],
            "relative_center_y": row["relative_center_y"],
            "inertia_x": row["inertia_x"],
            "inertia_y": row["inertia_y"],
            "relative_inertia_x": row["relative_inertia_x"],
            "relative_inertia_y": row["relative_inertia_y"],
        }
        # Добавляем отформатированную букву в список
        letters_content.append(letter_template.format(**letter_data))

# Объединяем все буквы в одну строку
letters_content = "\n".join(letters_content)

# Вставляем буквы в шаблон
new_content = template.format(letters=letters_content)

# Записываем новый файл
with open("README_updated.md", "w", encoding="utf-8") as file:
    file.write(new_content)

print("Новый файл README_updated.md успешно создан!")