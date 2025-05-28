from gensim.models import KeyedVectors

# Завантаження моделі (може зайняти багато пам’яті та часу)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# 1. 10 найбільш схожих слів до "dog"
similar_words = model.most_similar("dog", topn=10)
print("Слова, схожі на 'dog':")
for word, score in similar_words:
    print(f"{word} — {score:.4f}")

# 2. Семантична арифметика: cat - male + female
result = model.most_similar(positive=["cat", "female"], negative=["male"], topn=1)
print("\nРезультат семантичної арифметики (cat - male + female):")
print(result[0])
