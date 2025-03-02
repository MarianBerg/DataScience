#Modus berechnen

from collections import Counter

data = [3, 9, 6, 5, 45, 7, 4, 6, 4, 9, 4]

element_count = Counter(data)
print(element_count)

print(element_count.most_common(1)[0])