from collections import Counter 

data = [1, 9, 8, 4, 50, 4, 9]

data.sort()

print(data)

elements_count = Counter(data)

print(elements_count)
print(len(data))
length = len(data)

# Sort the Counter by keys (elements) and create a new dictionary
sorted_by_keys = dict(sorted(elements_count.items()))
print(sorted_by_keys)
mean_index = length / 2

mean = 0

index_count = 0
for key in sorted_by_keys:
    index_count += sorted_by_keys[key]
    if index_count >= mean_index:
        mean = key
        break

print(mean)
