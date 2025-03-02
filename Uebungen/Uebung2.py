import random
import math

def main():
    print("Hello")

    outcomes = ["head", "tails"]
    probabilities = [0.9, 0.1]    
    
    list_of_outcomes = random.choices(outcomes, weights = probabilities, k = 10)

    print(list_of_outcomes)

    new = set(list_of_outcomes)#so kriegt man in python die einzigartigen elemente

    print(new)
    print(list(new))
    elementary_data = list(new)
    anzahl = []
    relative_haeufigkeit = []
    length_list = len(list_of_outcomes)

    for i in range(0, len(elementary_data)):
        anzahl.append(list_of_outcomes.count(elementary_data[i]))
        relative_haeufigkeit.append(anzahl[i] / length_list)

    print(relative_haeufigkeit)












if __name__ == "__main__":
    main()