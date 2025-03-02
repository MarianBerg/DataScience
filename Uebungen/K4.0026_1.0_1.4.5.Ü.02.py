


def main():
    data = [32, 9, 87, 2, 73, 21, 9, 62]
    data.sort()
    max = data[-1]
    min = data[0]

    min_max_scaled_data = []

    for i in range(0, len(data)):
        min_max_scaled_data.append((data[i] - min) / (max - min))


    print(min_max_scaled_data)



if __name__ == "__main__":
    main()