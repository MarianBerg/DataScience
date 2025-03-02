





def main():
    data = [4, 9, 7, 1, 23, 45, 21]
    data.sort()
    max = data[-1]
    
    for i in range(0, len(data)):
        data[i] = data[i] / max

    print(data)


if __name__ == "__main__":
    main()