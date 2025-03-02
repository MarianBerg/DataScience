import statistics as stat

def main():
    data = [43, 8, 2, 99, 12, 23, 1, 32, 12,]

    mean = stat.mean(data)
    sigma = stat.stdev(data)

    for i in range(0, len(data)):
        data[i] = (data[i] - mean) / sigma

    print(data)
    

if __name__ == "__main__":
    main()