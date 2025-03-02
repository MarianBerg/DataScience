
def main():
    X = [3, 8, 19, 22, 31, 42, 48, 52, 54, 61]
    Y = [150, 180, 420, 480, 660, 1000, 1300, 1500, 1600, 1710]

    r = kovarianz(X, Y) / ( varianz(X) * varianz(Y))
    print(r)






def varianz(data):
    mue = sum(data) / len(data)

    added_squares = 0
    for i in range(0, len(data)):
        added_squares += (mue - data[i]) ** 2

    varianz = added_squares / len(data)
    return varianz

def kovarianz(X, Y):
    x_mean = sum(X) / len(X)
    y_mean = sum(Y) / len(Y)

    kovarianz = 0

    for i in range(0, len(X)):
        kovarianz += (x_mean - X[i]) * (y_mean - Y[i])

    return kovarianz /(len(X))


if __name__ == "__main__":
    main()