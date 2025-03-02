

def main():
    X = [20, 26, 32, 48, 26, 30, 30, 40]
    Y = [270, 460, 512, 550, 360, 399, 419, 390]

    x_mean = sum(X) / len(X)
    y_mean = sum(Y) / len(Y)

    print(x_mean)
    print(y_mean)

    print(varianz(X))
    print(varianz(Y))
    kovarianz = 0

    for i in range(0, len(X)):
        kovarianz += (x_mean - X[i]) * (y_mean - Y[i])

    print(kovarianz /(len(X)))

    #r = korrelation / (varianz(X) * varianz(Y) * (len(X)-1) )

    #print(r)





def varianz(data):
    mue = sum(data) / len(data)

    added_squares = 0
    for i in range(0, len(data)):
        added_squares += (mue - data[i]) ** 2

    varianz = added_squares / len(data)
    return varianz


if __name__ == "__main__":
    main()