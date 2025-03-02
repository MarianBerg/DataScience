from faker import Faker
import pandas as pd

#Erzeuge ein pandas-Dataframe mit synthetischen deutschen Nutzerdaten. 
#Schreibe dafür eine Funktion, die eine zufällige einzigartige User ID (Zahlen zwischen 0 und 100), 
#einen Namen, eine Adresse und eine Geo-Location erzeugt. Mit dieser Funktion erzeugst du dann den Datensatz mit 10 Einträgen.



def data_entry(fake, data): 
    id = fake.unique.random_int(min = 0, max = 100)
    name = fake.name()
    adress = fake.address()
    laengengrad = fake.longitude()
    breitengrad = fake.latitude()

    data.loc[len(data)] = [id, name, adress, laengengrad, breitengrad]


columns = ["ID", "Name", "Adresse", "Laengengrad", "Breitengrad"]
data = pd.DataFrame(columns = columns)
fake = Faker("DE_de")

for i in range(0, 10):
    data_entry(fake, data)


print(data)