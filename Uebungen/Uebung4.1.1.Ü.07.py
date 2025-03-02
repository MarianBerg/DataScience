from faker import Faker
import pandas as pd

#In dieser Aufgabe wollen wir uns genauer anschauen, wie wir Texte mit der Fake-Bibliothek generieren können. 
#Vorher haben wir ja schon die fake.text()-Methode kennengelernt, die uns eine zufällige Kombination von Worten ausspuckt. 
#In dieser Aufgabe wollen wir etwas tiefergehen und mit der fake.sentence()-Methode einen Satz erzeugen, 
#der bestimmte, von uns vorgegebene Worte enthält.



#Erzeuge nun eine Datenbank mit 10 Personenprofilen. 
#Jede Person soll: 
#einen Namen, eine Adresse, eine Kreditkartennummer und einen Text enthalten, 
#der folgende Worte enthält: 
#liebt, kauft, hasst, kauft nicht, Salat, Steak, Karotten.

fake = Faker('de_DE')
data = pd.DataFrame()

word_list = ['liebt', 'kauft', 'hasst', 'kauft nicht', 'Salat', 'Steak', 'Karotten']
for i in range(0, 11):
    data.loc[i, 'Name'] = fake.name()
    data.loc[i, 'Adresse'] = fake.address()
    data.loc[i, 'Kreditkartennummer'] = fake.credit_card_number()
    data.loc[i, 'Text'] = fake.sentence(ext_word_list = word_list)


print(data)