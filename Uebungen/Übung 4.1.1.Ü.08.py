#Aufgabe:
#
#Bei den bisher erklärten Methoden werden wir bei jeder Ausführung des Codes neue Outputs bekommen. Daher ist die Wahrscheinlichkeit, dass ein erzeugter Datenpunkt einzigartig ist, recht gering. Das wiederum führt dazu, dass bei beispielsweise tausendfacher Ausführung der Faker- Methoden Duplikate entstehen können. Um das zu vermeiden, gibt es in der Faker-Bibliothek eine Möglichkeit, um Datenpunkt einzigartig, oder unique, zu machen.
#
#Recherchiere nun, wie du 10 einzigartige Namen erzeugen kannst!
#
#
#	
#
#Hinweis: Mit List Comprehension lässt sich einfach eine Liste von Namen erzeugen
import faker as fake
names = [fake.unique.name() for i in range(10)]
for i in range (0, len(names)):
    print(names[i])