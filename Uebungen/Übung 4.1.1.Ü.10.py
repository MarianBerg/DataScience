#Aufgabe:
#
#Ein weiteres wichtiges Feature in Datensätzen sind Zeitangaben. An welchem Wochentag wurde eine bestimmte Tätigkeit ausgeführt? Welcher Monat ist am beliebtesten bei den Kunden und Kundinnen? Um Zeitangaben synthetisch zu erzeugen, bietet die Faker-Bibliothek ebenfalls einige Methoden. Recherchiere die Funktionen zur Generierung von
#
#    einem Jahr,
#    einem Monat,
#    dem Namen eines Monats,
#    einem Wochentag,
#    einer Zeitzone.

print(fake.year())
print(fake.month())
print(fake.month_name())
print(fake.day_of_week())
print(fake.timezone())