#Aufgabe:
#
#Gerade, wenn man als Data Scientist mit Kundendaten zu tun hat, sind die verwendeten Währungen von großer Relevanz. Daher ist es für die Erzeugung von synthetischen Daten auch wichtig, Währungen mit ihren jeweiligen Kürzeln generieren zu können. Recherchiere, wie du mit der Faker-Bibliothek Währungen und den dazugehörigen Code erzeugen kannst!

import faker as fake

print(fake.currency()) 
print(fake.currency_name()) 
print(fake.currency_code())