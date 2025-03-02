import pandas as pd

data = pd.read_csv('K4.0026_1.0_1.4.4.Ü.01_dirtydata.csv')

#Ersetze in Spalten mit numerischen Werten die leeren Zellen durch den Mittelwert.

numeric_columns = data.select_dtypes(include=['number'])


column_means = numeric_columns.mean()


data_filled = data.copy()
data_filled[numeric_columns.columns] = numeric_columns.fillna(column_means)

print(data_filled)


#Korrigiere die Datumsangaben (einheitliches Format, fehlende Angaben löschen).



data_filled['Datum'] = data_filled['Datum'].astype(str)



data_filled.drop(22, inplace= True) #konnte pandas nicht dazu bringen nan zu erkennen, selbst nach erzwungener konversion


data_filled.iloc[25, data_filled.columns.get_loc('Datum')] = '2020/12/26' #sieht in der Ausgabe nicht wie ein String aus, keine Ahnung woher das kommt.


print(data_filled)


#Bearbeite falsche Werte (Ausreißer aus den Daten).
for x in data_filled.index:
    if (data_filled.loc[x, 'Dauer'] > 80 ) or (data_filled.loc[x, 'Kunden'] > 200) or (data_filled.loc[x, 'MinKauf'] > 200) or (data_filled.loc[x, 'MaxKauf'] > 600.0):
        data_filled.drop(x, inplace = True)

print(data_filled)
#Entferne Duplikate.

data_filled.drop_duplicates(inplace = True)

print(data_filled)