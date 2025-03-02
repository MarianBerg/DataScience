
import numpy as np
import pandas as pd


def create_distribution_from_data(data, column_name):
    value_counts = data[column_name].value_counts()

    value_counts_tidy = value_counts.reset_index()

    distribution = value_counts_tidy['count'] / value_counts_tidy['count'].sum()

    return distribution



def calculate_entropy_from_distribution(distribution):
    entropy = 0

    for value in distribution:
        if value != 0:
            entropy += value * np.log2(1/ value)

    return entropy


def calculate_entropy_random_variable_under_feature_from_dataframe(dataframe, random_variable_column, feature_column):

    # Pivot the DataFrame
    feature = dataframe.pivot_table(index = feature_column, columns =  random_variable_column, aggfunc = 'size', fill_value = 0)
    
    #make the data tidy
    tidy_feature = feature.reset_index().melt(id_vars = feature_column, var_name = random_variable_column, value_name = 'Anzahl')

    #group the data by the feature
    grouped = tidy_feature.groupby(feature_column)

    #calculate the total occurences, should for this data always be 14
    total_occurences = tidy_feature['Anzahl'].sum()

    #calculate the entropy for every value the feature can take and add them up weighted by the relative frequency of the value
    total_entropy = 0
    for name, group in grouped:
        distribution = group['Anzahl'] / group['Anzahl'].sum()
        distribution_correct_shape = distribution.to_numpy()

        entropy = calculate_entropy_from_distribution(distribution_correct_shape)
        weighted_entropy = entropy * (group['Anzahl'].sum() / total_occurences)

        total_entropy += weighted_entropy


    return total_entropy


def main():
    #read in data
    project_data_sheet1 = pd.read_excel("K4.0026_1.0_1.5.C.01_ProjectData.xlsx", sheet_name = 'Tabellenblatt1')



    #Sheet 1 

    print("Sheet1:\n")

    #calculate entropy from random_variable_column
    distribution = create_distribution_from_data(project_data_sheet1, 'Draussen Essen')


    entropy_without_feature = calculate_entropy_from_distribution(distribution)

    print("Entropy Draussen Essen: {}".format(entropy_without_feature))



    #calculate the entropys for all Features except 'Tag'
    entropy = 0
    for column in project_data_sheet1.columns:
        if column != 'Draussen Essen' and column != 'Tag':


            entropy = calculate_entropy_random_variable_under_feature_from_dataframe(project_data_sheet1, 'Draussen Essen', column)

            print("Information {}: {}".format(column, entropy_without_feature - entropy))



    #Sheet 2

    print("\nSheet2:\n")

    #read in data
    project_data_sheet2 = pd.read_excel("K4.0026_1.0_1.5.C.01_ProjectData.xlsx", sheet_name = 'Tabellenblatt2')

    for column in project_data_sheet2.columns:
        #print arithmetic mean, median and standard deviation of every feature
        if column != 'Aussenverkauf':
            print("{} arithmetic_mean: {}".format(column, project_data_sheet2[column].mean()))
            print("{} median: {}".format(column, project_data_sheet2[column].median()))
            print("{} standard deviation: {}".format(column, project_data_sheet2[column].std()))

    #calculate the correlation between numerical features.
    for index1, column1 in enumerate(project_data_sheet2.columns):

        for index2, column2 in enumerate(project_data_sheet2.columns):
            if index1 < index2 and column1 != 'Aussenverkauf' and column2 != 'Aussenverkauf':
                print("{}_{} correlation: {}".format(column1, column2, project_data_sheet2[column1].corr(project_data_sheet2[column2])))



    #cathegorize data for feature entropy analysis

    #    Windgeschwindigkeit
    #
    #Die Windgeschwindigkeit kann in verschiedene Stärken eingeteilt werden: 
    #
    #    1-5 km/h
    #    6-11 km/h
    #    12-19 km/h

    def cathegorize_windgeschwindigkeit(windgeschwindigkeit):
        if windgeschwindigkeit <= 5:
            return 'schwach'
        elif windgeschwindigkeit <= 11:
            return 'mittel'
        else:
            return 'stark'
        
    project_data_sheet2['Windgeschwindigkeit'] = project_data_sheet2['Windgeschwindigkeit'].apply(cathegorize_windgeschwindigkeit)




    #Temperatur
    #
    #Teile die Temperatur in die folgenden Kategorien ein:
    #
    #    Kleiner als 18 Grad: kalt
    #    Größer als 28 Grad: heiss
    #    Dazwischen: mild
    #

    def cathegorize_temperatur(temperatur):
        if temperatur < 18:
            return 'kalt'
        elif temperatur <= 28:
            return 'mild'
        else:
            return 'heiss'
        
    project_data_sheet2['Temperatur'] = project_data_sheet2['Temperatur'].apply(cathegorize_temperatur)

 


    #Luftfeuchtigkeit
    #
    #Teile die Luftfeuchtigkeit in die folgenden Kategorien ein:
    #
    #    Größer als 5: hoch
    #    Kleiner: niedrig

    def cathegorize_luftfeuchtigkeit(luftfeuchtigkeit):
        if luftfeuchtigkeit <= 5:
            return 'niedrig'
        else:
            return 'hoch'
        
    project_data_sheet2['Luftfeuchtigkeit'] = project_data_sheet2['Luftfeuchtigkeit'].apply(cathegorize_luftfeuchtigkeit)


    #calculate the entropy of 'Aussenverkauf' and its features
    print("\nEntropies:\n")
    distribution = create_distribution_from_data(project_data_sheet2, 'Aussenverkauf')

    entropy_without_feature = calculate_entropy_from_distribution(distribution)
    print("Entropy Aussenverkauf: {}".format(entropy_without_feature))

    #calculate the entropys for all Features
    entropy = 0
    for column in project_data_sheet2.columns:
        if column != 'Aussenverkauf':


            entropy = calculate_entropy_random_variable_under_feature_from_dataframe(project_data_sheet2, 'Aussenverkauf', column)

            print("Information {}: {}".format(column, entropy_without_feature - entropy))





if __name__ == "__main__":
    main()