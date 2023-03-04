import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def datacleaning():   
    diabetes_df=pd.read_csv('C:/Users/vasuv/OneDrive/Desktop/Course_Structure/Second_Semester/Data_Mining/Assignment_4/dataset_diabetes/dataset_diabetes/diabetic_data.csv')
    diabetes_df.head(5)
    missing_values=(diabetes_df=='?').sum()
    missing_values
    diabetes_df.replace('?',np.nan, inplace=True)
    diabetes_df['race'].fillna('others',inplace = True)
    diabetes_df['race'].isna().any()
    diabetes_df['weight'].isna().any()
    mode_weight_by_age_group = diabetes_df.groupby('age')['weight'].apply(lambda x: x.mode()[0])
    diabetes_df['weight'].fillna(diabetes_df['age'].map(mode_weight_by_age_group), inplace=True)
    diabetes_df['weight'].isna().any()
    pd.set_option('display.max_columns', None)
    diabetes_df['medical_specialty'].fillna(diabetes_df['medical_specialty'].mode()[0],inplace = True) 
    diabetes_df['payer_code'].fillna(diabetes_df['payer_code'].mode()[0],inplace = True)
    diabetes_df = diabetes_df.dropna(subset=['diag_1'])
    diabetes_df = diabetes_df.dropna(subset=['diag_2'])
    diabetes_df = diabetes_df.dropna(subset=['diag_3'])
    
    for col in diabetes_df.columns:
        if diabetes_df[col].dtype == 'object':
            diabetes_df[col] = pd.factorize(diabetes_df[col])[0]
        elif diabetes_df[col].dtype.name == 'category':
            diabetes_df[col] = diabetes_df[col].cat.codes 
    diabetes_df = diabetes_df.select_dtypes(include=['float64', 'int64'])
    diabetes_df = diabetes_df.apply(pd.to_numeric)
    return diabetes_df

def lloyds_method(Sample_Space, number_of_centroids, maximum_iterations=1, threshold_value=10^-4):
    total_number_values = Sample_Space.shape[0]
    total_features = Sample_Space.shape[1]
    cluster_array = np.zeros(total_number_values, dtype=int)
    Centrod_Index_Values = np.random.choice(total_number_values, number_of_centroids)
    centroids = []
    for centroid_index_value in Centrod_Index_Values:
        centroids.append(Sample_Space.iloc[centroid_index_value, :])
    for no_use_value in range(0, maximum_iterations):
        for i in range(0, total_number_values):
            distances = np.zeros(number_of_centroids)
            for j in range(0, number_of_centroids):
                distances[j] = np.sqrt(np.sum(np.power(Sample_Space.iloc[i, :] - centroids[j], 2)))
            cluster_array[i] = np.argmin(distances)
        updated_centroids = np.zeros((number_of_centroids, total_features))
        new_loss = 0
        for j in range(0, number_of_centroids):
            index_position = np.where(cluster_array == j)
            Updated_Sample_Space = Sample_Space.iloc[index_position]
            updated_centroids[j] = Updated_Sample_Space.mean(axis=0)
            for i in range(0, Updated_Sample_Space.shape[0]):
                new_loss = new_loss + np.sum(np.power(Updated_Sample_Space.iloc[i, :] - centroids[j], 2))
        centroids = updated_centroids
    return centroids, cluster_array, new_loss

def plot(Sample_Space,centers,cluster_array):
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()
    s1 = plt.subplot(1, 2, 1)
    for i in range(50):
        s1.scatter(Sample_Space.iloc[:, i], Sample_Space.iloc[:, i], c = cluster_array, s = 2)
        s1.scatter(centers[:, i], centers[:,i], c = "r", s = 20)
    plt.show() 

def main():
    original_diabetes_df = datacleaning()
    diabetes_df = original_diabetes_df[:1000]
    Sample_Space = diabetes_df
    centers, cluster_array, value_lost = lloyds_method(Sample_Space, 2)
    plot(Sample_Space,centers,cluster_array)
main()

