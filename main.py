
import pandas as pd
from src.feature_engg import FeatureEngineer
from src.preprocessing import Preprocessor
from src.kmeans_clustering import KMeansClusterer
from src.dbscan import DBSCANClusterer

def main():

    df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\customer segmentation\data\marketing_data_raw.csv")

    print(f"Loaded raw data: {df.shape}")

    prep = Preprocessor()
    df_transformed = prep.transform(df)

    fe = FeatureEngineer()
    fe.fit(df_transformed)
    X = fe.transforms(df_transformed)

    clusterer = KMeansClusterer(n_clusters=3)
    labels = clusterer.train(X)
    df1 = df_transformed.copy()
    df1["Cluster"] = labels
    print("KMeans clustering completed.")

    clusterer.save_model()

    df1.to_csv("data/KMeansclustered_customers.csv", index=False)

    clusterer2 = DBSCANClusterer()
    labels = clusterer2.train(X)
    df2 = df_transformed.copy()
    df2["Cluster"] = labels
    print("DBSCAN clustering completed.")

    clusterer2.save_model()

    df2.to_csv("data/DBScanclustered_customers.csv", index=False)

if __name__ == "__main__":
    main()
