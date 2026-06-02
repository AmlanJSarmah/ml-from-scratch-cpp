import sys
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter

def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if not pd.api.types.is_numeric_dtype(y):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    n_clusters = len(set(y))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X)

    cluster_to_label = {}

    for cluster in range(n_clusters):
        indices = clusters == cluster

        if indices.sum() == 0:
            continue

        labels = y[indices]

        majority_label = Counter(labels).most_common(1)[0][0]

        cluster_to_label[cluster] = majority_label

    y_pred = [
        cluster_to_label[c]
        for c in clusters
    ]

    accuracy = accuracy_score(y, y_pred)

    print(accuracy)

if __name__ == "__main__":
    main()