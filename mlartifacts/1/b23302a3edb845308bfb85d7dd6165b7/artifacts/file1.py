import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()
x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42,stratify=y)

max_depth = 50
n_estimators = 50

# Mention your experiment name
mlflow.set_experiment("Exp1")

with mlflow.start_run():
    rf =  RandomForestClassifier(max_depth = max_depth,n_estimators=n_estimators, random_state=1)
    rf.fit(x_train,y_train)

    y_pred = rf.predict(x_test)

    acc = accuracy_score(y_test,y_pred)

    
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    mlflow.log_metric("accuracy", acc)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # add tags
    mlflow.set_tag("Author", "Jayesh" , "model", "RandomForest")

    print(f"Accuracy: {acc}")
