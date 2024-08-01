import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el vectorizador
with open(r'D:\AI Engineer Core\R4. Deep Learning y Natural Language processing para generar un chatbot\Sprint 3\Sprint 3\Entregable\data\conversations_vectorizer_bow.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Cargar etiquetas y vocabulario
with open(r'D:\AI Engineer Core\R4. Deep Learning y Natural Language processing para generar un chatbot\Sprint 3\Sprint 3\Entregable\data\tags.pkl', 'rb') as file:
    tags = pickle.load(file)

# Cargar datos de conversaciones
with open(r'D:\AI Engineer Core\R4. Deep Learning y Natural Language processing para generar un chatbot\Sprint 3\Sprint 3\Entregable\data\conversations.json', 'r') as file:
    conversations = json.load(file)

# Extraer patrones y etiquetas
patterns = []
categories = []
for convo in conversations:
    for pattern in convo['patterns']:
        patterns.append(pattern)
        categories.append(convo['tag'])

# Transformar los patrones en características numéricas utilizando el vectorizador
X = vectorizer.transform(patterns).toarray()
y = categories

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión logística
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Guardar el modelo entrenado
with open(r'D:\AI Engineer Core\R4. Deep Learning y Natural Language processing para generar un chatbot\Sprint 3\Sprint 3\Entregable\data\chatbot_model.pkl', 'wb') as file:
    pickle.dump(model, file)
