import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib # Usamos joblib para serializar los modelos

# 1. Cargar y limpiar datos
df = pd.read_csv('penguins.csv')
df = df.dropna()

# 2. Codificar la variable objetivo (Species)
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])
# Creamos el diccionario de mapeo para el output
unique_penguin_mapping = {index: label for index, label in enumerate(le.classes_)}

# 3. Preparar y Codificar Variables Predictoras
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
        'body_mass_g', 'island', 'sex']]
X = pd.get_dummies(X) # Aplica One-Hot Encoding a 'island' y 'sex'

# IMPORTANTE: Definir y mantener el orden de las columnas que usará el modelo
# Estas columnas deben coincidir con la lista de variables de entrada en tu script de Streamlit
feature_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 
                'island_Biscoe', 'island_Dream', 'island_Torgerson', 
                'sex_Female', 'sex_Male']

# Aseguramos que solo estas columnas estén en X y en el orden correcto
X = X.reindex(columns=feature_cols, fill_value=0) 
y = df['species_encoded']

# 4. Entrenar el Modelo
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)

# 5. Guardar los artefactos del modelo (Usando joblib)
# Guarda el modelo
joblib.dump(rfc, 'random_forest_penguin.pickle')

# Guarda el mapeo de las etiquetas
joblib.dump(unique_penguin_mapping, 'output_penguin.pickle')

print("¡Archivos .pickle (joblib) creados y listos para Streamlit!")