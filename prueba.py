#Importamos librerías

import streamlit as st
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Definición de variables
min_score = 300
max_score = 850
user_data = [23, 5000, 3, 0, 50, 0.6, 2, 0, 0, 0, 0]

#Cargar los datos
datos = pd.read_csv("credit_risk_dataset.csv", delimiter=",")

# Verificamos y borramos datos nulos y duplicados
datos.isnull().sum()
datos.duplicated().sum()
datos = datos.dropna()
datos = datos.drop_duplicates()

# Realizamos un label encoder para convertir las variables cualitativas en categóricas por niveles
LabelEncoder = LabelEncoder()
datos["person_home_ownership_num"] = LabelEncoder.fit_transform(datos["person_home_ownership"])
datos["loan_intent_num"] = LabelEncoder.fit_transform(datos["loan_intent"])
datos["loan_grade_num"] = LabelEncoder.fit_transform(datos["loan_grade"])
datos["cb_person_default_on_file_num"] = LabelEncoder.fit_transform(datos["cb_person_default_on_file"])

# Ahora eliminamos las variables cualitativas, pues son redundantes.
datos = datos.drop(["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"],axis=1)

# Realizamos un train-test split para dividir los datos en datos de entrenamiento y datos de testeo
X = datos.drop("loan_status", axis=1)  # Variables predictoras
y = datos["loan_status"]  # Variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Denifimos el modelo como una regresión logística (pues queremos clasificar)
model = LogisticRegression()
model.fit(X_train, y_train)

# aquí alojamos el score de la base de datos sin los datos de usuario
prob_scores = model.predict_proba(X)[:, 1]

# Definimos la función que estandariza el credit score entre 300 y 850
def scale_credit_score(probability, min_prob, max_prob):
    return min_prob + (max_prob-min_prob)*probability

# Definimos la función que procesa los datos ingresados por el usuario y devuelve el score credit escalado
# y_pred es la variable respuesta del modelo que nos dice si el crédito fue aprobado o no
def procesar_datos(data):
    suma_menores = 0
    pandas_data = pd.DataFrame(data)
    pandas_data = pandas_data.transpose()
    pandas_data.columns = ["person_age","person_income","person_emp_length","loan_amnt", "loan_int_rate",
                           "loan_percent_income", "cb_person_cred_hist_length","person_home_ownership_num",
                           "loan_intent_num","loan_grade_num","cb_person_default_on_file_num"]

    y_pred = model.predict(pandas_data)[0]

    prob_score_data = model.predict_proba(pandas_data)[:, 1]
    pandas_data['credit_score'] = prob_score_data

    pandas_data['credit_score_scaled'] = scale_credit_score(pandas_data['credit_score'], min_score, max_score)
    pandas_data['credit_score_scaled'] = pandas_data['credit_score_scaled'].astype(int)

    for i in prob_scores:
        if i < prob_score_data:
            suma_menores += 1
        else:
            pass

    percentil = (suma_menores/len(prob_scores))*100

    scale_credit_score_int = {
        "credit_score_scaled": int(pandas_data.loc[0,"credit_score_scaled"]),
        "y_pred": int(y_pred),
        "percentil": percentil
    }

    return scale_credit_score_int

    # return scale_credit_score_int


print(procesar_datos(user_data))

# prob_scores = prob_scores.append(prob_score_data)

# x = np.append(prob_scores, prob_score_data)