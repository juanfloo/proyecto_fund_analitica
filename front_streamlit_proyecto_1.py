#Importamos librerías

import streamlit as st
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Definición de variables
min_score = 300
max_score = 850

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

# Definimos la función que estandariza el credit score entre 300 y 850
def scale_credit_score(probability, min_prob, max_prob):
    return min_prob + (max_prob-min_prob)*probability

# Definimos la función que procesa los datos ingresados por el usuario y devuelve el score credit escalado
# y_pred es la variable respuesta del modelo que nos dice si el crédito fue aprobado o no
def procesar_datos(data):
    pandas_data = pd.DataFrame(data)
    pandas_data = pandas_data.transpose()
    pandas_data.columns = ["person_age","person_income","person_emp_length","loan_amnt", "loan_int_rate",
                           "loan_percent_income", "cb_person_cred_hist_length","person_home_ownership_num",
                           "loan_intent_num","loan_grade_num","cb_person_default_on_file_num"]

    y_pred = model.predict(pandas_data)[0]

    pandas_data['credit_score'] = model.predict_proba(pandas_data)[:, 1]

    pandas_data['credit_score_scaled'] = scale_credit_score(pandas_data['credit_score'], min_score, max_score)
    pandas_data['credit_score_scaled'] = pandas_data['credit_score_scaled'].astype(int)

    scale_credit_score_int = {
        "credit_score_scaled": int(pandas_data.loc[0,"credit_score_scaled"]),
        "y_pred": int(y_pred)
    }

    return scale_credit_score_int


#TODO====================================================================================================================================
#TODO -> Aquí comienza el código de la página

st.markdown("# :orange[Quick Cash]")
st.markdown("## Tu mejor aliado para solicitar crédito fácil y rápido")
st.markdown('''Aquí podrás consultar de manera fácil y rápida si puedes solicitar un crédito
            en una entidad bancaria solo con pocos datos personales y de historial crediticio.''')
st.markdown('''Si te interesa conocer más a cerca de este proyecto, puedes consultar nuestro blog, donde
            explicamos la construcción de esta solución web''')
st.markdown("### Puedes ver nuestro video promocional [Aquí](https://www.google.com/?hl=es)")


edad = st.text_input("Ingrese su edad en años", value=0)
ingresos_anuales = st.number_input("Indique sus ingresos anuales en miles de dólares", value=1)
tiempo_empleado = st.number_input("Indique cuántos años lleva empleado")
prestamo_intencion = st.number_input("Indique cuánto dinero desea prestar", value=1)
tasa_interes = 13.25
razon_prestamo_ingreso = prestamo_intencion/ingresos_anuales
tiempo_sistema = st.number_input("Indique cuántos años lleva activo en el sistema financiero")
tipo_casa = st.selectbox(
    "Indique su tipo de vivienda",
    ("Alquilada", "Propia", "Hipotecada", "Otro"))
tipo_credito = st.selectbox(
    "Indique para qué desea el crédito",
    ("Personal", "Educación", "Salud", "Emprendimiento", "Remodelación del hogar", "consolidación de deuda"))
grado_credito = st.selectbox(
    "Indique su grado crediticio",
    ("A", "B", "C", "D", "E", "F", "G"))
default = st.selectbox(
    "Indique si se encuentra en mora con alguna obligación bancaria",
    ("Sí", "No"))


def tipo_casa_num(casa):
    if casa == "Alquilada":
        respuesta = "3"
    elif casa == "Propia":
        respuesta = "2"
    elif casa == "Hipotecada":
        respuesta = "0"
    elif casa == "Otro":
        respuesta = "1"

    return respuesta

def tipo_credito_num(credito):
    if credito == "Personal":
        respuesta = "4"
    elif credito == "Educación":
        respuesta = "1"
    elif credito == "Salud":
        respuesta = "3"
    elif credito == "Emprendimiento":
        respuesta = "5"
    elif credito == "Remodelación del hogar":
        respuesta = "2"
    elif credito == "consolidación de deuda":
        respuesta = "0"

    return respuesta

def grado_num(Grado):
    if Grado == "A":
        respuesta = "0"
    elif Grado == "B":
        respuesta = "1"
    elif Grado == "C":
        respuesta = "2"
    elif Grado == "D":
        respuesta = "3"
    elif Grado == "E":
        respuesta = "5"
    elif Grado == "F":
        respuesta = "5"
    elif Grado == "G":
        respuesta = "6"

    return respuesta

def verif_default(verif_def):
    if verif_def == "Sí":
        respuesta = "1"
    elif verif_def == "No":
        respuesta = "0"

    return respuesta

tipo_casa = tipo_casa_num(tipo_casa)
tipo_credito = tipo_credito_num(tipo_credito)
grado_credito = grado_num(grado_credito)
default = verif_default(default)

user_data = [int(edad), ingresos_anuales, tiempo_empleado, prestamo_intencion, tasa_interes, razon_prestamo_ingreso, tiempo_sistema, int(tipo_casa),
        int(tipo_credito), int(grado_credito), int(default)]



if st.button('calcular'):
    st.text(procesar_datos(user_data))
else:
    pass


st.text("*Datos tratados según la política de tratamiento de datos")
st.text("*Sus datos no serán almacenados en ninguna base de datos ni serán compartidos con ninguna entidad externa")


footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #0E1117;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Desarrollado con ❤ por Quick Cash - equipo 3<a style='display: block; text-align: center;' href="https://github.com/juanfloo/proyecto_fund_analitica.git" target="_blank">Repositorio en github</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

# "person_age"
# "person_income"
# "person_emp_length"
# "loan_amnt"
# "loan_int_rate"
# "loan_percent_income"
# "cb_person_cred_hist_length"
# "person_home_ownership_num"
# "loan_intent_num"
# "loan_grade_num"
# "cb_person_default_on_file_num"