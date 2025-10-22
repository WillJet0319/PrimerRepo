import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# titulo de la página
st.set_page_config(layout='centered', page_title='Talento Tech Innovador', page_icon=':grinning:')

t1, t2 =st.columns([0.3, 0.7])

t1.image('./index.jpg', width=180)
t2.title('Mi primer tablero')
t2.markdown('**tel:** 123 **| email:** talentotech@gmail.com')

#secciones
steps = st.tabs(['Introduccion', 'Visualización de datos', 'Modelo ML', '$\int_{-\infty}^\infty e^{\sigma\mu}dt$'])
#sección 1
with steps[0]:
    st.title('Metadata')
    st.write('Bienvenido a mi proyecto')
    df=pd.read_csv('penguins.csv')
        

    st.write(df.head())
    #st.dataframe(df.head())
    #st.table(df.head())
print(df.columns)
with steps[1]:
    st.markdown('Gráfica de los tipos de Pinguios')
    species=st.selectbox('Escoja la espacie a visualizar',
                         ['Adelie', 'Gentoo', 'Chinstrap'])
    x=st.selectbox('Selecciona la variable x', list(df.columns))
    y=st.selectbox('Selecciona la variable y', list(df.columns))
    fig, ax= plt.subplots()
    ax= sns.scatterplot(x=df[x], y=df[y], data=df)
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot(fig)
    
import sklearn 
import joblib
rfc = joblib.load('random_forest_penguin.pickle')
unique_penguin_mapping = joblib.load('output_penguin.pickle')
with steps[2]:
    # Para ver si carga el modelo
    st.write(rfc)
    st.write(unique_penguin_mapping)
        # Opciones para el ususario
    island = st.selectbox('Isla', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.write('Los datos ingresados son {}'.format([island, sex, bill_length, 
                                                  bill_depth, flipper_length, body_mass]))
        # Codificación para las islas
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    if island == 'Biscoe':
        island_biscoe = 1
    elif island == 'Dream':
        island_dream = 1
    elif island == 'Torgerson':
        island_torgerson = 1

    sex_female, sex_male = 0, 0
    if sex == 'Female':
        sex_female = 1
    elif sex == 'Male':
        sex_male = 1
        
        # Modelo ML
    new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, 
                                       island_biscoe,island_dream, island_torgerson,
                                         sex_female,sex_male]])
    prediction_species = unique_penguin_mapping[new_prediction[0]]
    st.write('La especie del pingüino es {}'.format(prediction_species))  