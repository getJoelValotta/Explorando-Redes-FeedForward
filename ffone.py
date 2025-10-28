import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout, Normalization
from keras.models import Sequential, Model

datos_medicos = pd.read_csv('medical_insurance.csv')

#__________________________________Preprocesamiento de Features__________________________________#

# Devuelve los identificadores de features
datos_medicos.head()

# Primera parte One-Hot (Separa categorias de datos numericos, es decir, le coloca el tipo abstracto categoria para diferenciarlos)
categorias_col = ['sex', 'region', 'urban_rural', 'education', 'marital_status', 'employment_status', 'smoker', 'alcohol_freq', 'plan_type', 'network_tier']
for col in categorias_col:
    if col in datos_medicos.columns:
        datos_medicos[col] = datos_medicos[col].astype('category')

target = 'annual_medical_cost'

# Excluye a los targets e identificadores de los features
datos_basura = ['person_id', target]
features = datos_medicos.drop(columns=datos_basura, errors='ignore')

# Identifica los valores categoricos de los valores numericos (Es redundante incluir 'object', debido a que ya se discernio previamente)
features_categoria = features.select_dtypes(include=['category', 'object']).columns.tolist()
features_numericas = features.select_dtypes(include=[np.number]).columns.tolist()

# Segunda parte One-Hot (Codifica las variables categoricas en variables binarias)
features_codificados = pd.get_dummies(features, columns=features_categoria, drop_first=True)

# Define X:entrenamiento/test e Y:target para el modelo
x = features_codificados.to_numpy().astype('float32')
y = datos_medicos[target].to_numpy().astype('float32')

#__________________________________Definicion de los Batches y Capas del modelo__________________________________#

n_samples = x.shape[0]         # cantidad de datos (int)
dim_entrada = x.shape[1]       # número de features (int)

size_entrenamiento = int(n_samples * 0.8)
size_test = n_samples - size_entrenamiento
size_batch = 32

# Mezcla y crea los datasets de entrenamiento y test (Cada elemento es un par (feature, target))
datos_medicos = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(n_samples, seed=777)
# Los datos de entrenamiento y test son objetos batch de TensorFlow
datos_entrenamiento = datos_medicos.take(size_entrenamiento).batch(size_batch)
datos_test = datos_medicos.skip(size_entrenamiento).batch(size_batch)

# Capa de normalización de TensorFlow: Los datos de entrada se normalizan para que tengan media 0 y desviación estándar 1, mejorando el tiempo de entrenamiento
normalizador = Normalization()
# Se adapta el normalizador a los datos de entrenamiento y no con todos los datos para que al momento de hacer test no pierda su capacidad de generalizar.
# Solo pide features, datos entrenamientos es una tupla (features, target), por lo que se usa map para extraer solo las features
normalizador.adapt(datos_entrenamiento.map(lambda x, y: x))

capa_entrada = Input(shape=(dim_entrada,), name='Input_Layer')
capa_oculta_1 = Dense(26, activation='relu')(capa_entrada)
droput_1 = Dropout(0.2)(capa_oculta_1)
capa_oculta_2 = Dense(12, activation='relu')(capa_oculta_1)
droput_2 = Dropout(0.2)(capa_oculta_2)
capa_salida = Dense(1)(capa_oculta_2)

#__________________________________Definicion y configuracion del Modelo__________________________________#

""" # Se puede usar Sequential si no se conecta las capas con como esta ejemplificado arriba
modelo_feedforward = Sequential([
    capa_entrada,
    normalizador,
    capa_oculta_1,
    Dropout(0.2),
    capa_oculta_2,
    Dropout(0.2),
    capa_salida
])
"""
modelo_feedforward = Model(inputs=capa_entrada, outputs=capa_salida)

modelo_feedforward.compile(
    optimizer=keras.optimizers.Adam(),
    loss='mean_squared_error',
    metrics=['mean_absolute_error', 'mean_squared_error']
)

historial = modelo_feedforward.fit(
    datos_entrenamiento,
    epochs=100,
    validation_data=datos_test
)

modelo_feedforward.save('modelo_medico_feedforward.keras')

#__________________________________Graficos (Conclusiones)__________________________________#

fig, axs = plt.subplots(1, 2, figsize=(10, 8))

mse_entrenamiento = historial.history['mean_squared_error']
mse_validacion = historial.history['val_mean_squared_error']
epochs = range(1, len(mse_entrenamiento) + 1)

axs[0].plot(epochs, mse_entrenamiento, 'b-', label='MSE Entrenamiento')
axs[0].plot(epochs, mse_validacion, 'r--', label='MSE Validación')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Mean Squared Error')
axs[0].set_title('MSE Entrenamiento vs Validación')
axs[0].legend()
axs[0].grid()

x_test = x[int(n_samples*0.8):]
y_test = y[int(n_samples*0.8):]
predicciones = modelo_feedforward.predict(x_test)
axs[1].scatter(y_test, predicciones, alpha=0.6, color='teal')
axs[1].set_xlabel('Costo medico anual real')
axs[1].set_ylabel('Costo medico anual predicho')
axs[1].set_title('Costo Medico Anual Actual vs Predicho')
min_val = min(y_test.min(), predicciones.min())
max_val = max(y_test.max(), predicciones.max())
axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=2, label ='Recta de Aproximacion')
axs[1].set_xlim(0,65000)
axs[1].legend()
axs[1].grid()

plt.savefig('mse_entrenamiento_validacion_prediccion.png')
plt.tight_layout()
plt.show()