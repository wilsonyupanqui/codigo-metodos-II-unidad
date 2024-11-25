import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class GradientDescent:
    def __init__(self, learning_rate=0.01, momentum=0.9, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def optimize(self, func, gradient_func, initial_point):
        current_point = np.array(initial_point, dtype=float)
        velocity = np.zeros_like(current_point)
        
        history_values = [func(current_point)]
        history_points = [current_point.copy()]
        
        for i in range(self.max_iterations):
            gradient = gradient_func(current_point)
            velocity = self.momentum * velocity - self.learning_rate * gradient
            new_point = current_point + velocity
            
            history_points.append(new_point.copy())
            history_values.append(func(new_point))
            
            if np.linalg.norm(new_point - current_point) < self.tolerance:
                break
                
            current_point = new_point
            
        return current_point, history_values, history_points

# Funciones de ejemplo
def example_function(x):
    """Función cuadrática: f(x, y) = x^2 + 2y^2"""
    return x[0]**2 + 2 * x[1]**2

def example_gradient(x):
    """Gradiente de la función cuadrática"""
    return np.array([2 * x[0], 4 * x[1]])

def rastrigin_function(x):
    """Función Rastrigin: una función de prueba no convexa"""
    A = 10
    return A * len(x) + sum((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x)

def rastrigin_gradient(x):
    """Gradiente de la función Rastrigin"""
    A = 10
    return np.array([2 * xi + 2 * np.pi * A * np.sin(2 * np.pi * xi) for xi in x])

def create_contour_data(x_range, y_range, func):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    return X, Y, Z

# Configuración de la aplicación Streamlit
st.title("Visualización de Descenso de Gradiente")
st.write("""
Esta aplicación demuestra el algoritmo de descenso de gradiente con momento en diferentes funciones de prueba.
Ajusta los parámetros y observa cómo afectan a la convergencia del algoritmo.
""")

# Sidebar para parámetros
st.sidebar.header("Parámetros del Algoritmo")
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 1.0, 0.1, 0.01)
momentum = st.sidebar.slider("Momento", 0.0, 0.99, 0.9, 0.01)
max_iterations = st.sidebar.slider("Máximo de Iteraciones", 100, 2000, 1000, 100)
initial_x = st.sidebar.slider("Punto Inicial X", -5.0, 5.0, 2.0, 0.1)
initial_y = st.sidebar.slider("Punto Inicial Y", -5.0, 5.0, 1.0, 0.1)

# Sidebar para seleccionar la función
function_choice = st.sidebar.selectbox(
    "Elige la Función para Optimizar",
    ["Función Cuadrática", "Función Rastrigin"]
)

# Seleccionar función y gradiente
if function_choice == "Función Cuadrática":
    func = example_function
    grad_func = example_gradient
else:
    func = rastrigin_function
    grad_func = rastrigin_gradient

# Crear el optimizador con los parámetros seleccionados
optimizer = GradientDescent(
    learning_rate=learning_rate,
    momentum=momentum,
    max_iterations=max_iterations
)

# Optimizar
initial_point = np.array([initial_x, initial_y])
optimal_point, values_history, points_history = optimizer.optimize(
    func,
    grad_func,
    initial_point
)

# Crear las visualizaciones
points_history = np.array(points_history)

# Crear dos columnas para las gráficas
col1, col2 = st.columns(2)

with col1:
    st.subheader("Convergencia de la Función Objetivo")
    fig1, ax1 = plt.subplots()
    ax1.plot(values_history)
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Valor de la Función')
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    st.subheader("Trayectoria de Optimización")
    fig2, ax2 = plt.subplots()
    
    # Crear datos para el contour plot
    X, Y, Z = create_contour_data((-5, 5), (-5, 5), func)
    ax2.contour(X, Y, Z, levels=20)
    ax2.plot(points_history[:, 0], points_history[:, 1], 'b.-')
    ax2.plot(points_history[0, 0], points_history[0, 1], 'go', label='Inicio')
    ax2.plot(points_history[-1, 0], points_history[-1, 1], 'ro', label='Final')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# Visualización 3D
st.subheader("Visualización 3D de la Función")
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax3.plot(points_history[:, 0], points_history[:, 1], values_history, 'r.-')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('f(x, y)')
st.pyplot(fig3)

# Mostrar resultados
st.subheader("Resultados")
col3, col4 = st.columns(2)
with col3:
    st.write(f"Punto óptimo encontrado: [{optimal_point[0]:.4f}, {optimal_point[1]:.4f}]")
with col4:
    st.write(f"Valor mínimo de la función: {func(optimal_point):.4f}")

# Métricas adicionales
st.subheader("Métricas de Convergencia")
num_iterations = len(values_history) - 1
improvement = values_history[0] - values_history[-1]

col5, col6 = st.columns(2)
with col5:
    st.metric("Número de Iteraciones", num_iterations)
with col6:
    st.metric("Mejora Total", f"{improvement:.4f}")

# Descargar resultados
data = {
    "x": points_history[:, 0],
    "y": points_history[:, 1],
    "f(x, y)": values_history
}
df = pd.DataFrame(data)
st.download_button(
    "Descargar Resultados en CSV",
    data=df.to_csv(index=False),
    file_name="resultados_descenso_gradiente.csv",
    mime="text/csv"
)

# Información adicional
st.write("""
### Explicación de los Parámetros:
- *Tasa de Aprendizaje*: Controla el tamaño de los pasos en cada iteración.
- *Momento*: Determina cuánto influye la dirección anterior en el siguiente paso.
- *Máximo de Iteraciones*: Límite de iteraciones para el algoritmo.
- *Punto Inicial*: Coordenadas (x, y) desde donde comienza la optimización.
""")

# Mensaje de convergencia
if num_iterations < max_iterations:
    st.success("¡Convergencia alcanzada!")
else:
    st.warning("Se alcanzó el número máximo de iteraciones. Puede que el algoritmo no haya convergido.")
