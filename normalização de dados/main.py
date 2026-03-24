import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from classe_normalizadora import Normalizador


normalizador = Normalizador('./dados_normalizar.csv')
normalizador.mostrar()

# passar a coluna desejada para cada método
normalizador.min_max('idade')
normalizador.mostrar()

normalizador.min_max('altura')
normalizador.mostrar()

normalizador.min_max('peso')
normalizador.mostrar()

# normalizador.label_encoder('sexo')
# normalizador.mostrar()

normalizador.one_hot_encoder('sexo')
normalizador.mostrar()

# ====== revertendo ======
normalizador.reverse_min_max('idade')
normalizador.mostrar()

# normalizador.label_encoder('sexo')
# normalizador.mostrar()

normalizador.reverse_one_hot('sexo')
normalizador.mostrar()

