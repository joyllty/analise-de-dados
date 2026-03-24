from classe_normalizadora import Normalizador

normalizador = Normalizador('./dados_normalizar.csv')

print('======== dados originais ========')
normalizador.mostrar()

print('======== MinMax Enconder nas colunas numéricas ========')
normalizador.min_max('idade')
normalizador.min_max('altura')
normalizador.min_max('peso')
normalizador.mostrar()

print('======== Label Encoder na coluna categórica ========')
normalizador.label_encoder('sexo')
normalizador.mostrar()

print('======== One Hot Encoder na coluna categórica ========')
normalizador.reverse_label('sexo')
normalizador.one_hot_encoder('sexo')
normalizador.mostrar()

print('======== revertendo todas as normalizações ========')
normalizador.reverse_one_hot('sexo')
normalizador.reverse_min_max('idade')
normalizador.reverse_min_max('altura')
normalizador.reverse_min_max('peso')
normalizador.mostrar()