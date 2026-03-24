import pandas as pd

def transformar_one_hot(nova_instancia, colunas):
    # preenchendo as colunas
    zeros = []
    for i in range (len(colunas)):
        zeros.append(0)
    
    instancia_df = pd.DataFrame([zeros], columns=colunas)

    for atributo, valor in nova_instancia.items(): # par chave e valor da cor passada 
        coluna = atributo + '_' + valor.lower() # normalizando o nome das colunas

        if coluna in instancia_df.columns: # se encontrar a cor escolhida
            instancia_df[coluna] = 1

    return instancia_df