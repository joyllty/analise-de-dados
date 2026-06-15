# Classificação da Qualidade de Vinhos - Wine Quality

## 1. Pipeline do projeto

Fluxo até o treinamento do modelo:

1. Carregamento dos arquivos winequality-red.csv e winequality-white.csv;
2. Adição da coluna tipo_vinho, onde:
   - 0 representa vinho tinto;
   - 1 representa vinho branco;
3. Junção dos dois datasets em um único dataframe;
4. Salvar dataset unificado;
5. Separação dos atributos de entrada e da classe quality;
6. Embaralhamento dos dados antes da cross validation;
7. Hiperparametrização dos modelos com RandomizedSearchCV;
8. Avaliação dos modelos com cross_validate;
9. Geração da matriz de confusão;
10. Cálculo da acurácia por classe;
11. Comparação dos modelos;
12. Salvamento do melhor modelo em arquivo .pkl;
13. Demonstração do modelo no arquivo de inferência.

## 3. Tratamento do desbalanceamento
- A base é desbalanceada. As classes 5 e 6 aparecem com muito mais frequência, enquanto classes como 3, 4, 8 e 9 possuem bem poucos exemplos. Por isso, utilizei class_weight como opção na hiperparametrização com valor 'balanced' -> faz com que o modelo aumente o peso das classes menos frequentes e reduza o peso das classes mais frequentes.
- No grid de parâmetros, o class_weight foi deixado como opcional, deixando para o RandomizedSearchCV testar se vale a pena usar ou não.
- Eu cheguei a testar uma versão com SMOTE(treinamento_smote.py), mas nas classes com poucos exemplos as amostras sintéticas foram pouco representativas e reduziram a acurácia global e o F1 weighted. Apesar de melhorar algumas classes minoritárias, o SMOTE piorou o desempenho geral dos modelos

## 4. Métricas avaliadas
Usei as seguintes métricas:
'accuracy',
'precision_weighted',
'recall_weighted',
'f1_weighted',
'f1_macro',
Matriz de confusão,
Acurácia por classe

A acurácia por classe foi calculada a partir da matriz de confusão:
- acurácia da classe = acertos da classe / total real da classe

## 5. Resultado dos Modelos
Resultados obtidos sem SMOTE, usando class_weight na hiperparametrização:
                      accuracy  precision_weighted  recall_weighted  f1_weighted  f1_macro
modelo                                                                                    
Random Forest           0.6896              0.6975           0.6896       0.6776    0.4042
Extra Trees             0.6906              0.6972           0.6906       0.6792    0.4168
HistGradientBoosting    0.6748              0.6689           0.6748       0.6673    0.4052

## 6. Justificativa do modelo escolhido e Matriz de confusão 
- O modelo escolhido para implantação foi o Extra Trees, pois ele apresentou a maior acurácia entre os modelos, além do melhor F1-score weighted e do melhor F1-score macro. Porém, apesar de ter o melhor desempenho, ele e os outros modelos tiveram muita dificuldade nas classes extremas da variável quality, principalmente nas classes 3, 4, 8 e 9. As classes 5 e 6 tem a maior parte dos exemplos, e as classes extremas possuem poucos registros. No caso da 9, a situação é pior ainda com 5 exemplos, o modelo praticamente não consegue aprender o comportamento da classe. Por isso a classe 9 continuou sem acertos.
- Enfim, a diferença em relação à Random Forest foi pequena, mas o Extra Trees apresentou melhor equilíbrio geral nas métricas.
- O F1 macro é especialmente relevante nesse contexto, porque calcula a média entre as classes sem favorecer tanto as classes majoritárias, e como o Extra Trees teve o melhor F1 macro, ele demonstrou melhor equilíbrio entre as classes.

====== MATRIZ DE CONFUSÃO ======
        Previsto 3  Previsto 4  Previsto 5  Previsto 6  Previsto 7  Previsto 8  Previsto 9
Real 3           1           1          17          11           0           0           0
Real 4           2          40         112          60           2           0           0
Real 5           3          11        1556         554          14           0           0
Real 6           0           4         431        2258         140           3           0
Real 7           0           0          23         490         558           8           0
Real 8           0           0           1          75          43          74           0
Real 9           0           0           0           3           2           0           0

====== MÉTRICAS POR CLASSE ======
classe quality 3 -> suporte: 30, acertos: 1, acurácia/recall: 0.0333, precision: 0.1667, f1: 0.0556
classe quality 4 -> suporte: 216, acertos: 40, acurácia/recall: 0.1852, precision: 0.7143, f1: 0.2941
classe quality 5 -> suporte: 2138, acertos: 1556, acurácia/recall: 0.7278, precision: 0.7271, f1: 0.7274
classe quality 6 -> suporte: 2836, acertos: 2258, acurácia/recall: 0.7962, precision: 0.6543, f1: 0.7183
classe quality 7 -> suporte: 1079, acertos: 558, acurácia/recall: 0.5171, precision: 0.7352, f1: 0.6072
classe quality 8 -> suporte: 193, acertos: 74, acurácia/recall: 0.3834, precision: 0.8706, f1: 0.5324
classe quality 9 -> suporte: 5, acertos: 0, acurácia/recall: 0.0000, precision: 0.0000, f1: 0.0000


