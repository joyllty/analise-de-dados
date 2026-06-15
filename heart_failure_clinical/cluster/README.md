
## 1. Justificativa da escolha do metaestimador

O KMeans foi escolhido porque:

- Agrupa pacientes de acordo com a proximidade entre seus atributos;
- Permite representar cada paciente como um ponto em um espaço multidimensional;
- Usa centroides que facilitam a interpretação do perfil médio de cada grupo;
- Permite fazer inferência em novos pacientes usando predict();

## 2. Procedimentos de pré-processamento
### 2.1 Separação da classe alvo

As colunas removidas do treinamento foram: DEATH_EVENT e time
A coluna DEATH_EVENT foi removida porque representa o desfecho do acompanhamento, ou seja, se ocorreu morte ou não do paciente. Como o objetivo é receber um paciente desconhecido, essa informação não deve fazer parte dos dados de entrada.
A coluna time também foi removida porque representa o tempo de acompanhamento do paciente. Ela não é uma característica clínica inicial, então foi usada só na interpretação dos clusters.

### 2.2 Tratamento das variáveis binárias

A base possui variáveis binárias, como:

```text
anaemia
diabetes
high_blood_pressure
sex
smoking
```

Essas colunas foram mantidas no treinamento, porque o KMeans consegue usar elas como atributos numéricos binários. Como o normalizador usado foi o MinMaxScaler, valores que já em 0 e 1 permanecem na mesma escala.

a interpretação das variáveis binárias foi feita de forma diferente das variáveis contínuas:

- valores próximos de 1 indicam maior presença da característica no grupo;
- valores próximos de 0 indicam menor presença da característica no grupo;
- valores intermediários indicam presença moderada ou distribuição equilibrada

### 2.3 Normalização dos dados

Os dados foram normalizados usando MinMaxScale. A normalização foi necessária porque o KMeans utiliza distância euclidiana para calcular a proximidade entre os pacientes. 

## 3. Escolha do número de clusters

Para escolher a quantidade de clusters, utilizei o **método do cotovelo**
```text
K = range(1, 19)
```
Escolhi um range de K de 18, pois resultou em 6 clusters, mantendo grupos com quantidade boa de pacientes e nenhum com numero 
de pacientes abaixo de 30 (10% do dataset). Eu testei com numeros maiores, ou até mesmo com o número maximo de linhas do dataset como range, porém resultava em muitos clusters com poucos pacientes, distorcendo muito a proporção de mortes por cluster.

## 4. Demonstração da inferência funcionando

A inferência é feita com um paciente novo no final da main.py.

```python
novo_dado = pd.DataFrame([[
    65,
    1,
    160,
    0,
    35,
    1,
    250000,
    1.4,
    136,
    1,
    0
]], columns=[
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking'
])

novo_dado_norm = normalizar_novo_dado(novo_dado, normalizador)

cluster_novo_dado = prever_cluster(modelo_cluster, novo_dado_norm)
```

Na saída teremos algo como:
======================================
ANÁLISE DE DESFECHO POR CLUSTER
         total_pacientes  total_mortes  proporcao_mortes  mediana_time_dias
cluster                                                                    
0                     73            20            0.2740              123.0
1                     57            18            0.3158              172.0
2                     48            19            0.3958               86.5
3                     33             9            0.2727              115.0
4                     44            15            0.3409              116.5
5                     44            15            0.3409              132.0
======================================
INFERÊNCIA

>> Dados do novo paciente:
   age  anaemia  creatinine_phosphokinase  diabetes  ejection_fraction  high_blood_pressure  platelets  serum_creatinine  serum_sodium  sex  smoking
0   65        1                       160         0                 35                    1     250000               1.4           136    1        0

>> Novo paciente pertence ao Cluster: 2