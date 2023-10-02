
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_omega(mercado, jt01, jt02):
    
    # 1. Importar os dados históricos dos ativos
    df = pd.read_excel('dados_full.xlsx', sheet_name=mercado)
    df.to_csv('dados_full.csv', index=False)
    df = pd.read_csv('dados_full.csv')
    df['Data'] = pd.to_datetime(df['Data'])
    
    # 2. Dividir os dados em duas janelas temporais
    in_sample = df.loc[(df['Data'] >= jt01) & (df['Data'] < jt02)]
    if mercado == 'DWJ':
        in_sample = in_sample.drop(columns=['TLR','VIX+','VIX-'])
    if mercado == 'IBOV':
        in_sample = in_sample.drop(columns=['IBOV','TLR','VIX+','VIX-'])

    #Condição de ter sido negociada em 80% dos pregões da janela temporal
    for i in in_sample.columns:
        if in_sample[i].count()<(0.8*len(in_sample)):
            in_sample = in_sample.drop(columns=i)
    in_sample = in_sample.fillna(0)

    # 3. Tirando a coluna Data dos valores In Sample:
    in_sample2 = in_sample.drop(columns='Data')
    
    # 4. Função objetivo: Índice Ômega
    def omega_ratio(weights):
        weighted_returns = np.dot(in_sample2, weights)
        positive_returns = weighted_returns[weighted_returns >= 0]
        negative_returns = weighted_returns[weighted_returns < 0]
        omega = -np.sum(positive_returns) / np.sum(negative_returns)
        return omega
    
    # 5. Restrição: soma dos pesos igual a 1
    def constraint(weights):
        return np.sum(weights) - 1
    
    # Vetor de pesos inicial
    initial_weights = np.ones(len(in_sample2.columns)) / len(in_sample2.columns)
    
    # Definir limites dos pesos (0 a 1)
    bounds = [(0, 1)] * (len(in_sample2.columns))
    
    # Definir objeto de restrição
    constraints = {'type': 'eq', 'fun': constraint}
    
    # Otimização utilizando a função minimize do SciPy
    result = minimize(omega_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Resultados da otimização
    optimized_weights = result.x
    #optimized_omega = result.fun

    #6. Exportar o vetor de pesos, o resultado in sample e o resultado out of sample
    peso_df = pd.DataFrame(optimized_weights)
    peso_df.index=in_sample2.columns
    peso_df.to_excel('Pesos_Omega -'+mercado+' - '+str(jt01.year)+'-'+str(jt02.year)+'.xlsx')

    return peso_df





# # Calcular o retorno fora da amostra da carteira otimizada pelo Índice Ômega
# out_of_sample_returns = np.dot(out_of_sample, optimized_weights)
# out_of_sample_return = np.sum(out_of_sample_returns)

# print("Pesos otimizados:", optimized_weights)
# print("Ômega Ratio otimizado:", optimized_omega)
# print("Retorno fora da amostra:", out_of_sample_return)

# out_of_sample = df.loc[(df['Data'] >= '2019-01-01') & (df['Data'] < '2023-01-01')]
# out_of_sample = out_of_sample.drop(columns='TLR')
