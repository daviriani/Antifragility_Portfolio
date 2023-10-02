# -*- coding: utf-8 -*-

import pandas as pd
import cvxpy as cp
import numpy as np


def optimize_sharpe(mercado, jt01, jt02):
    
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

    # 3. Estimar os parâmetros do modelo de otimização utilizando a janela in sample
    mu = in_sample.mean().values
    mu[np.isnan(mu)] = 0
    
    # 4. Matriz de Covariância da amostra
    Sigma = in_sample.cov().values
    Sigma[np.isnan(Sigma)]= 0
    
    # 5. Número de ativos da carteira
    n_assets = len(mu)
    
    # 6. Criação da variável de peso dos ativos (w)
    w = cp.Variable(n_assets)
    
    #7. Definição da função de retorno esperado da carteira e matriz de covariância
    ret = mu
    risk = cp.quad_form(w, Sigma)
    
    #8. Defina o parâmetro de regularização
    lambda_ = cp.Parameter(nonneg=True)
    
    #9. Defina a função de utilidade quadrática
    utility = w.T @ ret - (lambda_ / 2)*risk
    objective = cp.Maximize(utility)
    constraints = [
        w >= 0,
        cp.sum(w) == 1
    ]
    problem = cp.Problem(objective, constraints)
    
    #10. Colocando os lambdas para percorrer o domínio
    lambdas = [0.01, 0.1, 1, 10, 100]
    
    for lam in lambdas:
        lambda_.value = lam
        problem.solve()
        print(f"Para lambda = {lam}, status = {problem.status}, retorno esperado = {ret.T @ w.value}, volatilidade = {np.sqrt(w.value.T @ Sigma @ w.value)}")
    
    #11. Pegar a carteira ótima in sample
    w.value
    
    #12. Calcular o retorno in sample (otimizado dentro da amostra)
    # 12.1. Tirando os valores de NaN:
    in_sample = in_sample.fillna(0.00)
    
    # 12.2. Tirando a coluna Data dos valores In Sample:
    in_sample2 = in_sample.drop(columns='Data')
    
    # Encontrando o valor ótimo da carteira (In Sample):
    retorno_in = in_sample2 @ w.value
    print(retorno_in.sum())

    #13. Exportar o vetor de pesos, o resultado in sample e o resultado out of sample
    peso = w.value
    peso_df = pd.DataFrame(peso)
    peso_df.index=in_sample2.columns
    peso_df.to_excel('Pesos_Sharpe -'+mercado+' - '+str(jt01.year)+'-'+str(jt02.year)+'.xlsx')
    
    return peso_df
    


# out_of_sample = df.loc[(df['Data'] >= '2019-01-01') & (df['Data'] < '2023-01-01')]
# out_of_sample = out_of_sample.drop(columns='TLR')



# # 13. Aplicar os parâmetros estimados na janela out of sample
# # 13.1. Tirando os valores de NaN:
# out_of_sample = out_of_sample.fillna(0.00)

# # 13.2. Tirando a coluna Data dos valores Out of Sample:
# out_of_sample2 = out_of_sample.drop(columns='Data')

# #14. Calculando o retorno out of sample:
# retorno = out_of_sample2 @ w.value
# print (retorno.sum()) 





