# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
from platypus import Hypervolume, experiment, calculate, display, algorithms 
from platypus import *
import quantstats as qs
from Algoritmo_Evolutivo_PIBIC_Final import Multiobjcard
from Sharpe_Optimization import optimize_sharpe
from Omega_Optimization import optimize_omega



diretorio = './'
card = 8

# In[001]
# Definição de Indicadores de Avaliação da Qualidade das Soluções Não-Dominadas
def calc_ind(aux):
    
    list_solutions = []
    
    for i in aux.index:
        solucao = Solution(Multiobjcard())
        solucao.variables = [None] * 16
        solucao.objectives = [None] * 3
        for j in range(len(solucao.variables)):
            solucao.variables[j]=aux['x'+str(j+1)][i]
        for k in range(3):
            solucao.objectives[k]=aux['f'+str(k+1)][i]
        list_solutions.append(solucao)
    
    calc_hyper = Hypervolume(list_solutions)

    return calc_hyper.calculate(list_solutions)

# In[002]
# Coletando a matriz original de preços
mercado = ''
jt01 = ''
jt02 = ''
jt03 = ''
jt04 = ''

files=[['PIBICIBOV19941998.csv'],['PIBICIBOV20042008.csv'],['PIBICIBOV20142018.csv'],['PIBICDWJ19941998.csv'],['PIBICDWJ20042008.csv'],['PIBICDWJ20142018.csv']]

#Agrupando os arquivos .CSV:
for file in files:    
    if file[0] == 'PIBICIBOV19941998.csv':
        mercado = 'IBOV'
        jt01='01/01/1994'
        jt02='31/12/1998'
        jt03='01/01/1999'
        jt04= '31/12/2003'
    if file[0] == 'PIBICIBOV20042008.csv':
        mercado = 'IBOV'
        jt01='01/01/2004'
        jt02='31/12/2008'
        jt03='01/01/2009'
        jt04= '31/12/2013'
    if file[0] == 'PIBICIBOV20142018.csv':
        mercado = 'IBOV'
        jt01='01/01/2014'
        jt02='31/12/2018'
        jt03='01/01/2019'
        jt04= '31/12/2022'
    if file[0] == 'PIBICDWJ19941998.csv':
        mercado = 'DWJ'
        jt01='01/01/1994'
        jt02='31/12/1998'
        jt03='01/01/1999'
        jt04= '31/12/2003'
    if file[0] == 'PIBICDWJ20042008.csv':
        mercado = 'DWJ'
        jt01='01/01/2004'
        jt02='31/12/2008'
        jt03='01/01/2009'
        jt04= '31/12/2013'
    if file[0] == 'PIBICDWJ20142018.csv':
        mercado = 'DWJ'
        jt01='01/01/2014'
        jt02='31/12/2018'
        jt03='01/01/2019'
        jt04= '31/12/2022'
    
    jt01 = pd.to_datetime(jt01)
    jt02 = pd.to_datetime(jt02)
    jt03 = pd.to_datetime(jt03)  
    jt04 = pd.to_datetime(jt04)  
    A = pd.read_csv(file[0])
    A = A.fillna(0)

# In[004]
# Calculando o Hypervolume de cada algoritmo:
    hypsall = []
    for algorithm in A['Alg'].unique():
        hyps=[]
        for run in A['Run'].unique():
            print(run)
            aux = A[(A['Alg']==algorithm)&(A['Run']==run)]
            if len(aux)==1 or (A[(A['Alg']==algorithm)&(A['Run']==run)]['f1'].max() - A[(A['Alg']==algorithm)&(A['Run']==run)]['f1'].min() == 0) or (A[(A['Alg']==algorithm)&(A['Run']==run)]['f2'].max() - A[(A['Alg']==algorithm)&(A['Run']==run)]['f2'].min() == 0): # or (A[(A['Alg']==algorithm)&(A['Run']==run)]['f3'].max() - A[(A['Alg']==algorithm)&(A['Run']==run)]['f3'].min() == 0):
                del aux
            else:
                h = calc_ind(aux)
                hyps.append(h)
        hypsall.append(hyps)
    hypall = pd.DataFrame(hypsall)
    hypall.index=('NSGAII','NSGAIII','GDE3','IBEA')
    labels = ['NSGAII','NSGAIII','GDE3','IBEA']
    fig = plt.figure()
    red_square = dict(markerfacecolor='g', marker='D')
    plt.boxplot(hypall.T, labels=labels, autorange=True, showmeans=True, meanprops=red_square)
    plt.title('Hypervolume - '+mercado+' - '+str(jt01.year)+'-'+str(jt02.year))
    fig.savefig(diretorio+'hypervol'+mercado+str(jt01.year)+'-'+str(jt02.year)+'k = '+str(card)+'.png')
    
    aux = hypall.T['NSGAII']
    aux2 = hypall.T['NSGAIII']
    aux3 = hypall.T['GDE3']
    aux4 = hypall.T['IBEA']
    aux5 = aux.append(aux2).append(aux3).append(aux4)
    
# In[004]
# Os Hypervolumes apresentaram a mesma variância? 
    
    #Se p-valor < 0.05, as variâncias não são iguais
    # levene = st.levene(aux,aux2,aux3,aux4)    
    # bartlett = st.bartlett(aux,aux2,aux3,aux4)
    # levene01,levene02 = levene[0],levene[1]
    # bartlett01,bartlett02 = bartlett[0],bartlett[1]
    # kruskall01, kruskall02 = st.kruskal(aux,aux2,aux3,aux4)
    
# In[005]
# Os Hypervolumes apresentaram distribuição normal ou o n é suficientemente grande (Se não, usar o Teste de Kruskal-Wallis)? 
            
    #Se p-valor < 0.05, as variâncias não são iguais
    
    # st.shapiro(aux)
    # st.shapiro(aux2)
    # st.shapiro(aux3)
    # st.shapiro(aux4)
    # normalidade = st.shapiro(aux5.values)
    # normal01,normal02 = normalidade[0],normalidade[1]

# In[006]
# Teste ANOVA - Existe diferença entre as médias dos hypervolumes? 
    
    f,p = st.f_oneway(aux,aux2,aux3,aux4)
    print ('O resultado do ANOVA foi:', 'F-value:', f, 'p-value:', p)
    
# #Teste de Nemenyi - Diferença de médias grupo a grupo não-paramétrica

#     resumo = {'levene':[levene01], 'levene-p':levene02, 'bartlett':bartlett01, 'bartlett-p':bartlett02, 'normal':normal01, 'normal-p':normal02, 'ANOVA F':f, 'ANOVA p-value':p, 'Kruskall-Wallis': kruskall01,'Kruskall p-value': kruskall02 }
#     resumo = pd.DataFrame(resumo).T


# In[007]
# Em quais dos algoritmos existe a diferença entre as médias dos hypervolumes? - Teste de Tukey
    
    auxpd = pd.DataFrame(columns=['Algo', 'HV'])
    auxpd['HV']=aux5.values
    auxpd['Algo'][0:100]='NSGAII'
    auxpd['Algo'][101:200]='NSGAIII'
    auxpd['Algo'][201:300]='GDE3'
    auxpd['Algo'][301:400]='IBEA'
     
    # mc = MultiComparison(data=auxpd['HV'].values, groups=auxpd['Algo'])
    # result = mc.tukeyhsd()
    # d = pd.DataFrame()
    # d['Algo']=[['GDE3 x IBEA'],['GDE3 x NSGA2'],['GDE3 x NSGA3'],['IBEA x NSGA2'],['IBEA x NSGA3'],['NSGA2 x NSGA3']]
    # d['Média das diferenças']=result.meandiffs
    # d['Limite inferior']=result.confint[:,0]
    # d['Limite superior']=result.confint[:,1]
    # d['Rejeita?']=result.reject
    
    # e = pd.DataFrame()
    # e = sci.posthoc_nemenyi(auxpd, val_col='HV', group_col='Algo')
    
    # fig = result.plot_simultaneous(xlabel='Hypervolume Médio', ylabel='Algoritmos Evolutivos')
    # fig.savefig(diretorio+'tukeyfigExp06'+mercado+str(jt01.year)+'-'+str(jt02.year)+'k = '+str(card)+'.png')
    # stats_pd = pd.ExcelWriter(diretorio+'StatsTestsExp06'+mercado+str(jt01.year)+'-'+str(jt02.year)+'k = '+str(card)+'.xlsx')
    # d.to_excel(stats_pd, sheet_name='statsTukey'+mercado+str(jt01.year)+'-'+str(jt02.year))
    # e.to_excel(stats_pd, sheet_name='statsNemenyi'+mercado+str(jt01.year)+'-'+str(jt02.year))
    # resumo.to_excel(stats_pd, sheet_name='resumo'+mercado+str(jt01.year)+'-'+str(jt02.year))
    # stats_pd.save()    
 
# In[008]
#Coletando a melhor execução dos algoritmos evolutivos
    #Escolhendo o melhor algoritmo:
    algo = auxpd[auxpd['HV'].argmax():auxpd['HV'].argmax()+1]['Algo'][auxpd['HV'].argmax()]
    index_max=hypall.T[algo].argmax(0)
    R = A[(A['Alg']==algo)&(A['Run']==index_max)]
    R.index=range(0,len(R))

    #Leitura dos preços por ativo, por bolsa e por data
    B = pd.read_excel(str(diretorio)+'dados_full.xlsx', sheet_name=mercado, na_values=['nan', '-' , ' '])
    #Transformando a data em formato %d/%m/%y e colocando como índice
    pd.to_datetime(B['Data']).apply(lambda x:x.strftime('%d/%m/%Y'))
    B.index = B['Data']
    #Colunas para deletar
    if mercado=='DWJ':
        cols_to_drop = ['Data','TLR','VIX+','VIX-']
    if mercado=='IBOV':
        cols_to_drop = ['Data','TLR', 'IBOV','VIX+','VIX-']
    C = B.drop(columns=cols_to_drop, inplace=False)

    #Recortando as janelas in-sample
    mask =(C.index >= jt01) & (C.index <= jt02)
    C = (C.loc[mask])

    #Apagando as empresas sem nenhuma informação
    C = C.dropna(axis = 'columns', how = 'all')
    label_col=C.columns

    #Condição de ter sido negociada em 80% dos pregões da janela temporal
    for i in label_col:
        if C[i].count()<(0.8*len(C)):
            C = C.drop(columns=i)
    label_col=C.columns
    n_row, n_col = C.shape
    matrix_array = np.asarray(C)
    C = C.fillna(0.00)

    #Lendo cada carteira da melhor execução
    cart_nd=np.zeros((len(R),C.shape[1]))
    for k in range(1,card+1):
        for j in range(len(cart_nd)):
            cart_nd[j][R['x'+str(k)][j]]+=R['x'+str(k+card)][j]
    cart_nd=pd.DataFrame(cart_nd)
    cart_nd.columns = label_col
    
    #Lendo a matriz out of sample
    D = B.drop(columns=cols_to_drop, inplace=False)
    #Recortando as janelas in-sample
    mask =(D.index >= jt03) & (D.index <= jt04)
    D = (D.loc[mask])
    #Apagando as empresas sem nenhuma informação
    D = D.dropna(axis = 'columns', how = 'all')
    label_col=D.columns
    #Condição de ter sido negociada em 80% dos pregões da janela temporal
    for i in label_col:
        if D[i].count()<(0.8*len(D)):
            D = D.drop(columns=i)
    label_col=D.columns
    n_row, n_col = D.shape
    matrix_array = np.asarray(D)
    D = D.fillna(0.00)
    
    #Lendo a carteira ótima out-of-sample do Algoritmo Evolutivo
    cart_otima = pd.DataFrame(columns=D.columns, index=range(0,len(cart_nd)))
    for i in range(0, len(cart_otima)):
        for col_name in cart_nd.columns:
    # Atribua o valor de cart_nd para cart_otima
            cart_otima.at[i, col_name] = cart_nd.at[i, col_name]
    
    cart_otima=cart_otima.fillna(0.00)

# In[009]
    #Lendo as principais estatísticas das demais estratégias
    
    #Indice:
    if mercado == 'DWJ':
        stock = qs.utils.download_returns('DJI')
        mask =(stock.index >= jt03) & (stock.index <= jt04)
        E = (stock.loc[mask])
    
    if mercado == 'IBOV':
        mask =(B.index >= jt03) & (B.index <= jt04)
        E = (B.loc[mask])
        E = E['IBOV']
    
    ret_acum_index = E.sum()  
    print(ret_acum_index)
    cv_index = (E.std()/E.mean())
    print(cv_index)
    dfi_index = pd.DataFrame(E)
    cvar_index = qs.stats.cvar(dfi_index)
    print(cvar_index)
    ddmax_index = qs.stats.max_drawdown(dfi_index)
    print(ddmax_index)
    
    
    #Carteira Sharpe:
    #Lendo a carteira ótima out-of-sample
    peso_gabarito = pd.DataFrame(0.00, index=D.columns, columns=range(1))
    peso_sharpe = optimize_sharpe(mercado,jt01,jt02)
    peso_sharpe=peso_sharpe.fillna(0.00)

    for i in range(len(peso_sharpe)):
        for k in range(len(peso_gabarito)):
            if peso_sharpe.index[i] == peso_gabarito.index[k]:
                peso_gabarito.loc[peso_gabarito.index[k]] = peso_sharpe.iloc[i, 0]

    cart_sharpe = np.dot(D, peso_gabarito)
    ret_acum_sharpe = cart_sharpe.sum()  
    print(ret_acum_sharpe)
    cv_sharpe = (cart_sharpe.std()/cart_sharpe.mean())
    print(cv_sharpe)
    dfi_sharpe = pd.DataFrame(cart_sharpe)
    cvar_sharpe = qs.stats.cvar(dfi_sharpe)
    print(cvar_sharpe)
    ddmax_sharpe = qs.stats.max_drawdown(dfi_sharpe)
    print(ddmax_sharpe)        
    
    #Carteira Ômega:
    #Lendo a carteira ótima out-of-sample
    peso_gabarito = pd.DataFrame(0.00, index=D.columns, columns=range(1))
    peso_omega = optimize_omega(mercado,jt01,jt02)
    peso_omega=peso_omega.fillna(0.00)
    
    for i in range(len(peso_omega)):
        for k in range(len(peso_gabarito)):
            if peso_omega.index[i] == peso_gabarito.index[k]:
                peso_gabarito.loc[peso_gabarito.index[k]] = peso_omega.iloc[i, 0]
    
    cart_omega = np.dot(D, peso_gabarito)
    
    ret_acum_omega = cart_omega.sum()  
    print(ret_acum_omega)
    cv_omega = (cart_omega.std()/cart_omega.mean())
    print(cv_omega)
    dfi_omega = pd.DataFrame(cart_omega)
    cvar_omega = qs.stats.cvar(dfi_omega)
    print(cvar_omega)
    ddmax_omega = qs.stats.max_drawdown(dfi_omega)
    print(ddmax_omega)      
    
    
    #Carteira Algoritmo Evolutivo
    resultado = {}

    for i in range(0, len(cart_otima)):
        peso = cart_otima[i:i+1].values
        resultado[i] = np.dot(D, peso.T)

    # Criando um Dataframe para armazenar o resultado
    df2 = pd.DataFrame()
    for key, value in resultado.items():
        df2[key] = value.flatten()

    
    #Retorno Acumulado
    alg = df2.sum()
    sharpe = ret_acum_sharpe
    omega = ret_acum_omega
    index = ret_acum_index 
    box = [alg, sharpe, omega, index]
    labels = ['Algo Evol','Sharpe','Omega','Index']
    fig = plt.figure()
    red_square = dict(markerfacecolor='g', marker='D')
    plt.boxplot(box, labels=labels, autorange=True, showmeans=True, meanprops=red_square)
    plt.title('Retorno Acumulado - '+mercado+' - '+str(jt03.year)+' a '+str(jt04.year))
    fig.savefig(diretorio+'Retorno'+mercado+str(jt03.year)+'-'+str(jt04.year)+'.png')
    
    
    #Coeficiente de Variação
    std = df2.std()
    mean = df2.mean()
    alg = std/mean
    sharpe = cv_sharpe 
    omega = cv_omega
    index = cv_index 
    box = [alg, sharpe, omega, index]
    labels = ['Algo Evol','Sharpe','Omega','Index']
    fig = plt.figure()
    red_square = dict(markerfacecolor='g', marker='D')
    plt.boxplot(box, labels=labels, autorange=True, showmeans=True, showfliers=False, meanprops=red_square)
    plt.title('Coeficiente de Variação - '+mercado+' - '+str(jt03.year)+' a '+str(jt04.year))
    fig.savefig(diretorio+'CoefVariacao'+mercado+str(jt03.year)+'-'+str(jt04.year)+'.png')
    
    
    #CVAR (95%)
    qs.extend_pandas()
    alg = qs.stats.cvar(df2)
    sharpe = cvar_sharpe 
    omega = cvar_omega 
    index = cvar_index
    box = [alg, sharpe, omega, index]
    labels = ['Algo Evol','Sharpe','Omega','Index']
    fig = plt.figure()
    red_square = dict(markerfacecolor='g', marker='D')
    plt.boxplot(box, labels=labels, autorange=True, showmeans=True, meanprops=red_square)
    plt.title('CVAR(95%) - '+mercado+' - '+str(jt03.year)+' a '+str(jt04.year))
    fig.savefig(diretorio+'CVAR'+mercado+str(jt03.year)+'-'+str(jt04.year)+'.png')
    
    
    #Drawdown Máximo 
    alg = -qs.stats.max_drawdown(df2)
    sharpe = ddmax_sharpe 
    omega = ddmax_omega
    index = ddmax_index 
    box = [alg, sharpe, omega, index]
    labels = ['Algo Evol','Sharpe','Omega','Index']
    fig = plt.figure()
    red_square = dict(markerfacecolor='g', marker='D')
    plt.boxplot(box, labels=labels, autorange=True, showmeans=True, meanprops=red_square)
    plt.title('Drawdown Máximo '+mercado+' - '+str(jt03.year)+' a '+str(jt04.year))
    fig.savefig(diretorio+'Drawdown'+mercado+str(jt03.year)+'-'+str(jt04.year)+'.png')
    
    
    #Composição das empresas escolhidas
    plt.figure(figsize=(10, 6))  # Define o tamanho da figura (opcional)
    plt.bar(cart_otima.mean().index[cart_otima.mean()>0], cart_otima.mean()[cart_otima.mean()>0])
    plt.ylabel('% na Carteira')    # Define o rótulo do eixo y (opcional)
    plt.title('Porcentagem Média por Empresa na Carteira '+mercado+' - '+str(jt03.year)+' a '+str(jt04.year))  # Define o título do gráfico (opcional)
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig(diretorio+'Choice'+mercado+str(jt03.year)+'-'+str(jt04.year)+'.png')   
    
    
