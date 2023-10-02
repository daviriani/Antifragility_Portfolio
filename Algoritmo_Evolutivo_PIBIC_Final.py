from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy import stats as st
from platypus import *
import random

diretorio = './'

# In[001]
class Multiobjcard(Problem):
    def __init__(self):
        super(Multiobjcard, self).__init__(1, 3)
        self.directions=[Problem.MINIMIZE]*self.nobjs
        self.C=None
        self.risk_free=None
        self.nobjs=3
        self.card=None
        self.Vix=None
        self.indexpos=None
        self.indexneg=None
        
    def evaluate(self,solution):
        var=self.function(solution.variables)
        solution.objectives[0]=var[0]
        solution.objectives[1]=var[1]
        solution.objectives[2]=var[2]
       
    def preparardados(self, diretorio, label, jt01, jt02, card):
        
        self.label=label
        self.jt01=jt01
        self.jt02=jt02
        
        #Definição das janelas temporais
        self.jt01 = pd.to_datetime(self.jt01)
        self.jt02 = pd.to_datetime(self.jt02)
    
        #Leitura dos preços por ativo, por bolsa e por data
        A = pd.read_excel(str(diretorio)+'dados_full.xlsx', sheet_name=self.label, na_values=['nan', '-' , ' '])
        
        #Transformando a data em formato %d/%m/%y e colocando como índice
        pd.to_datetime(A['Data']).apply(lambda x:x.strftime('%d/%m/%Y'))
        A.index = A['Data']
    
        #Colunas para deletar
        if label=='DWJ':
            cols_to_drop = ['Data','TLR']
        if label=='IBOV':
            cols_to_drop = ['Data','TLR', 'IBOV']
        self.C = A.drop(columns=cols_to_drop, inplace=False)
        
        #Recortando as janelas in-sample
        mask =(self.C.index >= jt01) & (self.C.index <= jt02)
        self.C = (self.C.loc[mask])
    
        #Apagando as empresas sem nenhuma informação
        self.C = self.C.dropna(axis = 'columns', how = 'all')
        label_col=self.C.columns
    
        #Condição de ter sido negociada em 80% dos pregões da janela temporal
        for i in label_col:
            if self.C[i].count()<(0.8*len(self.C)):
               self.C = self.C.drop(columns=i)
        label_col=self.C.columns
        #label_row=self.C.index
        n_row, n_col = self.C.shape
        
        #Leitura das taxas livres de risco 
        self.risk_free = A['TLR'][:]
        self.risk_free = self.risk_free.loc[mask]
        self.risk_free = np.nan_to_num(self.risk_free)
        self.risk_free = np.asarray(self.risk_free)
        self.card=8
        self.nvars=card*2
        self.types=[Real(0,1)]*self.nvars
        
        #Leitura do VIX
        self.vix = A['VIX'][:]
        self.vix = self.vix.loc[mask]
        self.vix = np.nan_to_num(self.vix)
        
        for i in range(self.card):
            self.types[i]=Real(0,n_col-1)
            
        self.function=self.funcao_tri_obj

        
    def funcao_tri_obj(self, w):
        
        z = w.copy()
        z = np.asarray(z)
        z[0:self.card]=np.round(z[0:self.card])
        
        peso = np.zeros(self.C.shape[1])
        
        for i in range(0,self.card):
            peso[z[i].astype(int)]+=z[self.card+i]

        y = sum(peso)
        for i in range(len(peso)):
            peso[i]=peso[i]/y
            peso = np.nan_to_num(peso)
        
        self.C = np.nan_to_num(self.C)
        cart_otima = self.C.dot(peso)
        cart_otima[~np.isfinite(cart_otima)] = np.nan
        
        #Calculando o prêmio da carteira ótima
        retorno = cart_otima-self.risk_free
    
        #Calculando o sharpe
        
        #Calculando o omega da carteira
        omega = (retorno[retorno>0]).sum(axis=0) / abs(retorno[retorno<=0].sum(axis=0))
    
        cart_otima_acum = np.zeros(len(cart_otima))
        cart_otima_acum[0]=cart_otima[0]
        for i in range (1,len(cart_otima)):
            cart_otima_acum[i]=cart_otima[i]+cart_otima_acum[i-1]
    
        minclose = np.zeros(len(cart_otima))    
        minclose[0]=cart_otima_acum[0]
    
        for i in range(1, len(cart_otima)):
            if cart_otima_acum[i] > minclose[i-1]:
                minclose[i]=cart_otima_acum[i]
            else:
                minclose[i]=minclose[i-1]
    
        maxclose = np.zeros(len(cart_otima))
        maxclose[0] = cart_otima_acum[0]
        for i in range(1, len(cart_otima)):
            if cart_otima_acum[i] < maxclose[i-1]:
                maxclose[i]=cart_otima_acum[i]
            else:
                maxclose[i]=maxclose[i-1]
    
        drawdown = np.zeros(len(cart_otima))
        for i in range(1, len(cart_otima)):
            drawdown[i] = (cart_otima_acum[i]-minclose[i])
    
        drawdownmax=min(drawdown)
        
        corr3 = np.corrcoef(self.vix, cart_otima)[1][0]
        
        return [-omega,-drawdownmax, corr3]    
        
# In[002]
def multiglobal2 (mercado, jt01, jt02, popsize, nfe, seeds, diretorio, card):
    Obj = Multiobjcard()
    Obj.preparardados(diretorio,mercado, jt01, jt02, card)
    n_row, n_col = Obj.C.shape

    mutation=BitFlip(probability=0.10)
    print (mutation)
    variation=SBX(probability=0.9, distribution_index=15.0)
    
    #Comparação dos algoritmos
    algorits=[(NSGAII, {"population_size":popsize, "variator":GAOperator(variation,mutation)}),(NSGAIII, {"population_size":popsize, "variator":GAOperator(variation,mutation),"divisions_outer":12}),(GDE3, {"population_size":popsize, "variator":DifferentialEvolution(crossover_rate=0.8, step_size=0.5)}),(IBEA, {"population_size":popsize})]
    
    results = experiment(algorits, [Obj], seeds, nfe, display_stats=True)
    return results, card, n_col
   

# In[003]
def execucao():
    lista = [8]
    for cardinal in lista:
        random.seed(a=None, version=2)      
        parametros = [('DWJ','01/01/1994', '31/12/1998'),('DWJ','01/01/2004', '31/12/2008'),('DWJ','01/01/2014', '31/12/2018') ]
       #parametros = [('IBOV','01/01/1994', '31/12/1998'),('IBOV','01/01/2004', '31/12/2008'),('IBOV','01/01/2014', '31/12/2018') ]
    
        for mercado, jt01, jt02 in parametros:
         
            results, card, n_col = multiglobal2(mercado,jt01,jt02,50,50000,50, diretorio, cardinal)
        
            for algorithm in results.keys():
                for i in range(len(results[algorithm]['Multiobjcard'])):
                    for j in range(len(results[algorithm]['Multiobjcard'][i])):
                        if sum(results[algorithm]['Multiobjcard'][i][j].variables[card:2*card]) == 0:
                            results[algorithm]['Multiobjcard'][i][j] = results[algorithm]['Multiobjcard'][i-1][j]
        
            ordem = len(results[list(results.keys())[0]]['Multiobjcard'][0][0].variables)
        
            columns = ['Alg','Run']
            for i in range(ordem):
                columns.append('x'+str(i+1))
        
            ordem2 = len(results[list(results.keys())[0]]['Multiobjcard'][0][0].objectives)
            for i in range(ordem2):
                columns.append('f'+str(i+1))
            
            nondom = pd.DataFrame(columns=columns)
        
            for algorithm in results.keys():
                for run in range(len(results[algorithm]['Multiobjcard'])):
                    dic = {'Alg':algorithm, 'Run':run+1, 'Nvars':ordem}
                    sol = nondominated(results[algorithm]['Multiobjcard'][run])
                    for i in range(len(sol)):
                        sol[i].variables[0:cardinal]=list(np.round(np.array(sol[i].variables[0:cardinal])))
                        aux2 = sum(sol[i].variables[cardinal:2*cardinal])
                        for m in range(0,cardinal):
                            dic['x'+str(m+1)]=np.round(sol[i].variables[m]).astype(int)
                        for j in range(card,2*cardinal):
                            dic['x'+str(j+1)]=sol[i].variables[j]/aux2
                        for k in range(ordem2):
                            dic['f'+str(k+1)]=sol[i].objectives[k]
                        aux = pd.DataFrame(dic, index=[0])
                        nondom = pd.concat((nondom, aux), sort=False)
            nondom.to_csv(diretorio+'PIBICTEST'+algorithm+mercado+str(pd.to_datetime(jt01).year)+str(pd.to_datetime(jt02).year)+'card'+str(cardinal)+'.csv')
execucao()    

