import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import Tools
import Models
import Otimizador

# Dados experimentais:
parametrosDadosXlsx:list[int] = [0, 240, 25, 3]
data_fit_P = Tools.ajustarXlsx("./xlsx1/dados_gouveia_rao_produto.csv", parametrosDadosXlsx)
data_fit_S = Tools.ajustarXlsx("./xlsx1/dados_gouveia_rao_substrato.csv", parametrosDadosXlsx)
data_fit_I = Tools.ajustarXlsx("./xlsx1/dados_gouveia_rao_AGV.csv", parametrosDadosXlsx)
data = [data_fit_S,  data_fit_P]
t = data_fit_P['tempo']
t_plot = np.linspace(min(t), max(t), 1000)

## Parâmetros do modelo otimizado de Gouvea et al. (2022):
paras = lm.Parameters()
paras.add('S_max', value=38.1, vary=False) # gCOD/L
paras.add('Kl', value=0.0133, vary=False) # dia^-1
paras.add('Kr', value=0.1532, vary=False) # dia^-1
paras.add('K2', value=0.1274, vary=False) # dia^-1
paras.add('K3', value=0.1181, vary=False) # dia^-1
paras.add('alpha', value=0.3532, vary=False) # [-]


resolucao = Models.modelo_analitico(t_plot, paras, False, None) #[sA, Sb, Sc, Sd]
    
R2_S_P = Models.modelo_analitico(t, paras, True, data)

data_multivar_SP = [data_fit_S, data_fit_P]
data_multivar_SPI = [data_fit_S, data_fit_I, data_fit_P]


objP = Otimizador.obj_modelo_analitico_P(paras, data_fit_P['concentração'], t)
objSP = Otimizador.obj_modelo_analitico_SP(paras, data_multivar_SP, t)
objSPI = Otimizador.obj_modelo_analitico_SPI(paras, data_multivar_SPI, t)

print(f'Valor da função objetivo P para os parâmetros otimizados: {np.sum(np.square(objP))}\n')
print(f'Valor da função objetivo SP para os parâmetros otimizados: {np.sum(np.square(objSP))}\n')
print(f'Valor da função objetivo SPI para os parâmetros otimizados: {np.sum(np.square(objSPI))}\n')


plt.plot(t, data_fit_P['concentração'], 'rx', label=f'$S_D  exp.$')
plt.plot(t, data_fit_S['concentração'], 'bx', label=f'$S_A  exp.$')
plt.plot(t_plot, resolucao[0], 'b', label=f'$S_A (R^2={R2_S_P[0]:.3f})$')
plt.plot(t_plot, resolucao[1], 'y', label='$S_B$')
plt.plot(t_plot, resolucao[2], 'g', label=f'$S_C$')
plt.plot(t_plot, resolucao[3], 'r', label=f'$S_D (R^2={R2_S_P[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()


#Estimação de parâmetros:
## Ajuste de Produto:

paras = lm.Parameters()
paras.add('S_max', value=36.55, min=0)
paras.add('Kl', value=1.84702752e-02/2, min=0) 
paras.add('Kr', value=1.84702752e-02, min=0) 
paras.add('K2', value = 1.84702752e-02 * 2, min=0) 
paras.add('K3', value=1.84702752e-02 * 3, min=0)
paras.add('alpha', value=0.5, min=0) # [-]

# paras.add('S_max', value=36., min=0)
# paras.add('Kl', value=0.0012, min=0) 
# paras.add('Kr', value=0.017, min=0) 
# paras.add('K2', value = 0.008, min=0) 
# paras.add('K3', value=0.4, min=0)
# paras.add('alpha', value=0.3, min=0) # [-]

res_otim_P = lm.minimize(Otimizador.obj_modelo_analitico_P, paras, 'ampgo', args=(data_fit_P['concentração'], data_fit_P['tempo']), nan_policy='omit')
print('\nResultado da otimização de P\n')

print(f'Valor da função objetivo para os parâmetros otimizados: {np.sum(np.square(res_otim_P.residual))}\n')
lm.report_fit(res_otim_P)

sim_ajuste_P = Models.modelo_analitico(t_plot, res_otim_P.params, False, None) #[sA, Sb, Sc, Sd]
    
R2_S_P_ajuste_P = Models.modelo_analitico(t, res_otim_P.params, True, data)

# Plotagem dos resultados da estimação de parâmetros de P:
# plt.title('Simulação do ajuste de P')
plt.plot(t, data_fit_P['concentração'], 'rx', label=f'$S_D$ exp.')
plt.plot(t, data_fit_S['concentração'], 'bx', label=f'$S_A$ exp.')
plt.plot(t_plot, sim_ajuste_P[0], 'b', label=f'$S_A (R^2={R2_S_P_ajuste_P[0]:.3f})$')
plt.plot(t_plot, sim_ajuste_P[1], 'y', label='$S_B$')
plt.plot(t_plot, sim_ajuste_P[2], 'g', label=f'$S_C$')
plt.plot(t_plot, sim_ajuste_P[3], 'r', label=f'$S_D (R^2={R2_S_P_ajuste_P[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()

# Estimação multivariável de parâmetros SP:

paras = lm.Parameters()
paras.add('S_max', value=20, min=0)
paras.add('Kl', value=1.8, min=0) 
paras.add('Kr', value=1., min=0) 
paras.add('K2', value = 0.1, min=0) 
paras.add('K3', value=10, min=0)
paras.add('alpha', value=0.1, min=0) # [-]

res_otim_SP = lm.minimize(Otimizador.obj_modelo_analitico_SP, paras, 'ampgo', args=(data_multivar_SP, t), nan_policy='omit')

print('\nResultado da otimização multivariada de SP\n')

print(f'Valor da função objetivo para os parâmetros otimizados: {np.sum(np.square(res_otim_SP.residual))}\n')
lm.report_fit(res_otim_SP)

sim_ajuste_SP = Models.modelo_analitico(t_plot, res_otim_SP.params, False, None) #[sA, Sb, Sc, Sd]
    
R2_S_P_ajuste_SP = Models.modelo_analitico(t, res_otim_SP.params, True, data)

# Plotagem dos resultados da estimação de parâmetros:
# plt.title('Simulação do ajuste multivariável SP')
plt.plot(t, data_fit_P['concentração'], 'rx', label=f'$S_D$ exp.')
plt.plot(t, data_fit_S['concentração'], 'bx', label=f'$S_A$ exp.')
plt.plot(t_plot, sim_ajuste_SP[0], 'b', label=f'$S_A (R^2={R2_S_P_ajuste_SP[0]:.3f})$')
plt.plot(t_plot, sim_ajuste_SP[1], 'y', label='$S_B$')
plt.plot(t_plot, sim_ajuste_SP[2], 'g', label=f'$S_C$')
plt.plot(t_plot, sim_ajuste_SP[3], 'r', label=f'$S_D (R^2={R2_S_P_ajuste_SP[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()




# Estimação multivariável de parâmetros SPI:
###parametros iniciais segundo Gouveia 
paras = lm.Parameters()
paras.add('S_max', value=36.55, min=0)
paras.add('Kl', value=1.84702752e-02/2, min=0) 
paras.add('Kr', value=1.84702752e-02, min=0) 
paras.add('K2', value = 1.84702752e-02 * 2, min=0) 
paras.add('K3', value=1.84702752e-02 * 3, min=0)
paras.add('alpha', value=0.5, min=0) # [-]


# paras = lm.Parameters()
# paras.add('S_max', value=38.16, vary=False)
# paras.add('Kl', value=1.84702752e-02/2, min=0) 
# paras.add('Kr', value=1.84702752e-02, min=0) 
# paras.add('K2', value = 1.84702752e-02 * 2, min=0) 
# paras.add('K3', value=1.84702752e-02 * 3, min=0)
# paras.add('alpha', value=0.3532, vary=False) # [-]




res_otim_SPI = lm.minimize(Otimizador.obj_modelo_analitico_SPI, paras, 'ampgo', args=(data_multivar_SPI, t), nan_policy='omit')

print('\nResultado da otimização multivariada SPI\n')

print(f'Valor da função objetivo para os parâmetros otimizados: {np.sum(np.square(res_otim_SPI.residual))}\n')
lm.report_fit(res_otim_SPI)

sim_ajuste_SPI = Models.modelo_analitico(t_plot, res_otim_SPI.params, False, None) #[sA, Sb, Sc, Sd]
    
R2_S_P_ajuste_SPI = Models.modelo_analitico(t, res_otim_SPI.params, True, data_multivar_SPI)

# Plotagem dos resultados da estimação de parâmetros:
# plt.title('Simulação do ajuste multivariável SPI')
plt.plot(t, data_fit_S['concentração'], 'bx', label=f'$S_A$ exp.')
plt.plot(t, data_fit_I['concentração'], 'gx', label=f'$S_C$ exp.')
plt.plot(t, data_fit_P['concentração'], 'rx', label=f'$S_D$ exp.')
plt.plot(t_plot, sim_ajuste_SPI[0], 'b', label=f'$S_A (R^2={R2_S_P_ajuste_SPI[0]:.3f})$')
plt.plot(t_plot, sim_ajuste_SPI[1], 'y', label='$S_B$')
plt.plot(t_plot, sim_ajuste_SPI[2], 'g', label=f'$S_C (R^2={R2_S_P_ajuste_SPI[1]:.3f})$')
plt.plot(t_plot, sim_ajuste_SPI[3], 'r', label=f'$S_D (R^2={R2_S_P_ajuste_SPI[2]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()







    
    