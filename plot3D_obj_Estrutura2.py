import numpy as np
import Otimizador
import Models
import Tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import lmfit as lm
from matplotlib import cm


def modelo_analitico_3dplot(t, S_0, alpha, kr, kl, k2, k3):
    
   def SA(t, S_0, alpha, kr, kl):
      Sa = S_0*(alpha*np.exp(-kr*t) + (1-alpha)*np.exp(-kl*t))
      return Sa

   def SB(t, S_0, alpha, kl, k2):
      Sb = S_0*((1-alpha)*kl*(np.exp(-kl*t) - np.exp(-k2*t))/(k2 - kl))
      return Sb

   def SC(t, S_0, alpha, kl, kr, k2, k3):
      Sc = S_0 * (alpha*kr*(np.exp(-kr*t) - np.exp(-k3*t))/(k3 - kr) + (1-alpha)* kl * k2 * ((k3 - k2)*np.exp(-kl*t) - (k3 - kl)*np.exp(-k2*t) + (k2 - kl)*np.exp(-k3*t))/((k2 - kl)*(k3 - kl)*(k3-k2)))
      return Sc    

   def SD(t, S_max, kl, kr, k2, k3, alpha):
      Sd = S_max*(alpha*(1-np.exp(-kr*t) - kr*(np.exp(-kr*t) - np.exp(-k3*t))/(k3 - kr)) + (1-alpha)*(1-np.exp(-kl*t) - kl*(np.exp(-kl*t) - np.exp(-k2*t))/(k2 - kl) - kl * k2 * ((k3 - k2)*np.exp(-kl*t) - (k3 - kl)*np.exp(-k2*t) + (k2 - kl)*np.exp(-k3*t))/((k2 - kl)*(k3 - kl)*(k3-k2))))
      return Sd
   
        
   sim_sa = SA(t, S_0, alpha, kr, kl)
   sim_sb = SB(t, S_0, alpha, kl, k2)
   sim_sc = SC(t, S_0, alpha, kl, kr, k2, k3)
   sim_sd = SD(t, S_0, kl, kr, k2, k3, alpha)
   
   
      
   return [sim_sa+sim_sb+sim_sc, sim_sb, sim_sc, sim_sd]



parametrosDadosXlsx:list[int] = [0, 240, 25, 3]
data_fit_P = Tools.ajustarXlsx("./xlsx1/dados_gouveia_rao_produto.csv", parametrosDadosXlsx)
data_fit_S = Tools.ajustarXlsx("./xlsx1/dados_gouveia_rao_substrato.csv", parametrosDadosXlsx)
data_fit_I = Tools.ajustarXlsx("./xlsx1/dados_gouveia_rao_AGV.csv", parametrosDadosXlsx)

pars = (38.1, 0.3532, 0.1532, 0.0133, 0.1274, 0.1181)
sima, simb, simc, simd = modelo_analitico_3dplot(data_fit_P['tempo'], *pars)
plt.plot(data_fit_P['tempo'], simd)
plt.show()

alpha_array = np.linspace(0, 1, 100)
kl_array = np.linspace(0, 0.1, 100)
kr_array = np.linspace(0, 2, 100)
k2_array = np.linspace(0, 2, 100)
k3_array = np.linspace(0, 2, 100)


print(alpha_array.shape)
Y = np.zeros((alpha_array.shape[0], data_fit_P['tempo'].shape[0]))
Y_kl = np.zeros((kl_array.shape[0], data_fit_P['tempo'].shape[0]))
Y_kr = np.zeros((kr_array.shape[0], data_fit_P['tempo'].shape[0]))
Y_k2 = np.zeros((k2_array.shape[0], data_fit_P['tempo'].shape[0]))
Y_k3 = np.zeros((k3_array.shape[0], data_fit_P['tempo'].shape[0]))
print(Y.shape)

for i, u in enumerate(alpha_array):
    print(f'i é {i} e u é {u}')
    sim_s_, sim_sb_, sim_sc_, sim_sd_ = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, u, 0.1532, 0.0133, 0.1274, 0.1181)
    Y[i][:] = np.square(data_fit_P['concentração'] - sim_sd_)
    plt.plot(data_fit_P['tempo'], Y[i], label=f'$\\alpha$ = {u:.3f}')

plt.legend()
plt.show()

for i, u in enumerate(kl_array):
    print(f'i é {i} e u é {u}')
    sim_s_, sim_sb_, sim_sc_, sim_sd_ = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, 0.3532, 0.1532, u, 0.1274, 0.1181)
    Y_kl[i][:] = np.square(data_fit_P['concentração'] - sim_sd_)
    plt.plot(data_fit_P['tempo'], Y_kl[i], label=f'$k_l$ = {u:.3f}')

plt.legend()
plt.show()

for i, u in enumerate(kr_array):
    print(f'i é {i} e u é {u}')
    sim_s_, sim_sb_, sim_sc_, sim_sd_ = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, 0.3532, u, 0.0133, 0.1274, 0.1181)
    Y_kr[i][:] = np.square(data_fit_P['concentração'] - sim_sd_)
    plt.plot(data_fit_P['tempo'], Y_kr[i], label=f'$k_r$ = {u:.3f}')

plt.legend()
plt.show()

for i, u in enumerate(k2_array):
    print(f'i é {i} e u é {u}')
    sim_s_, sim_sb_, sim_sc_, sim_sd_ = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, 0.3532, 0.1532, 0.0133, u, 0.1181)
    Y_k2[i][:] = np.square(data_fit_P['concentração'] - sim_sd_)
    plt.plot(data_fit_P['tempo'], Y_k2[i], label=f'$k_2$ = {u:.3f}')

plt.legend()
plt.show()

for i, u in enumerate(k3_array):
    print(f'i é {i} e u é {u}')
    sim_s_, sim_sb_, sim_sc_, sim_sd_ = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, 0.3532, u, 0.0133, 0.1274, 0.1181)
    Y_k3[i][:] = np.square(data_fit_P['concentração'] - sim_sd_)
    plt.plot(data_fit_P['tempo'], Y_k3[i], label=f'$k_3$ = {u:.3f}')

plt.legend()
plt.show()

x = alpha_array
print(f'shape x é: {x.shape}')
y = data_fit_P['tempo']
print(f'shape y é: {y.shape}')

xgrid, ygrid = np.meshgrid(x, y)

fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
surf = ax.plot_surface(xgrid, ygrid, np.transpose(Y), cmap='terrain')
ax.set_xlabel(f'$\\alpha$')
ax.set_ylabel('t')
ax.set_zlabel(f"$\\psi_P$")
fig.colorbar(surf)
plt.show()

x = kl_array
xgrid, ygrid = np.meshgrid(x, y)
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
surf = ax.plot_surface(xgrid, ygrid, np.transpose(Y_kl), cmap='terrain')
ax.set_xlabel(f'$k_l$')
ax.set_ylabel('t')
ax.set_zlabel(f'$\\psi_P$')
fig.colorbar(surf)
plt.show()

x = kr_array
xgrid, ygrid = np.meshgrid(x, y)
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
surf = ax.plot_surface(xgrid, ygrid, np.transpose(Y_kr), cmap='terrain')
ax.set_xlabel(f'$k_r$')
ax.set_ylabel('t')
ax.set_zlabel(f'$\\psi_P$')
fig.colorbar(surf)
plt.show()

x = k2_array
xgrid, ygrid = np.meshgrid(x, y)
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
surf = ax.plot_surface(xgrid, ygrid, np.transpose(Y_k2), cmap='terrain')
ax.set_xlabel(f'$k_2$')
ax.set_ylabel('t')
ax.set_zlabel(f'$\\psi_P$')
fig.colorbar(surf)
plt.show()

x = k3_array
xgrid, ygrid = np.meshgrid(x, y)
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
surf = ax.plot_surface(xgrid, ygrid, np.transpose(Y_k3), cmap='terrain')
ax.set_xlabel(f'$k_3$')
ax.set_ylabel('t')
ax.set_zlabel('$\\psi_P$')
fig.colorbar(surf)
plt.show()

Y_2 = np.zeros((alpha_array.shape[0], 1))

Y_3 = np.zeros((alpha_array.shape[0], 1))


for i in range(len(alpha_array)):
    for j in range(len(alpha_array)):
    # print(alpha_array[i], kl_array[i])
        sima, simb, simc, simd = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, alpha_array[i], 0.1532, kl_array[i], 0.1274, 0.1181)
        Y_2[i] = np.sum(np.square(data_fit_P['concentração'] - simd))
        print(f'alpha[i]: {alpha_array[i]}; kl[i]: {kl_array[i]}; Y[i]: {Y_2[i]}')
    
    
# print(Y_2)    
x = alpha_array
y = kl_array
xgrid, ygrid = np.meshgrid(x, y)

print(f'xgrid: {xgrid}; ygrid: {ygrid}')

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, Y_2, cmap='terrain')
ax.set_xlabel('alpha')
ax.set_ylabel('kl')
ax.set_zlabel('obj')
plt.show()

for i in range(len(alpha_array)):
    # print(alpha_array[i], kl_array[i])
    sima, simb, simc, simd = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, alpha_array[i], kr_array[i], 0.0133, 0.1274, 0.1181)
    Y_3[i] = np.sum(np.square(data_fit_P['concentração'] - simd)) 
    
x = alpha_array
y = kr_array
xgrid, ygrid = np.meshgrid(x, y)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, Y_3, cmap='terrain')
ax.set_xlabel('alpha')
ax.set_ylabel('kr')
ax.set_zlabel('obj')
plt.show()

for i in range(len(alpha_array)):
    # print(alpha_array[i], kl_array[i])
    sima, simb, simc, simd = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, 0.3532, kr_array[i], kl_array[i], 0.1274, 0.1181)
    Y_3[i] = np.sum(np.square(data_fit_P['concentração'] - simd))
    
x = kr_array
y = kl_array
xgrid, ygrid = np.meshgrid(x, y)
    
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, Y_3, cmap='terrain')
ax.set_xlabel('kr')
ax.set_ylabel('kl')
ax.set_zlabel('obj')
plt.show()

for i in range(len(alpha_array)):
    # print(alpha_array[i], kl_array[i])
    sima, simb, simc, simd = modelo_analitico_3dplot(data_fit_P['tempo'], 38.1, 0.3532, 0.1532, kl_array[i], k2_array[i], 0.1181)
    Y_3[i] = np.sum(np.square(data_fit_P['concentração'] - simd))
    
x = k2_array
y = kl_array
xgrid, ygrid = np.meshgrid(x, y)

print(Y_3)
    
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, Y_3, cmap='terrain')
ax.set_xlabel('k2')
ax.set_ylabel('kl')
ax.set_zlabel('obj')
plt.show()
    
paras = lm.Parameters()
paras.add('S_max', value=38.1, vary=False) # gCOD/L
paras.add('Kl', value=0.0133, vary=False) # dia^-1
paras.add('Kr', value=0.1532, vary=False) # dia^-1
paras.add('K2', value=0.1274, vary=False) # dia^-1
paras.add('K3', value=0.1181, vary=False) # dia^-1
paras.add('alpha', value=0.3532, vary=False) # [-]

sim_s, sim_sb, sim_sc, sim_sd = Models.modelo_analitico(data_fit_P['tempo'], paras, False, None)

paras2 = lm.Parameters()
paras2.add('S_max', value=38.1, vary=False) # gCOD/L
paras2.add('Kl', value=0.0133, vary=False) # dia^-1
paras2.add('Kr', value=0.1532, vary=False) # dia^-1
paras2.add('K2', value=0.1274, vary=False) # dia^-1
paras2.add('K3', value=0.1181, vary=False) # dia^-1
paras2.add('alpha', value=0.832, vary=False) # [-]

sim_s2, sim_sb2, sim_sc2, sim_sd2 = Models.modelo_analitico(data_fit_P['tempo'], paras2, False, None)

# sim_s, sim_sb, sim_sc, sim_sd = sim
plt.plot(data_fit_P["tempo"], sim_sd, label='sd')
plt.plot(data_fit_P['tempo'], sim_sd2, label='sd2')
plt.plot(data_fit_P['tempo'], simd, label='simd')
plt.legend()
plt.show()

obj = Otimizador.obj_modelo_analitico_P(paras, data_fit_P['concentração'], data_fit_P['tempo'])
obj2 = Otimizador.obj_modelo_analitico_P(paras2, data_fit_P['concentração'], data_fit_P['tempo'])
plt.plot(data_fit_P['tempo'], obj, label='obj')
plt.plot(data_fit_P['tempo'], obj2, label='obj2')
plt.legend()
plt.show()