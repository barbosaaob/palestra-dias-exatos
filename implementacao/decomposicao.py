import numpy as np
import matplotlib.pyplot as plt
import time


"""
Carrega a imagem a partir de um arquivo
"""
tic = time.time()
print("Carregando imagem...", end=" ", flush=True)
img = plt.imread("navio-original.png")
H, W, dim = img.shape
plt.imsave("out/original.png", img)
print(f"({time.time()-tic:.2f}s)")

"""
Calcula a decomposicao SVD
"""
tic = time.time()
print("SVD matriz vermelha...", end=" ", flush=True)
uR, dR, vR = np.linalg.svd(img[:,:,0], full_matrices=False)
print(f"({time.time()-tic:.2f}s)")
tic = time.time()
print("SVD matriz verde...", end=" ", flush=True)
uG, dG, vG = np.linalg.svd(img[:,:,1], full_matrices=False)
print(f"({time.time()-tic:.2f}s)")
tic = time.time()
print("SVD matriz azul...", end=" ", flush=True)
uB, dB, vB = np.linalg.svd(img[:,:,2], full_matrices=False)
print(f"({time.time()-tic:.2f}s)")

"""
Reconstroi a imagem
"""
nparcelas = 2448
nova_img = np.zeros(img.shape)
tic = time.time()
print(f"Reconstruindo a imagem com {nparcelas} parcelas...", end=" ", flush=True)
for i in range(nparcelas):
    nova_img[:,:,0] += dR[i] * uR[:,i].reshape(H,1) * vR[i,:].reshape(1,W)
    nova_img[:,:,1] += dG[i] * uG[:,i].reshape(H,1) * vG[i,:].reshape(1,W)
    nova_img[:,:,2] += dB[i] * uB[:,i].reshape(H,1) * vB[i,:].reshape(1,W)
# normalizacao [0,1]
nova_img = np.clip(nova_img, 0, 1)
print(f"({time.time()-tic:.2f}s)")

"""
Exibe a imagem reconstruida
"""
#tic = time.time()
print("Salvando a imagem...", end=" ", flush=True)
fig = plt.imshow(nova_img)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()
plt.imsave("out/nova" + str(nparcelas) + ".png", nova_img)
#print(f"({time.time()-tic:.2f}s)")
