import numpy as np
import matplotlib.pyplot as plt


"""
Carrega a imagem a partir de um arquivo
"""
img = plt.imread("navio-original.png")
H, W, dim = img.shape
plt.imsave("out/original.png", img)

"""
Calcula a decomposicao SVD
"""
uR, dR, vR = np.linalg.svd(img[:,:,0])
uG, dG, vG = np.linalg.svd(img[:,:,1])
uB, dB, vB = np.linalg.svd(img[:,:,2])

"""
Reconstroi a imagem
"""
nparcelas = 200
nova_img = np.zeros(img.shape)
for i in range(nparcelas):
    nova_img[:,:,0] += dR[i] * uR[:,i].reshape(H,1) * vR[i,:].reshape(1,W)
    nova_img[:,:,1] += dG[i] * uG[:,i].reshape(H,1) * vG[i,:].reshape(1,W)
    nova_img[:,:,2] += dB[i] * uB[:,i].reshape(H,1) * vB[i,:].reshape(1,W)

"""
Exibe a imagem reconstruida
"""
fig = plt.imshow(nova_img)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#plt.show()
plt.imsave("out/nova" + str(nparcelas) + ".png", nova_img)
