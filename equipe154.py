import numpy as np
import matplotlib.pyplot as plt
from suiteSn import SuiteSn


# A
print(f"a = \n {(a := np.full((6, 1), 1))}")

# B
print(f"b = {(b := np.arange(1, 7))}")

# C
print(f"c = {(c := a.reshape(6))}")

# D
print(f"d = {(d := 154 * c)}")

# I
print(f"i = \n {(I := np.identity(6))}")

# J
print(f"j = \n {(J := np.full((6, 6), 1))}")

# K
print(f"k = \n {(K := np.diag(b))}")

# L
print(f"L = {(L := (55 * I) - J + (2 * (a * c)))}")

# M
M = K.copy(); M[:, 0] = a.reshape(6); print(f"M = \n {M}")

# dd
print(f"dd = {(dd := np.linalg.det(M))}")

# x
print(f"x = {np.linalg.solve(M, a)}")

# N
print(f"N = \n {(N := np.linalg.solve(M, M.T))}")

# figure 1 
plt.matshow(N, cmap="RdBu_r", vmin=-1, vmax=1.2)
plt.colorbar()

# retirer les identification des cases superflues
plt.xticks([]) 
plt.yticks([])

# séparer les cases avec des lignes
for i in range(N.shape[0]+1):
    plt.axhline(i-0.5, color='black', linewidth=1)
for j in range(N.shape[1]+1):
    plt.axvline(j-0.5, color='black', linewidth=1)

# afficher les valeurs dans les cases
for i in range(N.shape[0]):
    for j in range(N.shape[1]):
        plt.text(j, i, f"{N[i, j]:.2f}", ha= "center", va= "center")

plt.title("Figure 1 : Matrice N")
plt.show()

# fonction f(z)
def f(z):
    fonction = -z**2 / 2 + np.exp(z) + np.sin(z)
    return fonction

# vecteur X 
x = np.linspace(0, 1, 101)

# figure 2 graphique de la fonction f(z) evaluée en x
plt.plot(x, f(x), linewidth=2, color="green")
plt.grid(True)
plt.title("Figure 2 : représentation graphique de f sur [0, 1]")
plt.xlabel("x")
plt.ylabel(r"$f(x)$")
plt.show()

# numero 2 C
S = SuiteSn(19)
n = np.arange(0, 20)

# graphique
plt.plot(n, S, color="m",  marker="o", markersize="3")
plt.title(r"Graphique des valeurs de $S_n$ en fonction de n ∈ [0, 19] ")
plt.xlabel("n")
plt.ylabel(r"$S_n$")
plt.xticks(n)
plt.show()