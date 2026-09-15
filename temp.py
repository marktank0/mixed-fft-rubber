E_matrix = 2
v_matrix = 0.49999


mu_matrix = E_matrix/(2*(1+v_matrix))
k_matrix = E_matrix/(3*(1-2*v_matrix))

print(f"mu: {mu_matrix}, k: {k_matrix}, ratio: {k_matrix/mu_matrix}")


