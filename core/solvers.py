import numpy as np

def thomas_solver(a, b, c, d):
    """
    Risolve un sistema tridiagonale Ax = d in O(n).
    a: sotto-diagonale, b: diagonale principale, c: sopra-diagonale, d: termini noti.
    """
    n = len(d)
    c_star = np.zeros(n-1)
    d_star = np.zeros(n)
    x = np.zeros(n)

    # Forward sweep
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_star[i-1]
        c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / denom
    
    d_star[n-1] = (d[n-1] - a[n-2] * d_star[n-2]) / (b[n-1] - a[n-2] * c_star[n-2])

    # Back substitution
    x[n-1] = d_star[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i+1]
    
    return x