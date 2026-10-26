| check | paper | claim | passed | detail |
|---|---|---|---|---|
| variance | Eq. 9 | N Var J_ij = (1-e/2)^2+(e/2)^2 = 1 - e + e^2/2 | ✅ | epsilon**2/2 - epsilon + 1 |
| eta | Eq. 10 | corr(J_ij,J_ji) = (1-e)/ups | ✅ | 1 - epsilon |
| antisymmetric part drops out | after Eq. 12 | sum S J S = (1-e/2) sum S J^S S | ✅ |  |
| R_inf | Eq. 13 | (1/N) S^T J S at SK ground state = 2 e0 (1-e/2) = e0 (2-e) | ✅ |  |
| elliptic-law edge | Eq. 20 | lambda_max = sqrt(ups)(1+eta) = (2-e)^2/(2 sqrt ups)  => T_c = lambda_max | ✅ |  |
| T_c from DMFT stability | Eq. 49-50 vs Eq. 20 | beta(1-e)chi = 1 - beta sqrt(ups) gives T_c(e) of Eq. 20 (paper's Eq. 49 lacks the beta) | ✅ | residual corrected=[0.0, 0.0, 0.0, 0.0], literal Eq.49=[0.5, 0.3896250481049452, 0.18801108758923143, -0.00042767966516321675] |
| EWMA noise | Eq. 18 / A2 | alpha^2 sum_k (1-alpha)^{2k} = alpha/(2-alpha) | ✅ | -alpha/(alpha - 2) |
| continuous limit | Eq. 33 | [Q(t+a)-Q(t)]/a = Q' + (a/2) Q'' + O(a^2) | ✅ |  |
| Sompolinsky-Crisanti | Eq. 37-39, D5 | Delta0 = 1-2/pi from U(Delta0)=0; tau1 = sqrt((pi-2)/(pi-3)) ~ 2.84 | ✅ | Delta0=[(-2 + pi)/pi], tau1=2.83946 |
| fixed-point complexity | Eq. 22-23 vs B36 | Sigma(1)=0.1992 (Tanaka-Edwards), Sigma(0)=0; Eq. 23 is the stationarity of Eq. 22 | ✅ | Sigma(1)=0.199228491035784, dS22/dx=9.11350349688699E-17, dSB36/dx(eta=1/2)=-0.306713042463949 |
| Sigma_2(alpha=1) = 2 Sigma_1 | Eq. B50 | F(r=k=v=0, w=x) = 2 x Eq.22 | ✅ |  |
| 2-cycle Jacobian | Eq. B37 | (J^alpha)^{-1} = [[1-a,1],[1,1-a]]/(1-(1-a)^2) | ✅ | alpha*(alpha - 2) |
| paramagnetic chi | Eq. 50 | chi = beta/(1-beta(1-e)chi) -> smaller root | ✅ | [(-sqrt(4*beta**2*epsilon - 4*beta**2 + 1) - 1)/(2*beta*(epsilon - 1)), (sqrt(4*beta**2*epsilon - 4*beta**2 + 1) - 1)/(2*beta*(epsilon - 1))] |
