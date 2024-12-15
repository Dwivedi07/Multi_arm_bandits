import numpy as np
import matplotlib.pyplot as plt

mu = [2, 1.8, 1.5, 1]


m = [700,1200,1500]
regretall = np.zeros(len(m))
N = 10000

def ETC(mp,N):
    
    mu = [2, 1.8, 1.5, 1]
    reward = [0, 0, 0, 0]
    pulls = [0, 0, 0, 0]
    mu_hat = [0, 0, 0, 0]
    regret = []

    for i in range(4*mp):
        j = i//mp      #arm j will be pulled
        sample = np.random.normal(mu[j], 1)
        if i==0:
            regret.append(2 - sample)
        else:
            regret.append(regret[i-1] + 2- sample)
        reward[j] = reward[j]+ sample
        pulls[j] = pulls[j] + 1
        mu_hat[j] = reward[j]/pulls[j]

    optimal_arm = np.argmax(mu_hat)

    for j in range(N-4*mp):
        sample = np.random.normal(mu[optimal_arm], 1)
        regret.append(regret[i-1] + 2- sample)
        reward[optimal_arm] = reward[optimal_arm]+ sample
        pulls[optimal_arm] = pulls[optimal_arm] + 1
        mu_hat[optimal_arm] = reward[optimal_arm]/pulls[optimal_arm]
    
    
    return regret

for k in range(len(m)):
    Reg = np.zeros(N)
    for i in range(500):
        Reg = (1/(i+1))*(i*Reg + ETC(m[k],N))
    plt.plot(Reg,label=f'm= {m[k]}')
    
plt.title('Regret vs Pulls')
plt.xlabel('Pulls')
plt.ylabel('Regret')
plt.legend()
plt.grid(True)
plt.show()

