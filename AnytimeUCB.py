import numpy as np
import matplotlib.pyplot as plt

mu = [2, 1.8, 1.5, 1]


deltaD = [0.1, 0.09,0.08]

N = 5000

def AnUCB(delta,N):
    
    mu = [2, 1.8, 1.5, 1]
    reward = [0, 0, 0, 0]
    pulls = [0, 0, 0, 0]
    mu_hat = [0, 0, 0, 0]
    mu_hat_UCB = [0, 0, 0, 0]
    regret = []

    for j in range(4):
        sample = np.random.normal(mu[j], 1)
        if j==0:
            regret.append(2 - sample)
        else:
            regret.append(regret[j-1] + 2- sample)
        reward[j] = reward[j]+ sample
        pulls[j] = pulls[j] + 1
        mu_hat[j] = reward[j]/pulls[j]
        mu_hat_UCB[j] =mu_hat[j] + np.sqrt((8*np.log(j+1))/(pulls[j]))



    for j in range(N-4):
        optimal_arm = np.argmax(mu_hat_UCB)
        t = 4+j+1
        sample = np.random.normal(mu[optimal_arm], 1)
        regret.append(regret[j-1] + 2- sample)
        reward[optimal_arm] = reward[optimal_arm]+ sample
        pulls[optimal_arm] = pulls[optimal_arm] + 1
        mu_hat[optimal_arm] = reward[optimal_arm]/pulls[optimal_arm]
        mu_hat_UCB[optimal_arm] =mu_hat[optimal_arm] + np.sqrt((8*np.log(t))/(pulls[optimal_arm]))
    
    return regret


for j in range(len(deltaD)):
    print(j)
    Reg = np.zeros(N)
    for i in range(500):
        Reg = (1/(i+1))*(i*Reg + AnUCB(deltaD[j], N))
    plt.plot(Reg,label=f'Anytime UCB with delta ={deltaD[j]}')
    
plt.title('Regret vs Pulls')
plt.xlabel('Pulls')
plt.ylabel('Regret')
plt.legend()
plt.grid(True)
plt.show()

