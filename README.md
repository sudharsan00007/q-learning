# Q Learning Algorithm


## AIM

To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT

Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM

Step 1:
Initialize Q-table and hyperparameters.

Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

Step 3:
After training, derive the optimal policy from the Q-table.

Step 4:
Implement the Monte Carlo method to estimate state values.

Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
### Name: SUDHANRSAN S
### Register Number:212224040335

python:
```
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
 ```






## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.


![image](https://github.com/user-attachments/assets/a0bb4fc2-5a5f-480a-a77c-be168552b339)

![image](https://github.com/user-attachments/assets/8a8354c3-de22-4bf5-8f2f-fe64c33f793a)

![image](https://github.com/user-attachments/assets/9a2d49ff-b955-4353-b0ee-3b5c2d5b604f)

![image](https://github.com/user-attachments/assets/dc0ec68c-391e-44f9-b4ec-603f4a0c94c0)

![image](https://github.com/user-attachments/assets/c839b1b6-d4ba-45c9-a974-a513af8cce0d)

![image](https://github.com/user-attachments/assets/847f7fab-3ba2-417b-8f6f-5f39221e1b51)


Include plot comparing the state value functions of Monte Carlo method and Qlearning.

## RESULT:

Write your result here
