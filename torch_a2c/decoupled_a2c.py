import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torch_a2c.model import ValueNetwork, PolicyNetwork


# unlike A2CAgent in a2c.py, here I separated value and policy network.
class A2CAgent():

    """
    init function
    input: env, which is the CartPole-v0
           gamma, 0.99 in this case
           lr, learning rate is 1e-4

    define: env = env, which is the CartPole-v0
            obs_dim: 4 obervations
                    Observation:
                    Type: Box(4)
                    Num	Observation                 Min         Max
                    0	Cart Position             -4.8            4.8
                    1	Cart Velocity             -Inf            Inf
                    2	Pole Angle                 -24 deg        24 deg
                    3	Pole Velocity At Tip      -Inf            Inf

            action_dim: 2 actions
                        Actions:
                        Type: Discrete(2)
                        Num	Action
                        0	Push cart to the left
                        1	Push cart to the right

            value_network: two layer network with input 4 (observation dim) and output 1 (reward?)
            policy_network: two layer network with input 4 (observation dim) and output 2 (action dim)

            value and policy optimizer using default Adam and learning rate
    """

    def __init__(self, env, gamma, lr):

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = gamma
        self.lr = lr

        self.value_network = ValueNetwork(self.obs_dim, 1)
        self.policy_network = PolicyNetwork(self.obs_dim, self.action_dim)

        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)


    """
    input state to get the next action
    
    using policy network to get the next state by using softmax
    """

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.policy_network.forward(state)
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()


    """
    form trajectory get all of the information, and calculated the discounted_rewards
    use value network to train the states with new values and compute the loss between the value and target value by using MSE
    same logic for policy network
    
    FloatTensor = FLOAT TYPE ARRAY
   
   t
tensor([[1, 2, 3],
        [4, 5, 6]])
t.view(-1,1)
tensor([[1],
        [2],
        [3],
        [4],
        [5],
        [6]]) 
     
    """
    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory])
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory])
        next_states = torch.FloatTensor([sars[3] for sars in trajectory])
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1)

        # compute value target

        ## Two for loop to calculate the discounted reward for each one
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma ** i for i in range(rewards[j:].size(0))]) \
                                        * rewards[j:]) for j in
                              range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1)

        # compute value loss
        values = self.value_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())

        # compute policy loss with entropy bonus
        logits = self.policy_network.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)

        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()

        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean() - 0.001 * entropy

        return value_loss, policy_loss

    """
        zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).

        loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        
        opt.step() causes the optimizer to take a step based on the gradients of the parameters.
        """

    def update(self, trajectory):
        value_loss, policy_loss = self.compute_loss(trajectory)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()