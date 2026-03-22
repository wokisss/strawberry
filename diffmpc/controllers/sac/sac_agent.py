import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .networks import GaussianPolicy, QNetwork

class SAC(object):
    """SAC 核心算法 Agent"""
    def __init__(self, num_inputs, action_space, config, device):
        self.gamma = config.sac_gamma
        self.tau = config.sac_tau
        self.alpha = config.sac_alpha

        self.target_update_interval = config.sac_target_update_interval
        self.device = device

        # ---- Q 网络 ----
        self.critic = QNetwork(num_inputs, action_space.shape[0], config.sac_hidden_dim).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.sac_lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], config.sac_hidden_dim).to(device)
        # 初始时使 target 参数 = eval 参数
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # ---- 自动调节 Alpha (熵系�? ----
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=config.sac_lr)

        # ---- 策略网络 (Actor) ----
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], config.sac_hidden_dim, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config.sac_lr)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # 采样 Batch
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # 计算 Critic 损失 (MSE)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # 更新 Critic
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # 计算 Actor 损失
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # 更新 Actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # 计算 Alpha 损失
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        # 更新 Alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # 软更�?Target Critic
        if updates % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            try:
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

