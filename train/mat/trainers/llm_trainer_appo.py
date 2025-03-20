import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mat.utils.util import get_gard_norm, huber_loss, mse_loss

class APPOTrainer:

    def __init__(self, args, agent, num_agents):
        self.tpdv = dict(dtype=torch.float32, device=torch.device("cuda:0")) # 数据类型、GPU设备
        self.agent = agent

        self.clip_param = args.clip_param # PPO 中的裁剪参数
        self.ppo_epoch = args.ppo_epoch # PPO 算法将执行的 epoch 数量（更高的 epoch 数量会增加策略更新的稳定性）
        self.num_mini_batch = args.num_mini_batch # PPO 中将经验数据拆分成的小批次数量
        self.value_loss_coef = args.value_loss_coef # 价值函数的损失系数
        self.max_grad_norm = args.max_grad_norm # 梯度裁剪的最大梯度范数      
        self.huber_delta = args.huber_delta # Huber 损失的阈值（误差较大时，Huber 损失变得更加平滑，防止过大的梯度）
        self.entropy_coef = args.entropy_coef # 熵正则化的系数（防止策略过早收敛到一个局部最优解）

        # 是否使用该值
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss

        self.lr = args.lr # 策略网络（actor）的学习率
        self.critic_lr = args.critic_lr # 价值网络（critic）的学习率
        self.opti_eps = args.opti_eps # 优化器的 epsilon 值
        self.gradient_cp_steps = args.gradient_cp_steps # 梯度检查点的步数（减少内存消耗）

        # 优化器
        self.policy_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.agent.actor.parameters()), lr=self.lr, eps=1e-5, weight_decay=0)
        self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.agent.critic.parameters()), lr=self.critic_lr, eps=1e-5)

    # 计算 policy_loss
    def cal_policy_loss(self, log_prob_infer, log_prob_batch, advantages_batch, entropy):
        
        # 当前策略与旧策略对相同动作的对数概率差值
        log_ratio = log_prob_infer - log_prob_batch 
        # 重要性采样权重 取指数e^x（因为取的对数比率）
        imp_weights = torch.exp(log_ratio)
        
        # 近似计算 KL 散度（信息增益、相对熵）的均值（KL 散度过大，说明策略发生了剧烈变化）
        # imp_weights - 1 近似计算这个差异
        # log_ratio 进行修正
        approx_kl = ((imp_weights - 1) - log_ratio).mean()
        
        # Surrogate Loss
        # 裁剪（目的是防止策略更新过大，从而避免优化过程中的不稳定性）
        # 裁剪后的权重保持在 [1.0 - clip_param, 1.0 + clip_param] 之间
        # 为什么是负值：为了梯度下降，通过 最小化负的目标函数 来间接地 最大化 奖励
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        # 未裁剪的目标函数
        # 直接乘优势函数值（优势函数值：表示当前动作相较于平均水平的好坏）
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        # 目标函数的均值 - 熵的正则化项
        # 熵是用来衡量策略的多样性或探索性的
        # 为什么减：减少策略的随机性
        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl
        
    # 计算 value_loss
    def cal_value_loss(self, values_infer, value_preds_batch, return_batch):
        # values_infer: 当前策略（或者模型）预测的值函数（例如，状态的估计价值）。
        # value_preds_batch: 样本批次中的真实值函数预测值（即目标值函数），通常是从环境中收集的实际回报或通过某些算法得到的估计值。
        # return_batch: 样本批次中的实际回报

        # 将真实的值函数预测值与当前预测值之间的差异限制在clip范围内
        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        # 计算误差：带clip和不带clip的两种情况
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer

        # 根据是否使用Huber损失函数计算损失
        if self._use_huber_loss:
            # 使用Huber损失
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            # 使用均方误差损失
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        # 选择clip和不clip情况下的最大损失，并取平均值
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        # 返回加权后的值损失（乘以value_loss_coef系数）
        return value_loss * self.value_loss_coef

    def ppo_update(self, sample):
        # 提取输入样本
        obs_batch, action_batch, log_prob_batch, \
            value_preds_batch, return_batch, advantages_batch, action_tokens_batch = sample

        # 将数据转移到GPU并转换为张量
        log_prob_batch = torch.from_numpy(log_prob_batch).to("cuda")
        value_preds_batch = torch.from_numpy(value_preds_batch).to("cuda")
        return_batch = torch.from_numpy(return_batch).to("cuda")
        advantages_batch = torch.from_numpy(advantages_batch).to("cuda")
        action_tokens_batch = torch.from_numpy(action_tokens_batch).to("cuda")
        batch_size = obs_batch.shape[0]
        
        # critic update
        # 1. Critic网络更新
        values_infer = self.agent.get_action_values(np.concatenate(obs_batch))
        values_infer = values_infer.view(batch_size, -1)
        
        # 计算值损失
        value_loss = self.cal_value_loss(values_infer, value_preds_batch, return_batch)
        # print("value_loss: ", value_loss)
        
        # 清零梯度，计算反向传播，梯度裁剪（如果启用），然后更新critic网络
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            # 如果启用最大梯度裁剪
            critic_grad_norm = nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
        else:
            # 否则手动计算梯度范数
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        value_loss = value_loss.item()  # 转换为数值
        self.critic_optimizer.zero_grad()
        critic_grad_norm = critic_grad_norm.item()  # 转换为数值

        # policy update
        # 2. Policy网络更新
        self.policy_optimizer.zero_grad()
        cp_batch_size = int(batch_size // self.gradient_cp_steps)  # 分批次进行处理
        total_approx_kl = 0  # 用于累积KL散度

        # 按照批次进行处理
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            log_prob_infer, entropy = self.agent.infer_for_action_update(np.concatenate(obs_batch[start:end]), 
                                                                         action_tokens_batch[start:end].view(-1, action_tokens_batch.shape[-1]))
        
            log_prob_infer = log_prob_infer.view(obs_batch[start:end].shape[0], -1)  # 重塑log_prob_infer形状
            
            # 对优势进行标准化
            cp_adv_batch = advantages_batch[start:end]
            cp_adv_batch = (cp_adv_batch - cp_adv_batch.mean()) / (cp_adv_batch.std() + 1e-8)
            
            entropy = entropy.view(obs_batch[start:end].shape[0], -1)   # 重塑熵的形状
            policy_loss, approx_kl = self.cal_policy_loss(log_prob_infer, log_prob_batch[start:end], cp_adv_batch, entropy)
            total_approx_kl += approx_kl / self.gradient_cp_steps  # 累加KL散度
            
            # print("policy_loss: ", policy_loss)
            
            # 反向传播计算策略损失
            policy_loss /= self.gradient_cp_steps  # 平均损失
            policy_loss.backward()
        # 如果KL散度过大，停止优化
        if total_approx_kl > 0.02:
            self.policy_optimizer.zero_grad()
            return value_loss, critic_grad_norm, 0, 0
            
        # 梯度裁剪并更新策略网络
        policy_grad_norm = nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        policy_loss = policy_loss.item()   # 转换为数值
        self.policy_optimizer.zero_grad()
        policy_grad_norm = policy_grad_norm.item()   # 转换为数值
        
        return value_loss, critic_grad_norm, policy_loss, policy_grad_norm

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info['value_loss'] = 0        # 初始化值损失
        train_info['value_grad_norm'] = 0   # 初始化值网络梯度范数
        train_info['policy_loss'] = 0       # 初始化策略损失
        train_info['policy_grad_norm'] = 0  # 初始化策略网络梯度范数

        update_time = 0  # 更新次数计数器

        # 多次进行PPO迭代更新
        for _ in range(self.ppo_epoch):
            # 从buffer中生成小批量训练数据
            data_generator = buffer.appo_sampler(self.num_mini_batch)
            for sample in data_generator:
                # 使用PPO算法进行一次更新，并获得损失和梯度范数
                value_loss, value_grad_norm, policy_loss, policy_grad_norm = self.ppo_update(sample)
                # 累加各项损失和梯度范数
                train_info['value_loss'] += value_loss
                train_info['value_grad_norm'] += value_grad_norm
                train_info['policy_loss'] += policy_loss
                train_info['policy_grad_norm'] += policy_grad_norm
                update_time += 1  # 记录更新次数

        # 平均化训练信息（根据更新次数）
        for k in train_info.keys():
            train_info[k] /= update_time
 
        return train_info  # 返回训练信息（如损失、梯度范数等）

    def prep_training(self):
        # 将actor和critic网络设置为训练模式
        self.agent.actor().train()
        self.agent.critic().train()

    def prep_rollout(self):
        # 将actor和critic网络设置为评估模式
        self.agent.actor().eval()
        self.agent.critic().eval()
