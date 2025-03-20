import time
import os
import numpy as np
from functools import reduce
import torch
from tensorboardX import SummaryWriter
from mat.agents.qwen_lora_agent import QwenLoRAgent
from mat.models.ms_prm import MSProcessRM
from mat.models.qwen_prm import QwenProcessRM
from mat.utils.language_buffer import LanguageBuffer
from mat.trainers.llm_trainer_appo import APPOTrainer
from mat.trainers.llm_trainer_tppo import TPPOTrainer
from mat.trainers.llm_trainer_grpo import GRPOTrainer

# 将 PyTorch 张量转换成 Numpy（GPU->CPU）
def _t2n(x):
    return x.detach().cpu().numpy()

class MathRunner:
    def __init__(self, config):
        self.num_agents = config['num_agents']                              # 智能体数量
        self.all_args = config['all_args']                                  # 所有超参数和配置的对象
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 评估回合的线程数
        self.num_env_steps = self.all_args.num_env_steps                    # 训练过程中环境的总步数
        self.episode_length = self.all_args.episode_length                  # 每个回合的最大步数
        self.n_rollout_threads = self.all_args.n_rollout_threads            # 训练回合的线程数
        self.log_interval = self.all_args.log_interval                      # 日志
        self.eval_interval = self.all_args.eval_interval                    # 评估保存的时间间隔
        self.save_interval = self.all_args.save_interval                    # 模型保存的时间间隔
        self.algo = self.all_args.algorithm_name                            # 使用的算法
        self.prm_type = self.all_args.prm_type                              # prm类型

        self.run_dir = config["run_dir"] # 实验根目录
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models/') # 模型保存的目录
        #self.save_dir = str('/mnt/data101_d2/wangzhu/llm_models/train/models/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config['envs'] # 训练环境
        self.eval_envs = config['eval_envs'] # 评估环境
        self.agent = QwenLoRAgent(self.all_args.model_name_or_path, self.all_args.max_new_tokens, self.algo) # 智能体对象（模型名称、最大新生成token数量、算法类型）
        self.buffer = LanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id) # 经验缓冲区（sample重用）
        
        # prm模型初始化
        if self.prm_type == "MS":
            self.prm = MSProcessRM(self.all_args)
        elif self.prm_type == "Qwen":
            self.prm = QwenProcessRM(self.all_args)
        else:
            raise NotImplementedError

        # 训练算法初始化
        if self.algo == "APPO":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "TPPO":
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "GRPO":
            self.trainer = GRPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError

    def run(self):
        obs = self.envs.reset() # 重置环境
        self.buffer.obs[0] = obs.copy() # 经验回放缓冲区 观测值初始化

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads # 总训练回合数
        
        episodic_returns = [] # 每个回合的累计奖励
        for episode in range(episodes):
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads  # 训练总步数
            for step in range(self.episode_length):
                # Sample actions（状态值、策略采样的动作、动作的标记、动作的对数概率（策略梯度计算））
                values, actions, action_tokens, log_probs = self.collect(step)
                
                # output rewards
                rewards = self.prm.get_reward(obs, actions)

                # Obs reward and next obs
                # 环境更新
                obs, fake_rewards, dones, infos = self.envs.step(actions)

                # insert data into buffer（新的观测、奖励、是否完成、状态值、动作、动作标记和动作概率）
                data = obs, rewards, dones, values, actions, action_tokens, log_probs
                self.insert(data)
                
                for i in range(self.n_rollout_threads):
                    if dones[i, 0]:
                        episodic_returns.append(rewards[i, 0])

            # compute return and update network
            self.before_update() # 预处理
            train_infos = self.trainer.train(self.buffer)  # 训练  
            self.buffer.after_update()
            
            # save model
            if (episode == episodes - 1 or episode % self.save_interval == 0):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                print("average_step_rewards: ", np.mean(self.buffer.rewards))
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                train_infos["average_currect_rate"] = np.mean(episodic_returns)
                self.log_infos(train_infos, total_num_steps)
                episodic_returns = []

            # eval
            # if self.all_args.use_eval and episode % self.eval_interval == 0:
            #     self.eval(total_num_steps)
        

    # 收集多个并行环境中的智能体行为数据
    @torch.no_grad() # 不计算梯度（节省内存和计算资源）
    def collect(self, step):
        behaviour_data = self.agent.infer_for_rollout(np.concatenate(self.buffer.obs[step])) # 策略优化方法 
        # 结构：behaviour_data = agent方法（拼接成一个数组（包含当前时间步的所有观察值，即状态数据））
        
        actions, action_tokens, values, log_probs = behaviour_data
        
        # [self.envs, agents] 按并行环境线程数进行拆分
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs

    # 将数据插入到一个缓冲区中
    def insert(self,data):
        obs, rewards, dones, values, actions, action_tokens, log_probs = data

        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32) 
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32) # 回合完成

        if self.algo == "APPO" or self.algo == "GRPO":
            self.buffer.insert_appo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        elif self.algo == "TPPO":
            self.buffer.insert_tppo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        else:
            raise NotImplementedError

    # 更新值函数
    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[-1])) # 根据不同策略优化方法获取value
        next_values = np.array(np.split(next_values, self.n_rollout_threads))
        if self.algo == "APPO":
            self.buffer.batch_process_appo(next_values)
        elif self.algo == "TPPO":
            self.buffer.batch_process_tppo(next_values)
        elif self.algo == "GRPO":
            self.buffer.batch_process_grpo()
        else:
            raise NotImplementedError

    def log_infos(self, infos, total_num_steps):
        for k, v in infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episodic_returns = []
        eval_obs = self.eval_envs.reset() # 重置评估环境

        while True:
            eval_actions, _ = self.agent.get_actions(np.concatenate(eval_obs)) # 获取智能体当前状态下的动作
            eval_actions = np.array(np.split(eval_actions, self.n_eval_rollout_threads)) # 按线程分割
            eval_obs, eval_rewards, eval_dones, _ = self.eval_envs.step(eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones[eval_i, 0]:
                    eval_episode += 1
                    eval_episodic_returns.append(eval_rewards[eval_i]) # 将该回合奖励加入回报

            if eval_episode >= self.all_args.eval_episodes:
                eval_currect_rate = np.mean(eval_episodic_returns) # 平均回报
                env_infos = {'eval_currect_rate': eval_currect_rate}     
                print("total_num_steps: ", total_num_steps)
                print("eval_currect_rate is {}.".format(eval_currect_rate))           
                self.log_infos(env_infos, total_num_steps)
                break
                
    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.agent.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.agent.restore(model_dir)


