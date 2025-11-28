import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# Handle gym/gymnasium compatibility
try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    import gym
    GYMNASIUM_AVAILABLE = False
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, TrainFrequencyUnit, RolloutReturn 
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3.td3.policies import TD3Policy, CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from stable_baselines3.common.preprocessing import get_action_dim
from .controllers import RL_withController

SelfTD3 = TypeVar("SelfTD3", bound="TD3")

def create_mlp_adj(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[th.nn.Module] = th.nn.ReLU,
    squash_output: bool = False,
) -> List[th.nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [th.nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(th.nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(th.nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        # modules.append(th.nn.Tanh())
        modules.append(th.nn.Softmax(dim=1))
    return modules


class ActorAdj(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: th.nn.Module,
        features_dim: int,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        # Adjust to consume full compact state as input (no assumption that last action_dim are market weights)
        actor_net = create_mlp_adj(features_dim, action_dim, net_arch, activation_fn, squash_output=True)

        # Deterministic action
        self.mu = th.nn.Sequential(*actor_net)
        # self.mkt_hidden_sm = th.nn.Softmax(dim=1) # Execute softmax on the market observer.

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        # Clean NaN/Inf from observation
        obs = th.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        features = self.extract_features(obs, self.features_extractor)
        # Clean NaN/Inf from features
        features = th.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        td3_decision = self.mu(features) # range [0, 1], sum=1
        td3_decision = th.nan_to_num(td3_decision, nan=0.0, posinf=0.0, neginf=0.0)
        # Map to [-1, 1]
        final_output = (2.0 * td3_decision) - 1.0
        # Final cleanup
        final_output = th.nan_to_num(final_output, nan=0.0, posinf=1.0, neginf=-1.0)
        return final_output

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)


class ActorOriginal(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: th.nn.Module,
        features_dim: int,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        actor_net = create_mlp_adj(features_dim, action_dim, net_arch, activation_fn, squash_output=True)

        # Deterministic action
        self.mu = th.nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        final_output = self.mu(features) # range [0, 1], sum=1
        return final_output

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)

class TD3PolicyAdj(TD3Policy):
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ActorAdj:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorAdj(**actor_kwargs).to(self.device)

class TD3PolicyOriginal(TD3Policy):
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ActorAdj:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorOriginal(**actor_kwargs).to(self.device)

class TD3Controller(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.
    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        'TD3PolicyAdj': TD3PolicyAdj,
        'TD3PolicyOriginal': TD3PolicyOriginal,
    }

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        entropy_coef: float = 0.0,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box,),
            support_multi_env=True,
        )
        self.entropy_coef = entropy_coef
        self._warmup_notice_printed = False

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Log that train() was called (ALWAYS log, not just when verbose)
        n_updates_before = getattr(self, '_n_updates', 0)
        buffer_size = self.replay_buffer.size() if hasattr(self.replay_buffer, 'size') else len(self.replay_buffer)
        print(f"[TRAIN] ⚡ train() method CALLED | _n_updates before: {n_updates_before} | Gradient steps: {gradient_steps} | Buffer size: {buffer_size}", flush=True)

        # Guard against premature training (respect learning_starts even if collect_rollouts misfires)
        learning_starts = getattr(self, 'learning_starts', 0)
        if buffer_size < learning_starts:
            if self.verbose >= 1:
                print(f"[TRAIN] ⏸ Buffer size {buffer_size} < learning_starts {learning_starts}. Skipping gradient update.", flush=True)
            return
        
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        # Log training start (always log first few times, then every 100)
        if self.verbose >= 1 and (self._n_updates < 10 or self._n_updates % 100 == 0):
            print(f"[TRAIN] Starting gradient updates | Total updates: {self._n_updates} | Gradient steps: {gradient_steps} | Buffer size: {buffer_size}", flush=True)

        actor_losses, critic_losses = [], []
        sampled_rewards = []
        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            try:
                sampled_rewards.append(replay_data.rewards.mean().item())
            except Exception:
                pass

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_actions = self.actor(replay_data.observations)
                base_actor_loss = -self.critic.q1_forward(replay_data.observations, actor_actions).mean()
                entropy_term = 0.0
                if getattr(self, "entropy_coef", 0.0) > 0:
                    probs = th.softmax(actor_actions, dim=1)
                    entropy = -(probs * (probs.clamp(min=1e-8)).log()).sum(dim=1).mean()
                    entropy_term = entropy
                    actor_loss = base_actor_loss - self.entropy_coef * entropy
                else:
                    actor_loss = base_actor_loss
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        
        # Log training completion with loss values (always log first few times, then every 100)
        n_updates_after = getattr(self, '_n_updates', 0)
        mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0.0
        mean_critic_loss = np.mean(critic_losses)
        mean_sample_reward = np.mean(sampled_rewards) if len(sampled_rewards) > 0 else 0.0
        config_ref = getattr(self, 'mafia_config', None)
        if config_ref is not None:
            config_ref.last_td3_actor_loss = mean_actor_loss
            config_ref.last_td3_critic_loss = mean_critic_loss
            config_ref.last_td3_mean_reward = mean_sample_reward
            config_ref.last_td3_updates = n_updates_after

        buffer_size = self.replay_buffer.size() if hasattr(self.replay_buffer, 'size') else len(self.replay_buffer)
        # Single-line completion summary (actor/critic losses, reward, buffer, updates)
        print(
            f"[TRAIN] ✅ Updates {n_updates_before}→{n_updates_after} (+{n_updates_after - n_updates_before}) | "
            f"Actor: {mean_actor_loss:.6f} | Critic: {mean_critic_loss:.6f} | "
            f"Sample reward: {mean_sample_reward:.6f} | Buffer: {buffer_size}",
            flush=True,
        )

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        """
        Copy of OffPolicyAlgorithm.learn with the single change that training starts
        as soon as ``num_timesteps >= learning_starts`` (instead of strictly greater).
        This matches our expectation that once the replay buffer reaches the
        warm-up threshold, TD3 should begin updating immediately.
        """
        # Log initial state
        if self.verbose >= 1:
            buffer_size = self.replay_buffer.size() if hasattr(self.replay_buffer, 'size') else len(self.replay_buffer)
            learning_starts = getattr(self, 'learning_starts', 100)
            print(f"[LEARN] Starting training | Buffer size: {buffer_size} | learning_starts: {learning_starts} | Total timesteps: {total_timesteps}", flush=True)
            print(f"[LEARN] train_freq: {self.train_freq} | gradient_steps: {self.gradient_steps}", flush=True)

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            # Only train when replay buffer has reached learning_starts
            buffer_size = self.replay_buffer.size() if hasattr(self.replay_buffer, 'size') else len(self.replay_buffer)
            if buffer_size < self.learning_starts:
                continue

            if self.num_timesteps > 0:
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        status_last_time = time.time()
        status_last_step = self.num_timesteps

        def _finalize_rollout_line() -> None:
            """Clear the live status line so future logs print normally."""
            if getattr(self, "_live_rollout_line_active", False):
                sys.stdout.write("\r\033[2K")
                sys.stdout.flush()
            self._live_rollout_line_active = False

        def _log_rollout(message: str) -> None:
            _finalize_rollout_line()
            print(message, flush=True)

        def _log_rollout_status(timestep: int, episode: int, collected_steps: int) -> None:
            nonlocal status_last_time, status_last_step
            # Throttle status updates to reduce spam
            status_interval = getattr(self, "_rollout_status_interval", 10)
            if collected_steps > 0 and collected_steps % status_interval != 0:
                return

            now = time.time()
            elapsed = max(now - status_last_time, 1e-8)
            delta_steps = max(timestep - status_last_step, 0)
            speed = delta_steps / elapsed
            status_last_time, status_last_step = now, timestep

            buffer_size = replay_buffer.size() if hasattr(replay_buffer, "size") else len(replay_buffer)
            learning_starts = getattr(self, "learning_starts", 0)
            warmup_phase = learning_starts > 0 and buffer_size < learning_starts
            buffer_target = learning_starts if learning_starts > 0 else getattr(replay_buffer, "max_size", None)
            buffer_display = f"{buffer_size}/{buffer_target}" if buffer_target else str(buffer_size)

            label = "[WARM-UP]" if warmup_phase else "[ROLLOUT]"
            message = f"{label} Global step {timestep} | Buffer {buffer_display} | Speed: {speed:.2f} steps/s"
            width = getattr(self, "_rollout_status_width", 0)
            width = max(width, len(message))
            self._rollout_status_width = width
            padded_message = message.ljust(width)
            # Always update on one line to avoid log spam (even when stdout is not a TTY)
            sys.stdout.write(f"\r\033[2K{padded_message}")
            sys.stdout.flush()
            self._live_rollout_line_active = True
        
        # Log rollout start
        if self.verbose >= 1 and self._episode_num % 10 == 0:
            _log_rollout_status(self.num_timesteps, self._episode_num, num_collected_steps)

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)
        
        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Check if we've reached total timesteps - stop collecting to prevent extra steps
            if hasattr(self, '_total_timesteps') and self._total_timesteps is not None:
                if self.num_timesteps >= self._total_timesteps:
                    if self.verbose >= 1:
                        _log_rollout(f"[ROLLOUT] Reached total timesteps: {self.num_timesteps}/{self._total_timesteps}, stopping collection")
                    continue_training = False
                    break
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            # actions: Range-[low, high], shape: [1, num_of_stocks], buffer_actions: [-1, 1] for actor and critic agent training
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            a_rlonly = np.array(actions[0]) # [1, num_of_stocks] -> [num_of_stocks, ]
            a_rl = a_rlonly
            # Unwrap environment to get the actual StockPortfolioEnv (not Monitor wrapper)
            unwrapped_env = env.envs[0]
            if hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env  # Unwrap Monitor
            if hasattr(unwrapped_env, 'unwrapped'):
                unwrapped_env = unwrapped_env.unwrapped
            
            # Clean NaN/Inf from a_rl
            a_rl = np.nan_to_num(a_rl, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check if a_rl is valid
            if np.any(np.isnan(a_rl)) or np.any(np.isinf(a_rl)) or np.sum(np.abs(a_rl)) == 0:
                # Fallback to uniform distribution
                a_rl = np.ones(len(a_rl)) / len(a_rl) * unwrapped_env.bound_flag
            else:
                a_rl = a_rl / np.sum(np.abs(a_rl))

            a_final = RL_withController(a_rl=a_rl, env=unwrapped_env)
            # a_final is already normalized in RL_withController, but double-check
            a_final_sum = np.sum(np.abs(a_final))
            if a_final_sum > 1e-8:
                a_final = a_final / a_final_sum
            else:
                # Fallback: uniform distribution
                a_final = np.ones(len(a_final)) / len(a_final)

            # Rescale and perform action
            a_final = np.array([a_final])
            new_obs, rewards, dones, infos = env.step(a_final)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            if self.verbose >= 1:
                _log_rollout_status(self.num_timesteps, self._episode_num, num_collected_steps)

            # Debug: Log dones to track episode end detection
            dones_has_true = any(dones) if hasattr(dones, '__iter__') else bool(dones)
            if self.verbose >= 1 and dones_has_true:
                _log_rollout(f"[ROLLOUT] Episode end detected! dones={dones}, type={type(dones)}, shape={dones.shape if hasattr(dones, 'shape') else 'no shape'}, num_collected_episodes before={num_collected_episodes}, timesteps={self.num_timesteps}")
                # Log individual done flags
                if hasattr(dones, '__iter__') and not isinstance(dones, (str, bytes)):
                    for i, done_flag in enumerate(dones):
                        _log_rollout(f"[ROLLOUT]   dones[{i}] = {done_flag}")
                else:
                    _log_rollout(f"[ROLLOUT]   dones value = {dones}")

            # Check if we've reached total timesteps - stop immediately to prevent extra steps
            if hasattr(self, '_total_timesteps') and self._total_timesteps is not None:
                if self.num_timesteps >= self._total_timesteps:
                    if self.verbose >= 1:
                        _log_rollout(f"[ROLLOUT] Reached total timesteps: {self.num_timesteps}/{self._total_timesteps}, stopping immediately")
                    _finalize_rollout_line()
                    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                _finalize_rollout_line()
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            
            # Log buffer size periodically
            if self.verbose >= 1 and num_collected_steps % 100 == 0:
                buffer_size = replay_buffer.size() if hasattr(replay_buffer, 'size') else len(replay_buffer)
                learning_starts = getattr(self, 'learning_starts', 100)
                max_buffer_size = getattr(replay_buffer, 'max_size', None)
                if max_buffer_size is not None:
                    _log_rollout(f"[ROLLOUT] Step {num_collected_steps} | Buffer size: {buffer_size}/{max_buffer_size} (threshold: {learning_starts}) | Timesteps: {self.num_timesteps}")
                else:
                    _log_rollout(f"[ROLLOUT] Step {num_collected_steps} | Buffer size: {buffer_size} (threshold: {learning_starts}) | Timesteps: {self.num_timesteps}")

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            episodes_ended = 0
            for idx, done in enumerate(dones):
                if bool(done):  # Convert numpy boolean to Python boolean
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1
                    episodes_ended += 1

                    if self.verbose >= 1:
                        _log_rollout(f"[ROLLOUT] Episode {self._episode_num} completed! num_collected_episodes={num_collected_episodes}, train_freq={train_freq}, done_idx={idx}")

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        # Mirror SB3 behaviour so TensorBoard/built-in logger still receives metrics
                        self.dump_logs()

            # Additional debug: Check if episodes were collected
            if self.verbose >= 1 and episodes_ended > 0:
                _log_rollout(f"[ROLLOUT] Total episodes ended this step: {episodes_ended}, total collected: {num_collected_episodes}")
        callback.on_rollout_end()
        
        # Log rollout completion with buffer info
        buffer_size = replay_buffer.size() if hasattr(replay_buffer, 'size') else len(replay_buffer)
        learning_starts = getattr(self, 'learning_starts', 100)
        warmup_phase = buffer_size < learning_starts
        if self.verbose >= 1:
            if warmup_phase:
                if not getattr(self, "_warmup_notice_printed", False):
                    _log_rollout(f"[ROLLOUT] Warm-up phase | Buffer size: {buffer_size}/{learning_starts} | Training disabled until buffer >= learning_starts")
                    self._warmup_notice_printed = True
            else:
                self._warmup_notice_printed = False
                collected_steps = num_collected_steps * env.num_envs
                should_train_flag = (
                    num_collected_episodes > 0
                    if train_freq.unit == TrainFrequencyUnit.EPISODE
                    else collected_steps >= train_freq.frequency
                )
                rollout_summary = (
                    f"[ROLLOUT] Completed | steps: {collected_steps} | episodes: {num_collected_episodes} | "
                    f"train_freq: {train_freq} | should_train: {should_train_flag} | "
                    f"_n_updates: {getattr(self, '_n_updates', 0)} | "
                    f"buffer: {buffer_size}/{learning_starts}"
                )
                _log_rollout(rollout_summary)
                if not should_train_flag:
                    _log_rollout(f"[ROLLOUT] ❌ No training trigger (insufficient data for {train_freq.unit.name})")

        _finalize_rollout_line()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
        
