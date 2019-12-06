import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, feature_size, base_depth, input_shape):
        super(FeatureExtractor, self).__init__()
        self._conv_output_shape = None
        self._conv1 = nn.Conv2d(input_shape[0], base_depth, 3, 2)
        self._conv2 = nn.Conv2d(base_depth, 2 * base_depth, 3, 2)
        self._conv3 = nn.Conv2d(2 * base_depth, 4 * base_depth, 5)
        self._feature_size = feature_size
        self._input_shape = input_shape
        self._conv_output_shape = self._get_conv_output_shape()
        self._dense = nn.Linear(np.prod(self._conv_output_shape), self._feature_size)

    def forward(self, img):
        img = self.format_obs(img)
        out = self._conv1(img)
        out = F.relu(out)
        out = self._conv2(out)
        out = F.relu(out)
        out = self._conv3(out)
        out = F.relu(out)
        out = torch.reshape(out, (img.shape[0], -1))
        return self._dense(out)

    def format_obs(self, obs):
        obs = obs / float(255)
        if len(obs.shape) < 3:
            raise ValueError("Observations must have at least 3 dimensions")
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.float()

    def _get_conv_output_shape(self):
        if self._conv_output_shape is not None:
            return self._conv_output_shape
        dummy_input = torch.rand(self._input_shape).unsqueeze(0)
        out = self._conv1(dummy_input)
        out = self._conv2(out)
        out = self._conv3(out)
        return out.shape

class CategoricalNetwork(nn.Module):
    def __init__(self, input_size, num_actions, num_atoms, layer_size=128):
        super(CategoricalNetwork, self).__init__()
        self._input_size = input_size
        self._num_actions = num_actions
        self._num_atoms = num_atoms
        self._layer1 = nn.Linear(input_size, layer_size)
        self._layer2 = nn.Linear(layer_size, layer_size)
        self._out_layer = nn.Linear(layer_size, num_actions * num_atoms)

    def forward(self, tensor):
        out = self._layer1(tensor)
        out = F.relu(out)
        out = self._layer2(out)
        out = F.relu(out)
        out = self._out_layer(out)
        out = out.reshape((tensor.shape[0], self._num_actions, self._num_atoms))
        out = F.softmax(out, dim=-1)
        return out

class CategoricalAgent(nn.Module):
    def __init__(
            self,
            observation_shape,
            action_space,
            N=51,
            v_min=-10,
            v_max=10,
            lr=3e-4,
            gamma=0.99,
            epsilon=0.05,
            epsilon_decay=0.0,
            feature_size=128,
            base_depth=128,
            layer_size=128):
        super(CategoricalAgent, self).__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._observation_shape = observation_shape
        self._action_space = action_space
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay

        self._num_atoms = N
        self._delta_z = (v_max - v_min) / self._num_atoms
        self._values = torch.tensor(
                [v_min + i * self._delta_z for i in range(self._num_atoms)]
        ).to(self._device)

        input_shape = self._observation_shape
        self._num_actions = action_space.n
        self._feature_extractor = FeatureExtractor(feature_size,
                                                   base_depth,
                                                   input_shape).to(self._device).float()
        self._categorical_network = CategoricalNetwork(feature_size,
                                                     self._num_actions,
                                                     N,
                                                     layer_size).to(self._device).float()
        feature_extractor_params = self._feature_extractor.parameters()
        categorical_network_params = self._categorical_network.parameters()
        self._optimizer = torch.optim.Adam(
                list(feature_extractor_params) + list(categorical_network_params),
                lr=lr)

    def forward(self, obs):
        """
        Presently only used for debugging/visualization purposes
        """
        features = self._feature_extractor(obs)
        return self._categorical_network(features)

    def train_agent(self, experience):
        obs, action, reward, next_obs, done = experience

        features = self._feature_extractor(obs)
        value_probs = self._categorical_network(features)

        next_features = self._feature_extractor(next_obs)
        next_value_probs = self._categorical_network(next_features)
        q_values = next_value_probs @ self._values
        optimal_actions = torch.argmax(q_values, dim=-1)
        # TODO: vectorize this part
        with torch.no_grad():
            m = torch.zeros((obs.shape[0], self._num_atoms)).to(self._device)
            for i in range(obs.shape[0]):
                if done[i] == 0:
                    future_value = self._gamma * self._values + reward[i]
                else:
                    future_value = reward[i] * torch.ones_like(self._values).to(self._device)
                tz = torch.clamp(future_value, self._values[0], self._values[-1])
                b = (tz - self._values[0]) / self._delta_z
                for j in range(self._num_atoms):
                    l = torch.floor(b[j]).int()
                    u = torch.ceil(b[j]).int()
                    m[i][l] += next_value_probs[i][optimal_actions[i]][j] * (u - b[j])
                    m[i][u] += next_value_probs[i][optimal_actions[i]][j] * (b[j] - l)
        loss = torch.FloatTensor([0]).to(self._device)
        for i in range(obs.shape[0]):
            loss -= torch.t(m[i]) @ torch.log(value_probs[i][action[i].int()])
        loss = loss.mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        artifacts = {
            'images': obs[0],
            'value_distribution': next_value_probs[0],
            'epsilon': self._epsilon,
        }
        self._epsilon = self._epsilon * (1. - self._epsilon_decay)
        return loss, artifacts

    def action(self, obs, explore=True):
        if explore and (np.random.uniform() < self._epsilon):
            action = self._action_space.sample()
        else:
            with torch.no_grad():
                obs = torch.tensor(obs).to(self._device)
                features = self._feature_extractor(obs)
                value_probs = self._categorical_network(features)
                q_values = value_probs @ self._values
                action = torch.argmax(q_values).squeeze().detach().cpu().numpy()
        return action

    def value_support(self):
        return self._values
