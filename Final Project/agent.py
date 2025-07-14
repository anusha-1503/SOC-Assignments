import numpy as np
import tensorflow as tf
from helper import KungFu

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_freq = 1000
        self.learn_rate = 0.00025

        self.model = KungFu(self.action_size, self.learn_rate)
        self.target_model = KungFu(self.action_size, self.learn_rate)
        self.update_target_network()

        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def train(self, memory):
        if len(memory) < self.batch_size:
            return

        minibatch = memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        next_qs = self.target_model.predict(next_states, verbose=0)
        target_qs = rewards + self.gamma * np.max(next_qs, axis=1) * (1 - dones)

        masks = tf.one_hot(actions, self.action_size)

        with tf.GradientTape() as tape:
            qs = self.model(states)
            q_action = tf.reduce_sum(masks * qs, axis=1)
            loss = self.loss_fn(target_qs, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
