import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from agent import DQNAgent
from memory import ReplayMemory

ENV_NAME = "ALE/KungFuMaster-v5"
NUM_EPISODES = 1000
MAX_MEMORY = 50000
STACK_SIZE = 4
FRAME_WIDTH, FRAME_HEIGHT = 84, 84

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
    return resized / 255.0

def stack_frames(stacked_frames, new_frame, is_new_episode):
    frame = preprocess_frame(new_frame)
    if is_new_episode:
        stacked_frames = deque([frame] * STACK_SIZE, maxlen=STACK_SIZE)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=2), stacked_frames

def main():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs, _ = env.reset()
    stacked_frames = deque([np.zeros((FRAME_WIDTH, FRAME_HEIGHT))] * STACK_SIZE, maxlen=STACK_SIZE)

    state, stacked_frames = stack_frames(stacked_frames, obs, True)
    state_shape = (FRAME_WIDTH, FRAME_HEIGHT, STACK_SIZE)
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size)
    memory = ReplayMemory(MAX_MEMORY)

    step = 0

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, obs, True)
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_obs, False)

            memory.push(state, action, reward, next_state, done)
            agent.train(memory)

            state = next_state
            total_reward += reward
            step += 1

            if step % agent.update_target_freq == 0:
                agent.update_target_network()

        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    main()
