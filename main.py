from environment import AvalonEnv
from agent import RandomAgent


def sample_run():
    env = AvalonEnv(5)

    # Initial
    obs = env.reset()
    action_space = env.action_space
    agent = RandomAgent()
    done = False
    info = {}
    rewards = None
    while not done:
        print("Observation", obs)
        print("Reward", rewards)
        print("Done status", done)
        print("Info", info)
        action = agent.get_next_action(obs, rewards, info, action_space)
        obs, rewards, done, info = env.step(action)
        env.render()

sample_run()

