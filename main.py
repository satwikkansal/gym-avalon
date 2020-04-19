from environment import AvalonEnv
from agent import QTableAgent, RandomAgent


def sample_run(env, agent):
    """
    Runs a single episode of the game.
    """

    # Initial
    obs = env.reset()
    done = False
    info = {}
    rewards = None
    while not done:
        print("Observation", obs)
        print("Reward", rewards)
        print("Done status", done)
        print("Info", info)
        action = agent.get_next_action(obs, rewards, info)
        obs, rewards, done, info = env.step(action)
        env.render()


if __name__ == "__main__":
    env = AvalonEnv(5, True)
    agent = QTableAgent(env=env)
    sample_run(env, agent)
