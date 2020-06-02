import numpy as numpy

import gym as openai_gym

from Frozen_Lake_Q_Learning.q_learning_agent import Agent

import matplotlib.pyplot as pyplot


if __name__ == "__main__":

    new_game_environment = openai_gym.make('FrozenLake-v0')

    # Number of Actions -> Left, Right, Up, Down -> 4 Actions
    # This is also a 4x4 grid, so there is 16 possible states...
    # Learning Rate: is > 10^(-6) and < 1.0
    # Typical Learning Rate is 0.001

    new_agent = Agent(learning_rate_alpha=0.0001, number_of_actions=4, number_of_states=16,
                      discount_factor_gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decrement=1e-5)

number_games = 50000

win_percentage = []

win_percentage_list = []

scores = []

new_game_environment.reset()

# Start with 500,000 episodes.
for episode in range(number_games):

    # Reset Environment
    done = False

    # Old State
    observation = new_game_environment.reset()

    # Set score for episode to 0
    score = 0

    # While the Episode is not done
    i = 0
    while not done:

        action = new_agent.choose_actions_based_on_current_state(observation)

        # observation_prime = New State
        observation_prime, reward, done, info = new_game_environment.step(action)

        new_agent.learn_given_input(old_state=observation, action=action, reward=reward, new_state=observation_prime)

        score += reward

        # Old State = New State
        observation = observation_prime

    scores.append(score)

    if episode % 100 == 0:

        win_percentage = numpy.mean(scores[-100:])

        win_percentage_list.append(win_percentage)

        print("Episode: ", episode, end=" ")

        print("Win Percentage: ", win_percentage, end=" ")

        print("Epsilon: %.2f", new_agent.epsilon_start, end="\n")


pyplot.plot(win_percentage_list)

pyplot.show()

new_game_environment.close()
