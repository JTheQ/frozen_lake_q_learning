
import numpy as numpy

import math

class Agent():

    def __init__(self, number_of_actions, number_of_states, learning_rate_alpha, discount_factor_gamma, epsilon_start, epsilon_end):

        self.number_of_actions = number_of_actions

        self.number_of_states = number_of_states

        self.learning_rate_alpha = learning_rate_alpha

        self.discount_factor_gamma = discount_factor_gamma

        self.epsilon_start = epsilon_start

        self.epsilon_end = epsilon_end

        self.q_table = {}

        self.initialize_q_table()

    def initialize_q_table(self):

        for state in range(self.number_of_states):

            for action in range(self.number_of_actions):

                self.q_table[(state, action)] = 0.0

    def choose_actions_based_on_current_state(self, current_state):

        if numpy.random.random() < self.epsilon_start:
            # Pick a random action from actions in number_of_actions
            # Random Action
            # We are using a list comprehension to create a list of integers in the range of number of actions.
            # We created a list with [0, 1, 2, 3, ..., number_of_actions]
            action = numpy.random.choice([i for i in range(self.number_of_actions)])
        else:

            '''
                We have used a list comprehension to create a list of elements corresponding to the action values
                for the given state by looking up the relevant quantities in Q-Table.
            '''
            actions = numpy.array([self.q_table[(current_state, action)] for action in range(self.number_of_actions)])
            # Optimal Action

            # .argmax() will always take the lowest value in a tie
            # (If action 0 and action 1 have the same value in the Q-Table, it will always return action 0).
            # We might want to improve upon this.
            action = numpy.argmax(actions)

        return action

    def decrement_epsilon(self):

        if self.epsilon_start > self.epsilon_end:
            # The numpy.random.random() is just a method to decrement self.epsilon_start
            self.epsilon_start = self.epsilon_start * numpy.log(2.718277777777)

            #print(numpy.log(2.71827777777))

        else:

            self.epsilon_start = self.epsilon_end


    def learn_given_input(self, old_state, action, reward, new_state):

        actions = numpy.array([self.q_table[(new_state, action)] for action in range(self.number_of_actions)])

        optimal_action = numpy.argmax(actions)

        self.q_table[(old_state, action)] += self.learning_rate_alpha * (reward + self.discount_factor_gamma * self.q_table[(new_state, optimal_action)] - self.q_table[(old_state, action)])

        self.decrement_epsilon()
