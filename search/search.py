# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    from util import Stack

    # Initialize a stack to manage the nodes to explore
    stack = Stack()
    # Push the starting state and an empty list of actions to the stack
    stack.push((problem.getStartState(), []))  # (state, actions)

    # Initialize a set to keep track of visited states
    visited = set()

    # Loop until the stack is empty
    while not stack.isEmpty():
        # Pop the current state and actions from the stack
        current_state, actions = stack.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            # If it is, return the list of actions to reach this state
            return actions

        # If the state hasn't been visited, explore it
        if current_state not in visited:
            # Mark the state as visited
            visited.add(current_state)

            # Get the successors of the current state
            for successor, action, _ in problem.getSuccessors(current_state):
                # If the successor hasn't been visited, add it to the stack
                if successor not in visited:
                    # Push the successor state and the updated action list to the stack
                    stack.push((successor, actions + [action]))

    # If no solution is found, return an empty list
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the shallowest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    from util import Queue

    # Initialize a queue to manage the nodes to explore
    queue = Queue()
    # Push the starting state and an empty list of actions to the queue
    queue.push((problem.getStartState(), []))  # (state, actions)

    # Initialize a set to keep track of visited states
    visited = set()

    # Loop until the queue is empty
    while not queue.isEmpty():
        # Pop the current state and actions from the queue
        current_state, actions = queue.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            # If it is, return the list of actions to reach this state
            return actions

        # If the state hasn't been visited, explore it
        if current_state not in visited:
            # Mark the state as visited
            visited.add(current_state)

            # Get the successors of the current state
            for successor, action, _ in problem.getSuccessors(current_state):
                # If the successor hasn't been visited, add it to the queue
                if successor not in visited:
                    # Push the successor state and the updated action list to the queue
                    queue.push((successor, actions + [action]))

    # If no solution is found, return an empty list
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the node of least total cost first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    from util import PriorityQueue

    # Initialize a priority queue to manage the nodes to explore
    priority_queue = PriorityQueue()
    # Push the starting state, an empty list of actions, and a cost of 0 to the priority queue
    priority_queue.push((problem.getStartState(), []), 0)  # (state, actions), priority

    # Initialize a set to keep track of visited states
    visited = set()

    # Loop until the priority queue is empty
    while not priority_queue.isEmpty():
        # Pop the current state, actions, and cost from the priority queue
        current_state, actions = priority_queue.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            # If it is, return the list of actions to reach this state
            return actions

        # If the state hasn't been visited, explore it
        if current_state not in visited:
            # Mark the state as visited
            visited.add(current_state)

            # Get the successors of the current state
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # If the successor hasn't been visited, add it to the priority queue
                if successor not in visited:
                    # Calculate the total cost to reach the successor
                    total_cost = problem.getCostOfActions(actions + [action])
                    # Push the successor state, updated action list, and total cost to the priority queue
                    priority_queue.push((successor, actions + [action]), total_cost)

    # If no solution is found, return an empty list
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """
    Search the node that has the lowest combined cost and heuristic first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    from util import PriorityQueue

    # Initialize a priority queue to manage the nodes to explore
    priority_queue = PriorityQueue()
    # Push the starting state, an empty list of actions, and a priority of 0 to the priority queue
    priority_queue.push((problem.getStartState(), []), 0)  # (state, actions), priority

    # Initialize a dictionary to keep track of the best cost to reach each state
    cost_so_far = {}
    cost_so_far[problem.getStartState()] = 0

    # Loop until the priority queue is empty
    while not priority_queue.isEmpty():
        # Pop the current state and actions from the priority queue
        current_state, actions = priority_queue.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            # If it is, return the list of actions to reach this state
            return actions

        # Get the successors of the current state
        for successor, action, step_cost in problem.getSuccessors(current_state):
            # Calculate the new cost to reach the successor
            new_cost = cost_so_far[current_state] + step_cost

            # If the successor hasn't been visited or the new cost is better than the previous cost
            if successor not in cost_so_far or new_cost < cost_so_far[successor]:
                # Update the cost to reach the successor
                cost_so_far[successor] = new_cost
                # Calculate the priority (f(n) = g(n) + h(n))
                priority = new_cost + heuristic(successor, problem)
                # Push the successor state, updated action list, and priority to the priority queue
                priority_queue.push((successor, actions + [action]), priority)

    # If no solution is found, return an empty list
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch