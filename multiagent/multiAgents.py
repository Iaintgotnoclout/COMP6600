# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Initialize score with the successor state's score
        score = successorGameState.getScore()

        # Calculate distance to the nearest food
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1.0 / minFoodDistance  # Reciprocal of distance to food

        # Calculate distance to the nearest ghost
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if ghostDistance > 0:
                if ghostDistance < 2:  # If ghost is too close, penalize heavily
                    score -= 10.0 / ghostDistance
                else:
                    score -= 1.0 / ghostDistance  # Reciprocal of distance to ghost

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def minimax(agentIndex, depth, gameState):
            # Base case: if the game is over or the depth limit is reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                value = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # After Pacman's move, it's the first ghost's turn (agent 1)
                    value = max(value, minimax(1, depth, successor))
                return value

            # Ghosts' turn (minimizing agents)
            else:
                value = float('inf')
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():  # All ghosts have moved
                    nextAgent = 0  # Reset to Pacman
                    depth += 1  # Increment depth after all agents have moved
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, minimax(nextAgent, depth, successor))
                return value

        # Start the minimax process from Pacman (agent 0) at depth 0
        bestAction = None
        bestValue = -float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # After Pacman's move, it's the first ghost's turn (agent 1)
            value = minimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        with alpha-beta pruning.
        """
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: if the game is over or the depth limit is reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                value = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(1, depth, successor, alpha, beta))
                    if value > beta:  # Prune if value exceeds beta
                        return value
                    alpha = max(alpha, value)
                return value

            # Ghosts' turn (minimizing agents)
            else:
                value = float('inf')
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():  # All ghosts have moved
                    nextAgent = 0  # Reset to Pacman
                    depth += 1  # Increment depth after all agents have moved
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(nextAgent, depth, successor, alpha, beta))
                    if value < alpha:  # Prune if value is less than alpha
                        return value
                    beta = min(beta, value)
                return value

        # Start the alpha-beta process from Pacman (agent 0) at depth 0
        bestAction = None
        bestValue = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(1, 0, successor, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction.

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(agentIndex, depth, gameState):
            # Base case: if the game is over or the depth limit is reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                value = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(1, depth, successor))
                return value

            # Ghosts' turn (expectation layer)
            else:
                value = 0
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():  # All ghosts have moved
                    nextAgent = 0  # Reset to Pacman
                    depth += 1  # Increment depth after all agents have moved
                legalActions = gameState.getLegalActions(agentIndex)
                probability = 1.0 / len(legalActions)  # Uniform probability
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value += probability * expectimax(nextAgent, depth, successor)
                return value

        # Start the expectimax process from Pacman (agent 0) at depth 0
        bestAction = None
        bestValue = -float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function considers the following features:
    - Current game score.
    - Distance to the nearest food pellet (reciprocal).
    - Distance to the nearest ghost (penalize if too close).
    - Number of remaining food pellets (penalize for more food).
    - Distance to the nearest capsule (encourage eating capsules).
    - Ghost scared timer (encourage hunting scared ghosts).
    """
    # Extract useful information from the current game state
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsulePositions = currentGameState.getCapsules()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Initialize score with the current game score
    score = currentGameState.getScore()

    # Calculate distance to the nearest food pellet
    foodList = foodGrid.asList()
    if foodList:
        minFoodDistance = min([manhattanDistance(pacmanPosition, food) for food in foodList])
        score += 2.0 / (minFoodDistance + 1)  # Reciprocal of distance to food (add 1 to avoid division by zero)

    # Calculate distance to the nearest ghost
    for i, ghostState in enumerate(ghostStates):
        ghostPosition = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPosition, ghostPosition)
        if scaredTimes[i] > 0:  # Ghost is scared
            if ghostDistance > 0:
                score += 20.0 / (ghostDistance + 1)  # Encourage hunting scared ghosts
        else:  # Ghost is not scared
            if ghostDistance > 0:
                if ghostDistance < 2:  # If ghost is too close, penalize heavily
                    score -= 50.0 / (ghostDistance + 1)
                else:
                    score -= 2.0 / (ghostDistance + 1)  # Reciprocal of distance to ghost

    # Penalize for the number of remaining food pellets
    score -= 5 * len(foodList)  # Encourage eating food

    # Calculate distance to the nearest capsule
    if capsulePositions:
        minCapsuleDistance = min([manhattanDistance(pacmanPosition, capsule) for capsule in capsulePositions])
        score += 10.0 / (minCapsuleDistance + 1)  # Encourage eating capsules

    # Penalize for the number of remaining capsules
    score -= 10 * len(capsulePositions)  # Encourage eating capsules

    return score

# Abbreviation
better = betterEvaluationFunction