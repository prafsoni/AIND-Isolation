"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # raise NotImplementedError
    if game.is_loser(player=player):
        return float("-inf")

    if game.is_winner(player=player):
        return float("inf")
    center = (int(game.width / 2), int(game.height / 2))
    opp = game.get_opponent(player=player)
    own_moves = len(game.get_legal_moves(player=player))
    opp_moves = len(game.get_legal_moves(opp))

    weighted_improved = lambda: float(4 * own_moves - 5 * opp_moves)

    manhattan_distance = lambda x, y=center: abs((y[0] - x[0]) + (y[1] - x[1]))

    own_loc = game.get_player_location(player=player)
    opp_loc = game.get_player_location(opp)

    return 5 * weighted_improved() - 3 * ((manhattan_distance(own_loc, opp_loc) + manhattan_distance(own_loc)) / 2)


class CustomPlayer:
    """Game-playing agent to use in the optional player vs player Isolation
    competition.

    You must at least implement the get_move() method and a search function
    to complete this class, but you may use any of the techniques discussed
    in lecture or elsewhere on the web -- opening books, MCTS, etc.

    **************************************************************************
          THIS CLASS IS OPTIONAL -- IT IS ONLY USED IN THE ISOLATION PvP
        COMPETITION.  IT IS NOT REQUIRED FOR THE ISOLATION PROJECT REVIEW.
    **************************************************************************

    Parameters
    ----------
    data : string
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted.  Note that
        the PvP competition uses more accurate timers that are not cross-
        platform compatible, so a limit of 1ms (vs 10ms for the other classes)
        is generally sufficient.
    """

    def __init__(self, data=None, timeout=1.):
        self.score = custom_score
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        # OPTIONAL: Finish this function!
        self.time_left = time_left

        legal_moves = game.get_legal_moves()

        center = (int(game.width / 2), int(game.height / 2))

        if len(legal_moves) == 0:
            return -1, -1

        if center in legal_moves:
            return center

        best_move = random.choice(legal_moves)

        try:
            for depth in range(game.width * game.height + 1):
                if self.time_left() < self.TIMER_THRESHOLD:
                    return best_move
                _, best_move = self.search(game, depth + 1)

        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration
        return best_move
        # raise NotImplementedError

    def search(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implementation minimax search with alpha-beta pruning.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        score = None
        best_moves = None

        center = (int(game.width / 2), int(game.height / 2))

        moves = game.get_legal_moves()

        if center in moves:
            return float("inf"), center

        if depth == 0 or len(moves) <= 0:
            return self.score(game, self), (-1, -1)

        if maximizing_player:
            score = float("-inf")
            best_moves = [(-1, -1)]
            for move in moves:
                future_game = game.forecast_move(move)
                future_score, _ = self.search(future_game, depth - 1, alpha, beta, not maximizing_player)
                if future_score > score:
                    best_moves = [move]
                elif future_score == score:
                    best_moves.append(move)
                score = max(score, future_score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return score, random.choice(best_moves)
        else:
            score = float("inf")
            best_moves = [(-1, -1)]
            for move in moves:
                future_game = game.forecast_move(move)
                future_score, _ = self.search(future_game, depth - 1, alpha, beta, not maximizing_player)
                if future_score < score:
                    best_moves = [move]
                elif future_score == score:
                    best_moves.append(move)
                score = min(score, future_score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return score, random.choice(best_moves)
