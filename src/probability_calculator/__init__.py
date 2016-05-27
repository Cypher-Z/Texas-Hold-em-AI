# Copyright (c) 2013 Kevin Tseng
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#
# Following codes are modified based on codes from
# https://github.com/ktseng/holdem_calc/

import multiprocessing
import holdem_functions

Card = holdem_functions.Card


def evaluate(hole_cards, adversary_hole_cards, given_board=None,
             is_parallel=True, num_iterations=1000000):
  """Evaluate the winning probability given player's cards and
   board cards.

  :param hole_cards: a list of string like ["As", "Td"]
  :param adversary_hole_cards: a list of string like ["As", "Td"]
  :param given_board:  a list of string like ["Ac", "Kd", "3c"]
  :param num_iterations: int
  :return: two floats, player's winning probability and his adversary's
  """
  hole_cards = tuple([Card(x) for x in hole_cards])
  adversary_hole_cards = tuple([Card(x) for x in adversary_hole_cards])
  all_cards = [hole_cards, adversary_hole_cards]
  hole_cards = (hole_cards, adversary_hole_cards)
  if given_board is not None:
    given_board = [Card(x) for x in given_board]
    all_cards.append(given_board)
  deck = holdem_functions.generate_deck(all_cards)

  if is_parallel:
    return _parallel_evaluate(
      hole_cards, deck,
      given_board=given_board, num_iterations=num_iterations
    )
  else:
    return _evaluate(
      hole_cards, deck,
      given_board=given_board, num_iterations=num_iterations
    )


def _evaluate(hole_cards, deck, given_board, num_iterations):
  num_players = len(hole_cards)
  result_list, winner_list = [None] * num_players, [0] * (num_players + 1)
  result_histograms = []
  for player in xrange(num_players):
    result_histograms.append([0] * 10)

  board_length = 0 if given_board is None else len(given_board)
  if given_board is not None:
    generate_boards = holdem_functions.generate_exhaustive_boards
  else:
    generate_boards = holdem_functions.generate_random_boards

  # Run simulations
  for remaining_board in generate_boards(deck, num_iterations, board_length):
    # Generate a new board
    if given_board:
      board = given_board[:]
      board.extend(remaining_board)
    else:
      board = remaining_board
    # Find the best possible poker hand given the created board and the
    # hole cards and save them in the results data structures
    (suit_histogram, histogram, max_suit) = \
      holdem_functions.preprocess_board(board)
    for index, hole_card in enumerate(hole_cards):
      result_list[index] = holdem_functions.detect_hand(
        hole_card, board, suit_histogram, histogram, max_suit
      )
    # Find the winner of the hand and tabulate results
    winner_index = holdem_functions.compare_hands(result_list)
    winner_list[winner_index] += 1
    # Increment what hand each player made
    for index, result in enumerate(result_list):
      result_histograms[index][result[0]] += 1

  float_iterations = float(sum(winner_list))
  return winner_list[1] / float_iterations, winner_list[2] / float_iterations


def _simulation(remaining_board):
  # Extract variables shared through inheritance
  given_board, hole_cards = _simulation.given_board, _simulation.hole_cards
  num_players = _simulation.num_players
  winner_list = _simulation.winner_list
  result_histograms = _simulation.result_histograms
  # Generate a new board
  if given_board:
    board = given_board[:]
    board.extend(remaining_board)
  else:
    board = remaining_board
  # Extract process id from the name of the current process
  # Names are of the format: PoolWorker-1 - PoolWorker-n
  proc_id = int(multiprocessing.current_process().name[-1]) - 1
  # Create results data structure which tracks results of comparisons
  result_list = []
  for player in xrange(num_players):
    result_list.append([])
  # Find the best possible poker hand given the created board and the
  # hole cards and save them in the results data structures
  (suit_histogram,
   histogram, max_suit) = holdem_functions.preprocess_board(board)
  for index, hole_card in enumerate(hole_cards):
    result_list[index] = holdem_functions.detect_hand(
      hole_card, board,
      suit_histogram, histogram, max_suit
    )
  # Find the winner of the hand and tabulate results
  winner_index = holdem_functions.compare_hands(result_list)
  winner_list[proc_id * (num_players + 1) + winner_index] += 1
  # Increment what hand each player made
  for index, result in enumerate(result_list):
    result_histograms[10 * (proc_id * num_players + index)
                      + result[0]] += 1


def _simulation_init(given_board, hole_cards, winner_list,
                     result_histograms, num_players):
  _simulation.given_board = given_board
  _simulation.hole_cards = hole_cards
  _simulation.winner_list = winner_list
  _simulation.result_histograms = result_histograms
  _simulation.num_players = num_players


def _parallel_evaluate(hole_cards, deck, given_board=None, num_iterations=1000):
  num_players = len(hole_cards)
  # Create data structures to manage multiple processes:
  # 1) winner_list: number of times each player wins a hand
  # 2) result_histograms: a list for each player that shows the number of
  #    times each type of poker hand (e.g. flush, straight) was gotten
  num_processes = multiprocessing.cpu_count()
  winner_list = multiprocessing.Array('i', num_processes * (num_players + 1))
  result_histograms = multiprocessing.Array('i',
                                            num_processes * num_players * 10)
  # Choose whether we're running a Monte Carlo or exhaustive simulation
  board_length = 0 if given_board == None else len(given_board)
  # When a board is given, exact calculation is much faster than Monte Carlo
  # simulation, so default to exact if a board is given
  if given_board is not None:
    generate_boards = holdem_functions.generate_exhaustive_boards
  else:
    generate_boards = holdem_functions.generate_random_boards
  # Create threadpool and use it to perform hand detection over all boards
  pool = multiprocessing.Pool(processes=num_processes,
                              initializer=_simulation_init,
                              initargs=(given_board, hole_cards, winner_list,
                                        result_histograms, num_players))
  pool.map(_simulation, generate_boards(deck, num_iterations, board_length))
  # Tallying and printing results
  combined_winner_list, combined_histograms = [0] * (num_players + 1), []
  for player in xrange(num_players):
    combined_histograms.append([0] * 10)
  # Go through each parallel data structure and aggregate results
  for index, element in enumerate(winner_list):
    combined_winner_list[index % (num_players + 1)] += element
  for index, element in enumerate(result_histograms):
    combined_histograms[(index // 10) % num_players][index % 10] += element
  # Print results
  holdem_functions.print_results(hole_cards, combined_winner_list,
                                 combined_histograms)

  float_iterations = float(sum(winner_list))
  return winner_list[1] / float_iterations, winner_list[2] / float_iterations
