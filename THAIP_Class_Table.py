#-------------------------------------------------------------------------------
# Name:        Texas Hold'em AI Program: Table Class
# Purpose:     A class containing attributes of an active Texas Hold'em table
#              and methods corrosponding to the actions of a dealer.
#
# Author:      Joseph Carrick
#
# Created:     06/09/2015
#-------------------------------------------------------------------------------

import random
import math
from THAIP_Class_Card import Card


class Table:
    """A class representing an active heads-up Texas Hold'em tournament table. Keeps track of
    rounds, turns, each players' cards, the community cards, each players'
    chip stack, the pot stack, the last player's action, the blind or ante
    level, and the deck used in play. """


    def __init__(self,AI_Stack,Player_Stack,SB):
        """Initializes a Table object."""

        self.AI_Stack = AI_Stack                            # The starting chip count for the AI must be inputed.

        self.Player_Stack = Player_Stack                    # The starting chip count for the human player must be inputed.

        self.Winner = 0                                     # Indicates the winner of the last hand: 1 = human player, 2 = AI player, 0 = tie/none.

        self.SB = SB                                        # The small blind must be inputed.

        self.BB = 2*self.SB                                 # The big blind is always twice the small blind.

        self.Round = 1                                      # The game always starts at round 1.

        self.Dealer_Button = random.randint(1,2)            # The inital player on the button is random.

        self.Action_To = self.Dealer_Button                 # The dealer is first to act in heads-up.

        self.Community_Cards = []                           # Initially, 0 of the 5 community cards are on the table.

        self.Player_Cards = []                              # Initially, 0 of the 2 Player_Cards are on the table.

        self.AI_Cards = []                                  # Initially, 0 of the 2 AI_Cards are on the table.

        self.Best_Cards = []                                # Best_Cards contain the best possible cards determined by
                                                            # Determine Hand methods. The player with these cards is
                                                            # determined in the game loop.

        self.Player_Best_Cards = []                         # This list will contain the human player's best hand on the river.

        self.AI_Best_Cards = []                             # This list will contain the AI's best hand on the river.

        self.Pot_Stack = 0                                  # There shouldn't be any chips in the pot before the cards have been dealt
                                                            # and the player on the button performs his/her action.

        self.lastAction = [0,"",0]                         # Keeps track of the last action played.
                                                            # Format: [PLAYER (type int: 1 for human player, 2 for AI, 0 for no action),
                                                            # ACTION (type string: "Check","Call","Raise", or "Fold"),
                                                            # AMOUNT (type int: 0 for check, > 0 for call, > 0 for raise, 0 for fold)]

        self.RaisePot = 0                                   # RaisePot keeps track of the bets before they go to the pot.

        self.New_Hand = False                               # There are many ways a hand can end. New_Hand is a boolean at the end of every possible end to each hand. When a hand ends, New_Hand is set to True.

        self.Deck = [Card(2,"Hearts",1),Card(3,"Hearts",2),Card(4,"Hearts",3),Card(5,"Hearts",4),
                    Card(6,"Hearts",5),Card(7,"Hearts",6),Card(8,"Hearts",7),Card(9,"Hearts",8),
                    Card(10,"Hearts",9),Card(11,"Hearts",10),Card(12,"Hearts",11),Card(13,"Hearts",12),
                    Card(14,"Hearts",13),Card(2,"Diamonds",14),Card(3,"Diamonds",15),Card(4,"Diamonds",16),
                    Card(5,"Diamonds",17),Card(6,"Diamonds",18),Card(7,"Diamonds",19),Card(8,"Diamonds",20),
                    Card(9,"Diamonds",21),Card(10,"Diamonds",22),Card(11,"Diamonds",23),Card(12,"Diamonds",24),
                    Card(13,"Diamonds",25),Card(14,"Diamonds",26),Card(2,"Clubs",27),Card(3,"Clubs",28),
                    Card(4,"Clubs",29),Card(5,"Clubs",30),Card(6,"Clubs",31),Card(7,"Clubs",32),
                    Card(8,"Clubs",33),Card(9,"Clubs",34),Card(10,"Clubs",35),Card(11,"Clubs",36),
                    Card(12,"Clubs",37),Card(13,"Clubs",38),Card(14,"Clubs",39),Card(2,"Spades",40),
                    Card(3,"Spades",41),Card(4,"Spades",42),Card(5,"Spades",43),Card(6,"Spades",44),
                    Card(7,"Spades",45),Card(8,"Spades",46),Card(9,"Spades",47),Card(10,"Spades",48),
                    Card(11,"Spades",49),Card(12,"Spades",50),Card(13,"Spades",51),Card(14,"Spades",52)]

#------------------------Game Mechanics Methods------------------------------#

    def Shuffle_Deck(self):
        """ Randomizes self.Deck. Returns nothing """
        shuffled_deck = random.shuffle(self.Deck)
        return self.Deck


    def Deal_Players(self):
        """ Removes four Card objects from the deck and puts two in
            each players' card list- AI_Cards & Player_Cards """
        for i in range(2):
            self.Player_Cards.append(self.Deck.pop())
            self.AI_Cards.append(self.Deck.pop())


    def Deal_Flop(self):
        """ Removes three Card objects from the deck and puts them all
            into the Community_Cards list. """
        for i in range(3):
            self.Community_Cards.append(self.Deck.pop())


    def Deal_Turn(self):
        """ Removes one Card object from the deck and puts it
            into the Community_Cards list. """
        self.Community_Cards.append(self.Deck.pop())


    def Deal_River(self):
        """ Removes one Card object from the deck and puts it
            into the Community_Cards list. """
        self.Community_Cards.append(self.Deck.pop())

