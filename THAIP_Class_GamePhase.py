#-------------------------------------------------------------------------------
# Name:        Texas Hold'em AI Program: GamePhase Class
# Purpose:     A class to store the game phase states of a Texas Hold'em game.
#
# Author:      Joseph Carrick
#
# Created:     15/09/2015
#-------------------------------------------------------------------------------

class GamePhase:

    def __init__(self):
        self.isPreFlop     = True
        self.isFlop        = False
        self.isTurn        = False
        self.isRiver       = False
        self.isEnd_of_Hand = False
        self.isGame_Over   = False
        self.isAllIn       = False

    def allToFalse(self):
        """ Makes every game phase boolean false.
            Excludes isAllIn and isGame_Over. """
        self.isPreFlop     = False
        self.isFlop        = False
        self.isTurn        = False
        self.isRiver       = False
        self.isEnd_of_Hand = False
