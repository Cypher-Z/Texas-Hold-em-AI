#-------------------------------------------------------------------------------
# Name:        Texas Hold'em AI Program: Card Class
# Purpose:     A class containing the information of a single card, and methods
#              to output that information.
#
# Author:     Joseph Carrick
#
# Created:     06/09/2015
#-------------------------------------------------------------------------------

class Card:

    def __init__(self,NumVal,Suit,ID):
        self.NumVal = NumVal
        self.Suit = Suit
        self.ID = ID

        if self.NumVal == 14:
            self.Ace = True
        else:
            self.Ace = False


    def ReturnAsString(self):
        """ PURPOSE:  The purpose of this method is to print Card object with the suit in a word format.
                      This is more or less just to have the option, but it also may help to avoid the confusion
                      that symbols might cause. """

        if self.NumVal == 11:
            val = "Jack"
        if self.NumVal == 12:
            val = "Queen"
        if self.NumVal == 13:
            val = "King"
        if self.NumVal == 14:
            val = "Ace"
        elif self.NumVal < 11:
            val = str(self.NumVal)
        print str(val + " of " + self.Suit)
        return str(val + " of " + self.Suit)

    def ReturnAsShorthand(self):
        """PURPOSE:  The purpose of this method is to print the Card object in a way that is quick and easy to read,
                     and also to test if the input for NumVal and Suit were inputted correctly.This method should be run
                     on every Card instance in the Deck attribute of the Table class when a Table object is instanciated
                     to insure all of the Card objects were correctly instanciated."""

        if self.NumVal == 10:
            val = "T" # Tens are often represented with a "T" in Hold'em shorthand.
        elif self.NumVal == 11:
            val = "J"
        elif self.NumVal == 12:
            val = "Q"
        elif self.NumVal == 13:
            val = "K"
        elif self.NumVal == 14:
            val = "A"
        elif self.NumVal%1 == 0 and 1 < self.NumVal < 10: # If the value for NumVal is a whole number and is a number within the range of 2 to 9, it is a valid card number value.
            val = str(self.NumVal)
        else:
            print "Incorrect number input for Card object: " + str(self.NumVal) + " of " + str(self.Suit)

        # The shorthand is for testing purposes only so the symbols are arbitrary and carry no external meaning other than easy signifiers of different suits.
        if self.Suit == "Hearts":
            suit = "@"
        elif self.Suit == "Diamonds":
            suit = "#"
        elif self.Suit == "Clubs":
            suit = "%"
        elif self.Suit == "Spades":
            suit = "!"
        else:
            print "Incorrect suit input for Card object: " + str(self.NumVal) + " of " +str(self.Suit)

        print val+suit
        return val+suit




