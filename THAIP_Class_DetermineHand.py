#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      carricktel
#
# Created:     16/09/2015
# Copyright:   (c) carricktel 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from THAIP_Class_Card import Card
class DetermineHand:

    def __init__(self,seven_card_hand):

        self.seven_card_hand = seven_card_hand
        self.best_hand = []

        self.findBestHand()

    def isSameHand(self,first_hand,second_hand):
        """ PURPOSE:  Checks to see if the two given hands are the same hand.
                      This is needed for the findStraightFlush method because
                      it needs to see if any of the straights are the same hand
                      as any of the flushes.
            CONTRACT: isSameHand : 2 lists of 5 Card objects -> boolean """

        c = 0
        for i in first_hand:
            for j in second_hand:
                if i == j:
                    c+=1
        if c == 5:
            return True
        else:
            return False

    def Rank_Cards(self,n,cards):
        """ PURPOSE:  Given a list of Card objects, Rank_Cards determines
                      the top n high cards in the list, excluding multiples, and returns a list
                      of those high cards. This method is used in determining
                      a player's best hand.
            CONTRACT: Rank_Cards : list of Card objects -> list of n Card objects """

        outlist = []
        m = [] # list of multiples in cards list.
        for i in cards:
            m = []
            for j in cards:
                if i > j and i.NumVal == j.NumVal:
                    m.append(i)
        for i in m:
                cards.remove(i)
        for i in cards:
            c = 0
            for j in cards:
                if i.NumVal > j.NumVal:
                    c += 1
            if c > (len(cards)-1-n):
                outlist.append(i)
        return outlist


    def Possible_Hands(self):
        """ PURPOSE:  returns a list of lists containing all possible 5 card hands out of a
                      list of 7 cards. The other 'Determine Hand' methods will call this
                      method to find the player's best 5 card hand out of all of the possible
                      hands returned here.
            CONTRACT: Possible_Hands : list of 7 Card objects -> list of 5 Card objects """

        #This series of for-loops yields every 5 card combination out of the 7 possible cards.
        for i in self.seven_card_hand:
            for j in self.seven_card_hand:
                if j > i:
                    for k in self.seven_card_hand:
                        if k > j:
                            for l in self.seven_card_hand:
                                if l > k:
                                    for m in self.seven_card_hand:
                                        if m > l:
                                            yield [i,j,k,l,m]


    def findStraightFlush(self,player_cards):
        """ PURPOSE:  Checks to see if any of the straights are also flushes. If it finds any
                      it appends them to self.straightflushes.
            CONTRACT: findStraightFlush : list of 5 Card objects -> boolean """

        isstraight = False
        for i in player_cards:

            # Look for a straight with the Ace number value being 1.
            if i.Ace == True:
                for j in player_cards:
                    if j.NumVal == 2 and j.Suit == i.Suit:
                        for k in player_cards:
                            if k.NumVal == 3 and k.Suit == i.Suit:
                                for l in player_cards:
                                    if l.NumVal == 4 and l.Suit == i.Suit:
                                        for m in player_cards:
                                            if m.NumVal == 5 and m.Suit == i.Suit:
##                                                self.isStraight = True
##                                                self.straights.append([i,j,k,l,m])
                                                isstraight = True

            # Look for a straight with the Ace number value being 14.
            for j in player_cards:
                if j.NumVal == i.NumVal+1 and j.Suit == i.Suit:
                    for k in player_cards:
                        if k.NumVal == j.NumVal+1 and k.Suit == i.Suit:
                            for l in player_cards:
                                if l.NumVal == k.NumVal+1 and l.Suit == i.Suit:
                                    for m in player_cards:
                                        if m.NumVal == l.NumVal+1 and m.Suit == i.Suit:
                                            isstraight = True
        if isstraight == True:
            return True
        return False


    def findQuads(self,player_cards):
        """ PURPOSE:  Checks to see if there are quads in the hand of five cards given.
                      if there is, it appends the quad to self.quads.
            CONTRACT: findQuads : list of 5 Card objects -> boolean """

        for i in player_cards:
            for j in player_cards:
                if j > i and i.NumVal == j.NumVal:
                    for k in player_cards:
                        if k > j and i.NumVal == k.NumVal:
                            for l in player_cards:
                                if l > k and i.NumVal == l.NumVal:
                                    return True
        return False


    def findFullHouse(self,player_cards):
        """ PURPOSE:  Checks to see if there is a fullhouse in the hand.
            CONTRACT: findFullHouse : list of 5 Card objects -> boolean """

        for i in player_cards:
            for j in player_cards:
                if j > i and i.NumVal == j.NumVal:
                    for k in player_cards:
                        if k > j and i.NumVal == k.NumVal:
                            for l in player_cards:
                                if i.NumVal != l.NumVal:
                                    for m in player_cards:
                                        if m != l and m.NumVal == l.NumVal:
                                            return True
        return False

    def findFlush(self,player_cards):
        """ PURPOSE:  Checks to see if the hand given is a flush. If it is, the hand
                      is appended to self.flushes.
            CONTRACT: findFlush : list of 5 Card objects -> boolean """

        isflush = False
        for i in player_cards:
            for j in player_cards:
                if j > i and i.Suit == j.Suit:
                    for k in player_cards:
                        if k > j and i.Suit == k.Suit:
                            for l in player_cards:
                                if l > k and i.Suit == l.Suit:
                                    for m in player_cards:
                                        if m > l and m.Suit == l.Suit:
                                            isflush = True
        if isflush == True:
            return True
        return False

    def findStraight(self,player_cards):
        """ PURPOSE:  Checks to see if the hand given is a straight. If it is, the hand
                      is appended to self.straights.
            CONTRACT: findStraight : list of 5 Card objects -> boolean """

        isstraight = False
        for i in player_cards:

            # Look for a straight with the Ace number value being 1.
            if i.Ace == True:
                for j in player_cards:
                    if j.NumVal == 2:
                        for k in player_cards:
                            if k.NumVal == 3:
                                for l in player_cards:
                                    if l.NumVal == 4:
                                        for m in player_cards:
                                            if m.NumVal == 5:
                                                isstraight = True

            # Look for a straight with the Ace number value being 14.
            for j in player_cards:
                if j.NumVal == i.NumVal+1:
                    for k in player_cards:
                        if k.NumVal == j.NumVal+1:
                            for l in player_cards:
                                if l.NumVal == k.NumVal+1:
                                    for m in player_cards:
                                        if m.NumVal == l.NumVal+1:
                                            isstraight = True
        if isstraight == True:
            return True
        return False

    def findTrips(self,player_cards):

        """ PURPOSE:  Checks to see if there are trips in the hand of five cards given.
                      if there is, it appends the trips to self.trips.
            CONTRACT: findTrips : list of 5 Card objects -> boolean """

        for i in player_cards:
            for j in player_cards:
                if j > i and i.NumVal == j.NumVal:
                    for k in player_cards:
                        if k > j and i.NumVal == k.NumVal:
                            return True
        return False


    def findTwoPair(self,player_cards):
        """ PURPOSE:  Checks to see if there is more than one pair in the hand.
            CONTRACT: findTwoPair : list of 5 Card objects -> boolean """

        for i in player_cards:
            for j in player_cards:
                if j > i and i.NumVal == j.NumVal:
                    for k in player_cards:
                        if i.NumVal != k.NumVal:
                            for l in player_cards:
                                if l != k and l.NumVal == k.NumVal:
                                    return True
        return False


    def findPair(self,player_cards):
        """ PURPOSE:  Checks to see if there is a pair(s) in the hand of five cards given.
                      If there is, it appends the pair(s) to self.pairs.
            CONTRACT: findPair : list of 5 Card objects -> boolean """

        for i in player_cards:
            for j in player_cards:
                if j > i and i.NumVal == j.NumVal:
                    return True
        return False


    def findBestHand(self):
        """ PURPOSE:  Checks every possible 5 card hand out the 7 given; finds the best
                      hand out of these combinations, and updates self.best_hand to it. """

        allhands = []
        self.best_hand = [0,0,0]
        for i in self.Possible_Hands():
            allhands.append([self.findHandType(i),list(i), self.WeightHand(i,self.findHandType(i))])
        for i in allhands:
            if i[2] > self.best_hand[2]:
                self.best_hand = i

    def WeightHand(self,hand,handtype):
        """ PURPOSE:  Takes a five card hand, determines how strong it is,
                      and represents that strength as a float number.
                      The hand type (flush, straight, two pair, etc) is the
                      ones-value of the weight. Any cards used in the hand
                      type (ex: Qh 10s 3d 3s 3h -> 3d 3s 3h) are the next
                      values in the weight in descending order of number value,
                      and the other cards in the hand are the last values of the
                      weight in descending order.
                      (ex: Qh 10s 3d 3s 3h -> 4.0303031210)
            CONTRACT: WeightHand : list of 5 Card objects, handtype string -> weight float"""

        if handtype == "A Pair":
            pair = []
            for i in hand:
                for j in hand:
                    if i > j and i.NumVal == j.NumVal:
                        pair.append(i)
                        pair.append(j)
            for i in pair:
                hand.remove(i)
            for i in hand:
                for j in hand:
                    for k in hand:
                        if i.NumVal > j.NumVal and j.NumVal > k.NumVal:
                            weight = float(2) + (float(pair[0].NumVal))*0.01 + (float(pair[1].NumVal))*0.0001 + (float(i.NumVal))*0.000001 + (float(j.NumVal))*0.00000001 + (float(k.NumVal))*0.0000000001
                            return weight

        if handtype == "Two Pair":
            weight = float(3)
            pair1 = []
            pair2 = []

            breakloop = False
            for i in hand:
                for j in hand:
                    if i > j and i.NumVal == j.NumVal:
                        pair1.append(i)
                        pair1.append(j)
                        breakloop = True
                        break
                if breakloop == True:
                    break

            for i in pair1:
                hand.remove(i)

            for i in hand:
                for j in hand:
                    if i > j and i.NumVal == j.NumVal:
                        pair2.append(i)
                        pair2.append(j)
            for i in pair2:
                hand.remove(i)

            if pair1[0].NumVal > pair2[0].NumVal:
                weight += (float(pair1[0].NumVal))*0.01 + (float(pair1[1].NumVal))*0.0001 + (float(pair2[0].NumVal))*0.000001 + (float(pair2[1].NumVal))*0.00000001 + (float(hand[0].NumVal))*0.0000000001
            if pair1[0].NumVal < pair2[0].NumVal:
                weight += (float(pair2[0].NumVal))*0.01 + (float(pair2[1].NumVal))*0.0001 + (float(pair1[0].NumVal))*0.000001 + (float(pair1[1].NumVal))*0.00000001 + (float(hand[0].NumVal))*0.0000000001
            return weight

        if handtype == "Trips":
            trips = []
            for i in hand:
                for j in hand:
                    if j > i and i.NumVal == j.NumVal:
                        for k in hand:
                            if k > j and i.NumVal == k.NumVal:
                                trips.append(i)
                                trips.append(j)
                                trips.append(k)
            for i in trips:
                hand.remove(i)

            for i in hand:
                for j in hand:
                    if i.NumVal > j.NumVal:
                        weight = float(4) + float(trips[0].NumVal)*0.01 + float(trips[1].NumVal)*0.0001 + float(trips[2].NumVal)*0.000001 + float(i.NumVal)*0.00000001 + float(j.NumVal)*0.0000000001
            return weight

        if handtype == "Quads":
            quads = []
            for i in hand:
                for j in hand:
                    if j > i and i.NumVal == j.NumVal:
                        for k in hand:
                            if k > j and i.NumVal == k.NumVal:
                                for l in hand:
                                    if l > k and i.NumVal == l.NumVal:
                                        quads.append(i)
                                        quads.append(j)
                                        quads.append(k)
                                        quads.append(l)
            for i in quads:
                hand.remove(i)

            weight = float(8) + float(quads[0].NumVal)*0.01 + float(quads[1].NumVal)*0.0001 + float(quads[2].NumVal)*0.000001 + float(quads[3].NumVal)*0.00000001 + float(hand[0].NumVal)*0.0000000001
            return weight

        if handtype == "A Full House":
            trips = []
            for i in hand:
                for j in hand:
                    if j > i and i.NumVal == j.NumVal:
                        for k in hand:
                            if k > j and i.NumVal == k.NumVal:
                                trips.append(i)
                                trips.append(j)
                                trips.append(k)
            for i in trips:
                hand.remove(i)
            weight = float(7) + float(trips[0].NumVal)*0.01 + float(trips[1].NumVal)*0.0001 + float(trips[2].NumVal)*0.000001 + float(hand[0].NumVal)*0.00000001 + float(hand[1].NumVal)*0.0000000001
            return weight
        if handtype == "A Straight Flush":
            weight = float(9)
        if handtype == "A Flush":
            weight = float(6)
        if handtype == "A Straight":
            weight = float(5)
        if handtype == "A Highcard":
            weight = float(1)

        for i in hand:
            for j in hand:
                for k in hand:
                    for l in hand:
                        for m in hand:
                            if i.NumVal > j.NumVal and j.NumVal > k.NumVal and k.NumVal > l.NumVal and l.NumVal > m.NumVal:
                                weight += float(i.NumVal)*0.01 + float(j.NumVal)*0.0001 + float(k.NumVal)*0.000001 + float(l.NumVal)*0.00000001 + float(m.NumVal)*0.0000000001
                                return weight

    def findHandType(self,player_cards):
        """ PURPOSE:  Calls all of the find hand type methods to determine
                      what the hand type of player_cards are.
            CONTRACT: findHandType : list of 5 Card objects -> hand type string """

        if self.findStraightFlush(player_cards) == True:
            return "A Straight Flush"
        if self.findQuads(player_cards) == True:
            return "Quads"
        if self.findFullHouse(player_cards) == True:
            return "A Full House"
        if self.findFlush(player_cards) == True:
            return "A Flush"
        if self.findStraight(player_cards) == True:
            return "A Straight"
        if self.findTrips(player_cards) == True:
            return "Trips"
        if self.findTwoPair(player_cards) == True:
            return "Two Pair"
        if self.findPair(player_cards) == True:
            return "A Pair"
        else:
            return "A Highcard"
