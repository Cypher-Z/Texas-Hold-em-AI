#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      carricktel
#
# Created:     23/09/2015
# Copyright:   (c) carricktel 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from THAIP_Class_Table import Table

class AI:

    def __init__(self,table):
        self.table = table
        self.action = [0,"",0]


    def Run_Draft_Algorithm(self):
        """ PURPOSE:  Looks at the state of the table and follows the draft
                      algorithm to determine what action to play. The algorithm
                      is very simple: If the opponent raised, then call. If the
                      opponent checked, folded, or called, then check. """

        if self.table.lastAction[1] == "Raise":
            self.action = [2,"Call",self.table.lastAction[2]]
        if self.table.lastAction[1] == "Check" or self.table.lastAction[1] == "Fold" or self.table.lastAction[1] == "Call" or self.table.lastAction[1] == "" or self.table.lastAction[1] == "pass":
            self.action = [2,"Check",0]
