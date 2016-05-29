from THAIP_Class_Table import Table

class Player:

    def __init__(self,table):
        self.table = table
        self.action = [0,"",0]


    def Player_Action_Algorithm(self):

        if self.table.lastAction[1] == "Raise":
            self.action = [2,"Call",self.table.lastAction[2]]
        if self.table.lastAction[1] == "Check" or self.table.lastAction[1] == "Fold" or self.table.lastAction[1] == "Call" or self.table.lastAction[1] == "" or self.table.lastAction[1] == "pass":
            self.action = [2,"Check",0]
