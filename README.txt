   
   APPLICATION NAME:   THAIP_Game_Loop

	     AUTHOR:   Joseph Carrick

	      EMAIL:   joecarrick106@gmail.com

SYSTEM REQUIREMENTS:   Python 2.7




INSTALLATION AND INSTRUCTIONS:

	-This application is a 1v1 game of no-limit Texas Hold'em Poker against an AI program.
	 To learn about Texas Hold'em and to review it's rules, go to
	 https://en.wikipedia.org/wiki/Texas_hold_%27em#Rules

	-Once python 2.7 is installed, run THAIP_Game_Loop.py. Make sure that the following files
	 are in the same directory as THAIP_Game_Loop.py:

		- THAIP_Class_Table.py

		- THAIP_Class_Card.py

		- THAIP_Class_GamePhase.py

		- THAIP_Class_DetermineHand.py

		- THAIP_Class_AI.py


	-Text displaying information about the game will appear in the application window.
	 The format of the text is:


		<dealer's actions>
	
		<human player's chip total>

		<human player's cards>

		<AI player's chip total>

		<AI player's cards>

		<the community cards>

		<the pot chip total>
	
		<AI's actions or input promp for user>


	 -Every phase of the game (pre-flop, flop, turn and river, show cards) will bring up text
	  in this format. At the bottom of this information will be text promping the user to input
	  his/her action. There are 4 possible action inputs: "Check", "Call", "Fold", and "Raise".

	 -Type in the desired action and press return.

	 -If "Raise" is inputed, another input prompt will appear asking the user to input the 
	  amount he/she wants to raise. Possible inputs are any whole integer from 1 to the user's
	  chip total. Use the number pad to enter the amount to raise and press return.
	 
	 -The game loop runs until either player is out of chips.
	