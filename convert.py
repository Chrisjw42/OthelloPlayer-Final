# Author: Argens Ng
# Description: This program converts raw match data into particular state in a match

import os.path
import random
from pathlib import Path

# ================ Non Program Specific ================ #

# Description: Clears the standard output screen
def clearScreen ():
	print ("\033c")

# Description: Display a message after clearing screen
# Input:
#	Msg: message to be displayed
# Output: None
def displayMsg (msg = ""):
	clearScreen ()
	print (msg)


# Description: Helping function which helps query the user for answers with or without choices
# Input:
#	[STR] question: questions to be asked
#	[(*, ) STR] choices: [optional] choices for the user to choose from
# Output:
#	[STR] answer: 	1) answer
#					2) index of choice [if choices are provided]
def query (question = "", choices = []):

	if len (choices) == 0:
		clearScreen ()
		question = question +"\n\n"
		answer = input (question)
		
	else:
		for i in range (len (choices)):
			question = question + "\n"
			question = question + str(i) + ") "
			question = question + choices [i]
		question = question + "\n\n"

		answer = len (choices)

		while answer >= len (choices):
			clearScreen ()
			try:
				answer = int (input (question))
			except ValueError:
				answer = len (choices)

	return (answer)

# Description: Helps unclash a filename so as to prevent overwriting files
# Input:
#	[STR] name: filename
#	[STR] extension: extension
# Output:
#	[STR] suitable filename that is not used
def unclash (name):
	counter = 1

	filename = Path (name + str (counter))
	while filename.is_file():
		counter += 1
		filename = Path (name + str (counter))

	return (name + str(counter))



# ================== Program Specific ================== #

# Description: Collect user options specific to this program
# Input: None
# Output:
#	[(STR, INT, INT)] tuple of filename, output format (0 being compressed states aand 1 being neat and presentable states) and number of games to decompress
def collectOptions ():

	q = "What is the filename?"	
	file = Path ("")
	while not file.is_file():
		filename = query (question = q)
		file = Path (filename)

	q = "What is the output format?"
	option = list()
	option.append ("Byte")
	option.append ("Human-readable Complete Game")
	output_format = query (question = q, choices = option)

	q = "How many games do you want to decompress? (Default is 1.)"
	while True:
		try:
			ans = query (question = q)
			if len (ans) == 0:
				gameSize = 1
			else:
				gameSize = int (ans)
		except ValueError:
			continue
		else:
			break

	return (filename, output_format, gameSize)

# ==================== Main Program ==================== #

filename, output_format, gameSize = collectOptions()

if output_format == 1:
	q = "Any game to start from? (Count starts from 0. Can be left empty)"
	while True:
		try:
			ans = query (question = q)
			if len (ans) == 0:
				start = 0
				break
			start = int (ans)
		except ValueError:
			continue
		else:
			break

	end = gameSize + start

if output_format == 0:
	q = "Any particular seed to set on? (Can be left empty)"
	while True:
		try:
			ans = query (question = q)
			if len (ans) == 0:
				break
			seedn = int (ans)
			random.seed (seedn)
		except ValueError:
			continue
		else:
			break

# ==================== Getting Moves =================== #

fobj = open (filename, "rb")
biglist = list()
counter = -1

for line in fobj:

	if len (line) <= 1:
		continue
	if line[0] == "/":
		continue

	counter += 1
	if output_format == 1:
		if counter < start:
			continue
		if counter >= end:
			break

	size = len (line) / 2
	templist = list ()
	for i in range (int (size)):
		index = 2 * i
		index1 = index + 1
		x = ord (line[index])
		if x < ord ('a'):
			x = x - ord ('A')
		else:
			x = x - ord ('a')
		y = int (line [index1]) - 1
		templist.append ((x, y))
	biglist.append (templist)

fobj.close ()

# ============ Generating States + Writing ============ #

currentDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = currentDirectory + "/.."

if output_format == 0:
	directoryName = "Byte"
else:
	if output_format == 1:
		directoryName = "Game"

outputDirectory = parentDirectory + "/" + directoryName

if not os.path.exists (outputDirectory):
	try:
		os.makedirs (outputDirectory)
	except OSError:
		displayMsg ("OSError: We do not have permission or space to create a directory.")

if output_format == 1:
	counter = start - 1
	for game in biglist:
		counter += 1

		name = "Record_" + str (counter)
		name = outputDirectory + "/" + name
		outputFile = open (name, "w")
		outputFile.close ()

		state = State ()
		for line in game:
			state.printToFile (name, line)
			x = line [0]
			y = line [1]
			
			outputFile = open (name, "a")
			outputFile.write ("Move:	" + line.__str__() + "\n")
			outputFile.write ("==========================\n\n")
			outputFile.close ()
			state = state.move (int (x), int (y))

		state.printToFile (name)

		outputFile = open (name, "a")

		if (state.bc > state.wc):
			outputFile.write ("Black has won by:")
			outputFile.write (state.count.__str__())
		else:
			if (state.wc > state.bc):
				outputFile.write ("White has won by:")
				outputFile.write (state.count.__str__())
			else:
				outputFile.write ("It is a tie of:")
				outputFile.write (state.count.__str__())

		outputFile.close ()

	print (counter)

if output_format == 0:

	name = outputDirectory + "/Train_"
	name = unclash (name)
	#random.shuffle (biglist)
	with open (name, "w") as f:

		counter = 0
		for game in biglist:
			counter += 1
			if counter > gameSize:
				break

			# Mid-Game moves were picked
			pickedMove = random.randint (5, len (game)-1)
			moveCounter = 0
			state = State ()
			for line in game:
				if moveCounter == pickedMove:
					(rep, player) = state.asByte ()
					f.write (rep)
					f.write (line.__str__())
				state = state.move (int (line[0]), int (line[1]))
				moveCounter += 1
			
			if state.bc > state.wc:
				if player == 1:
					f.write ("1")
				else:
					f.write ("-1")
			else:
				if state.bc < state.wc:
					if player == 1:
						f.write ("-1")
					else:
						f.write ("1")
				else:
					f.write ("0")

			f.write ("\n")