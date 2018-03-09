import numpy as np
from bs4 import BeautifulSoup
import requests
import json
import os
import game 
import pickle


def _create_game_urls():
	gameURLs = []
	for round in range(1,13):
		for t in range(1, 25): # extra numbers, some not actually associated with games
			url = "http://www.o-wc.com/live/t{}/{}.html".format(round, t)
			gameURLs.append(url)
	return gameURLs

def _convert_wtb_to_move_lists():
	gameMoves = []
	files = os.listdir("data\\gamedata_txt")
	for file in files:
		print(file)
		with open("{}{}".format("data\\gamedata_txt\\", file), "r") as fp:
			
			lines = fp.readlines()
			lines = [x for x in lines if not x == "\n"]
			gameMoves.extend(lines)

		#print(lines)
	#print(gameMoves)

	print(len(gameMoves))
	return gameMoves

def _convert_movelists_to_json(rawDB):
	
	#try:
	#	#Probably redudant since the whole DB comes as one now
	#	with open("data\\gamedb2.json", "r") as fp:
	#		games = json.load(fp)
	#	print("HELLO")
	#	id = 999999999999
	#	for i in games:	
	#		id = max(id, i)
	#except:
	games = {}
	id = 999999999999


	for moveList in rawDB:
		id = id + 1
		# assign ID
		if id % 1000:
			print(id)
		games[id] = {}
		
		# decipher winner
		movesArr = []
		moveList = moveList.upper()
		for idx in range(int(len(moveList) / 2)):
			movesArr.append(moveList[idx*2:idx*2+2])

		blackScore, whiteScore, winner = game.get_winner_and_scores(movesArr)

		games[id]["ScoreBlack"] = str(blackScore) # Forgive me, needed to match previous html crawler input
		games[id]["ScoreWhite"] = str(whiteScore) # Forgive me, needed to match previous html crawler input
		games[id]["Winner"] = winner
		games[id]["ListMoves"] = movesArr

		# Checkpointing
		if id % 1000 == 0:
			print("SAVING CHECKPOINT")
			with open("data\\gamedb2.json", "w") as fp:
				fp.write(json.dumps(games, "gamedb2.json", indent=4))

	with open("data\\gamedb2.json", "w") as fp:
		fp.write(json.dumps(games, "gamedb2.json", indent=4))
		
		#games[g]["ListMoves"] = str(games[g]["ListMoves"]).upper()
		#if games[g]["ScoreWhite"] > games[g]["ScoreBlack"]:
		#	games[g]["Winner"] = "White"
		#else:
		#	games[g]["Winner"] = "Black"

		#moves = games[g]["ListMoves"]
		#movesArr = []
		#for idx in range(int(len(moves) / 2)):
		#	movesArr.append(moves[idx*2:idx*2+2])

		#games[g]["ListMoves"] = movesArr


# Web scraper for pulling down the game info
def _get_games():
	games = {}
	gameURLs = _create_game_urls()
	sess = requests.Session()

	for gameURL in gameURLs:
		print(gameURL)

		r = sess.get(gameURL)

		if r.status_code == 200:
			# Grab the html element for the game table
			try:

				soup = BeautifulSoup(r.text)
				gameTable = soup.find_all("table", id="gameTable")

				# Convert the text to a string and get all the rows from it.
				scoreBlack = soup.find_all("label", id="scoreBlack")[0].text
				scoreWhite = soup.find_all("label", id="scoreWhite")[0].text
				listMoves = soup.find_all(attrs={"name": "ListMoves"})[0]["value"].strip()

				games[gameURL] = {"ScoreBlack": scoreBlack,
						 "ScoreWhite": scoreWhite,
						 "ListMoves": listMoves}
				print(games[gameURL])
			except:
				pass
		else:
			print("{} Failed".format(gameURL))
	return games

def _establish_game_database(override = False):
	if not os.path.isfile("data/gamedb.json") or override:
		with open("data/gamedb.json", "w") as fp:

			fp.write(json.dumps(_get_games(), indent=4))

	# Process Data
	with open("data/gamedb.json", "r") as fp:
		games = json.load(fp)

	for g in games:
		games[g]["ListMoves"] = str(games[g]["ListMoves"]).upper()
		if games[g]["ScoreWhite"] > games[g]["ScoreBlack"]:
			games[g]["Winner"] = "White"
		else:
			games[g]["Winner"] = "Black"

		moves = games[g]["ListMoves"]
		movesArr = []
		for idx in range(int(len(moves) / 2)):
			movesArr.append(moves[idx*2:idx*2+2])

		games[g]["ListMoves"] = movesArr
	
	with open("data/gamedb.json", "w") as fp:
		fp.write(json.dumps(games, indent=4))

def convert_state_to_tensorformat(state):
	#state = state.reshape(100)
	state[state == 0] = -1
	state[np.isnan(state)] = 0

	return state

def _augment_board_states(boardStates):

	augmentedBoardStates = []

	for board in boardStates:
		tests = []

		padded = np.zeros((10,10))
		padded.fill(9)
		padded[1:board.shape[0]+1, 1:board.shape[1]+1] = board

		for k in [1,2,3,4]:
			tmp = np.rot90(padded, k)
			augmentedBoardStates.append(tmp)
			augmentedBoardStates.append(np.fliplr(tmp))

	return augmentedBoardStates

def get_tensorinputs_and_labels(forceNew = False, flip = False, rotate = False, takeEveryStep = False, howManyStates = 2):
	"""
	HowManyStates: How many of the stages (turns) of each game will be sampled, a lower number will 
	reduce the risk of overfitting (I think), but result in a smaller dataset
	"""
	if not os.path.isfile("data/gamedb80.json"):
		print("There is no Game DB")
		_establish_game_database()

	# If the processing has not ben performed, or forcing new
	if not os.path.isfile("data/tensorinputs.p") or not os.path.isfile("data/labels.p") or forceNew:
		for file in os.listdir("data"):
			n = os.path.splitext(file)[0]
			n = str.replace(n, "gamedb", "")
			print(n)

			try:
				with open("data/{}".format(file)) as fp:
					db = json.load(fp)
			except:
				print("Failed to load data/{}, it probably was a filder, not a JSON file".format(file))
				continue

			# We are now sure that the data is there, and can begin converting it to tf input format

			labels = []
			tensorinputs = []
			db = {}

			for i in list(db.keys()):
				print(i)
				thisGame = db[i]
				boardStates = game.get_artificial_boards(thisGame["ListMoves"], 2)
			
				print(len(boardStates))

				# Should we augment (oversample) the boards, by rotating or flipping them?
				if flip or rotate:
					boardStates = _augment_board_states(boardStates)
					print(len(boardStates))

				for b in boardStates:
					tensorinputs.append(convert_state_to_tensorformat(b))
					if thisGame["Winner"] == "Black":
						labels.append(1)
					else:
						labels.append(-1)

				# Only do it once
				#break
			pickle.dump(tensorinputs, open("data/tensorinputs{}.p".format(n), "wb"))
			pickle.dump(labels, open("data/labels{}.p".format(n), "wb"))
	else:
		tensorinputs = pickle.load(open("data/tensorinputs.p", "rb"))
		labels = pickle.load(open("data/labels.p", "rb"))

	print("Returned {} data points".format(len(tensorinputs)))
	return tensorinputs, labels
		

def try_read():
	with open("data/WTH_2001.wtb", "rb") as fobj:

		biglist = list()
		counter = -1

		for line in fobj:

			if len (line) <= 1:
				continue
			if line[0] == "/":
				continue

			counter += 1
			"""
			if output_format == 1:
				if counter < start:
					continue
				if counter >= end:
					break
			"""

			size = len (line) / 2
			templist = list ()
			for i in range (int (size)):
				index = 2 * i
				index1 = index + 1
				char = line[index]
				x = ord (char)
				if x < ord ('a'):
					x = x - ord ('A')
				else:
					x = x - ord ('a')
				y = int (line [index1]) - 1
				templist.append ((x, y))
			biglist.append (templist)

		fobj.close ()
		
		print(check_bom(file))

		print(file[:10])