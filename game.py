import numpy as np
import random
import time
import copy
import collections

import tensorflow as tf

import dbGeneration as dbGen


# Load CNN model (ai gameboard evaluation)
sess = tf.Session()


# Unfortunately the scoping behaves very strangely on these tf variables, so they have to be inialised in this ugly way. 
try:
	reuse = True
	gameboards = tf.placeholder(tf.float32, (None, 100), name="gameBoards")
	gameboards2d = tf.reshape(gameboards, (-1, 10, 10, 1), name="gameBoards2d")
		
	# First Convulutional layer, small 2x2 filter
	conv1 = tf.layers.conv2d(gameboards2d, filters=256, kernel_size=2, padding="same", name="Conv1", activation=tf.nn.relu, strides=1, reuse=reuse)
	pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name="Pool1", reuse=reuse)
	conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=3, padding="same", name="Conv2", activation=tf.nn.relu, strides=1, reuse=reuse)
	pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name="Pool2", reuse=reuse)
	conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=4, padding="same", name="Conv3", activation=tf.nn.relu, strides=1, reuse=reuse)
	pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name="Pool3", reuse=reuse)
	# Reshape the 2D tensor back to 1D to be fed into "Dense"
	# Flatten out the pooling - GET THIS NUMBER FROM POOLx.SHAPE
	pool2_flat = tf.reshape(pool3, (-1, int(1*1*128)), name="Pool2_Flat", reuse=reuse)
	# The dropout allows us to train a subset of the neurons at any given iteration.  
	keep_prob = tf.placeholder(tf.float32, name="Keep_Probability", reuse=reuse)
	# A dense layer with dropout
	# DENSE - a fully connected linear transofmration of every dimension of the data
	dense = tf.layers.dense(pool2_flat, int(128), activation=tf.nn.relu, name="Dense", reuse=reuse)
	# DROPOUT - if set to 0.5, rendomly select 50% of the neurons to ignore (different with each computation)
	dropout = tf.nn.dropout(dense, keep_prob, name="Dropout", reuse=reuse)
	dense2 = tf.layers.dense(dropout, int(128), activation=tf.nn.relu, name="Dense2", reuse=reuse)
	dropout2 = tf.nn.dropout(dense2, keep_prob, name="Dropout2", reuse=reuse)
	# A dense layer to classify the final values. Only 2 neurons. 
	predictions = tf.layers.dense(dropout2, 2, activation=None, name="Predictions", reuse=reuse)
	labels = tf.placeholder(tf.int32, [None, 2], name="labels", reuse=reuse)
except:
	gameboards = tf.placeholder(tf.float32, (None, 100), name="gameBoards")
	gameboards2d = tf.reshape(gameboards, (-1, 10, 10, 1), name="gameBoards2d")
		
	# First Convulutional layer, small 2x2 filter
	conv1 = tf.layers.conv2d(gameboards2d, filters=256, kernel_size=2, padding="same", name="Conv1", activation=tf.nn.relu, strides=1)
	pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name="Pool1")
	conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=3, padding="same", name="Conv2", activation=tf.nn.relu, strides=1)
	pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name="Pool2")
	conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=4, padding="same", name="Conv3", activation=tf.nn.relu, strides=1)
	pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name="Pool3")
	# Reshape the 2D tensor back to 1D to be fed into "Dense"
	# Flatten out the pooling - GET THIS NUMBER FROM POOLx.SHAPE
	pool2_flat = tf.reshape(pool3, (-1, int(1*1*128)), name="Pool2_Flat")
	# The dropout allows us to train a subset of the neurons at any given iteration.  
	keep_prob = tf.placeholder(tf.float32, name="Keep_Probability")
	# A dense layer with dropout
	# DENSE - a fully connected linear transofmration of every dimension of the data
	dense = tf.layers.dense(pool2_flat, int(128), activation=tf.nn.relu, name="Dense")
	# DROPOUT - if set to 0.5, rendomly select 50% of the neurons to ignore (different with each computation)
	dropout = tf.nn.dropout(dense, keep_prob, name="Dropout")
	dense2 = tf.layers.dense(dropout, int(128), activation=tf.nn.relu, name="Dense2")
	dropout2 = tf.nn.dropout(dense2, keep_prob, name="Dropout2")
	# A dense layer to classify the final values. Only 2 neurons. 
	predictions = tf.layers.dense(dropout2, 2, activation=None, name="Predictions")
	labels = tf.placeholder(tf.int32, [None, 2], name="labels")

with tf.name_scope("Loss"):
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

with tf.name_scope("Optimizer"):
    train = tf.train.AdamOptimizer(learning_rate=0.001, name="Adam").minimize(loss)

with tf.name_scope("Error"):
	error = tf.reduce_mean(
		tf.cast(tf.not_equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), tf.float32), name="Mean")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint("nn_model\\"))

# 8 Directions
UP = [-1,0]
UPRIGHT = [-1,1]
RIGHT = [0,1]
DOWNRIGHT = [1,1]
DOWN = [1,0]
DOWNLEFT = [1,-1]
LEFT = [0,-1]
UPLEFT = [-1,-1]

WALLNUMBERS = [0,7]

WALL_BONUS = 5
CORNER_BONUS = 10
EMPTY_CORNER_ADJACENCY_PENALTY = -7


AXES = range(8)

# Storing these manually is much cheaper comutationally that computing manhattan distance for every tile
##                          TOP-LEFT          BOTTOM-RIGHT      TOP-RIGHT         BOTTOM-LEFT          
TILES_ADJACENT_TO_CORNER = [[0,1],[1,0],[1,1],[6,7],[7,6],[6,6],[0,6],[1,7],[1,6],[6,0],[7,1],[6,1]]

DIRECTIONS = [UP, UPRIGHT, RIGHT, DOWNRIGHT, DOWN, DOWNLEFT, LEFT, UPLEFT]

class GameBoard(object):
	"""
	0 = Black team
	1 = White team
	NaN = Empty square

	First index is ROW
	Second index is COLUMN e.g. [2,1] is Third row, second column.

	"""

	__slots__ = ["state", "whosTurn", "parent", "depth"]

	def __init__(self, parent):
		"""
		Initialise gameBoard
		"""
		self.whosTurn = 0
		self.parent = parent
		if not parent == None:
			self.depth = parent.depth
		else:
			self.depth = 0

		self.state = np.empty((8,8))
		for i in self.state:
			for j in range(len(i)):
				i[j] = None

		# Starting positions
		self.state[3][3] = self.state[4][4] = 1
		self.state[3][4] = self.state[4][3] = 0        

		return super().__init__()

	def perform_move(self, moveChoice, whosTurn):

		# Movechoice = [[row, col], dirs]
		square = moveChoice[0]

		# dir = [((row, col), n)], where n is number of steps
		for dir in moveChoice[1]:
			movingSquare = square
			for numSteps in range(dir[1]+1):
				movingSquare = np.add(movingSquare, dir[0])
				self.state[movingSquare[0], movingSquare[1]] = whosTurn
		self.state[square[0], square[1]] = whosTurn

	def __str__(self, **kwargs):
		# include info about who's turn it is, etc. 

		boardPrint = ""
		
		boardPrint += "===== Current Board =====\n".format(self.state)

		for row in range(-1,8):
			if row == -1:
				for col in ["A","B","C","D","E","F","G","H"]:
					boardPrint += "\t{}".format(col)
			else:
				for col in range(8):
					if col == 0:
						boardPrint += "{}".format(row+1)
					if self.state[row][col] == 1 or self.state[row][col] == 0:
						boardPrint += "\t{}".format(int(self.state[row][col]))
					else:
						boardPrint += "\t."
			boardPrint += "\n\n"
		if self.whosTurn == 0:
			boardPrint += "===== Player's turn ====="
		else:
			boardPrint += "===== AI's turn ====="

		return boardPrint


def play_game(board, maxSearchDepth = 5):
	print("Game Begin!")
	playerOutOfMoves = False
	aiOutOfMoves = False

	while(True):
		# Player
		if playerOutOfMoves and aiOutOfMoves == True:
			print("Game Over!")
			who_wins(board)

		print(board)

		if board.whosTurn == 0:
			moves = get_available_moves(board.state, board.whosTurn)

			if len(moves) == 0:
				playerOutOfMoves = True
				board.whosTurn = not board.whosTurn
				continue
			else:
				playerOutOfMoves = False
		
			# choose move, then perform        
			board.perform_move(select_move(moves), board.whosTurn)
		# AI
		else:
			moves = get_available_moves(board.state, board.whosTurn)

			if len(moves) == 0:
				aiOutOfMoves = True
				board.whosTurn = not board.whosTurn
				continue
			else:
				aiOutOfMoves = False
			
			#print("Pre move eval:")
			#evaluate_state(board.state, board.whosTurn, moves)

			suggestedMove = select_move_minimax(board, maxSearchDepth)
			print("Suggested Move: ", suggestedMove)

			#choice = random.randint(0, len(moves)-1)
			#print("Choosing move at random")
			board.perform_move(suggestedMove, board.whosTurn)
			
			#print("Post move eval:")
			#evaluate_state(board.state, board.whosTurn, moves)
			
		board.whosTurn = not board.whosTurn

def select_move_minimax(board, maxDepth):
	print("Selecting move with minimax (depth {}), a-B pruning, and a CNN evaluation function".format(maxDepth))

	def get_child_boards(board):
		moves = get_available_moves(board.state, board.whosTurn)
		child_boards = []
		thisDepth = board.depth + 1

		# For every possible move
		for move in moves:
			# copy the board, perform the move
			
			b = copy.deepcopy(board)

			b.perform_move(move, board.whosTurn)
			b.depth = thisDepth
			b.whosTurn = not b.whosTurn

			# Don't evaluate boards on generation, many wil not need to be evaluated. 
			child_boards.append(b)

		return child_boards

	# Note, this function utilises the maxDepth of the outer function (select_move_minimax)
	def evaluate_board_ab_pruning(board, parentAlpha, parentBeta):
		#print("Depth: {}, Turn: {}".format(board.depth, board.whosTurn))
		if board.depth == maxDepth:
			# OLD evaluation function - manual
			#val = evaluate_state(board.state, board.whosTurn, print_messages = False)

			# NEW evaluation function - Convulutional Neural Network
			val = evaluate_board_cnn(board)

			return val 
		else:
			children = get_child_boards(board)
			children_with_values = {}

			# If there are no children available, simply evaluate this current board. 
			if len(children) == 0:
				return evaluate_board_cnn(board)

			if board.whosTurn == whosTurnMinimax: # Maximising
				localAlpha = -999
				maxBoard = None

				for child in children:
					value = evaluate_board_ab_pruning(child, localAlpha, parentBeta) # Alpha is low, Beta is high

					# PRUNE - Stop checking these leaves, they are now redundant
					# At this point, we know this node will never be selected.
					# The proposed value is too low, the parent already has a better (lower) score to pick. 
					if value > parentBeta:
						#print("PRUNE (MAX): {:2f} > {:2f}\t Child {} of {}".format(value, parentBeta, i, len(children)))

						localAlpha = value
						break
					
					if value > localAlpha:
						localAlpha = value
						maxBoard = child
				#i = i + 1

				if board.depth == 0:
					return localAlpha, maxBoard 
				return localAlpha

			else: # Minimising
				localBeta = 999
				for child in children:
					value = evaluate_board_ab_pruning(child, parentAlpha, localBeta)
					
					if value < parentAlpha:
						# At this point, we know this node will never be selected.
						# The proposed value is too low, the parent already has a better (higher) score to pick. 
						localBeta = value
						#print("PRUNE (MIN): {:2f} < {:2f}\t Child {} of {}".format(value, parentBeta, i, len(children)))

						break

					if value < localBeta:
						localBeta = value
					#i = i + 1

				return localBeta

	whosTurnMinimax = board.whosTurn

	# Results[0] is the score, results[1] is the board
	results = evaluate_board_ab_pruning(board, -999, 999)
	print("EVAL SCORE:", results[0])

	print(results)

	# Check which of the available moves matches the minimax result (this is a turd way to do this, I am sorry)
	moves = get_available_moves(board.state, board.whosTurn)

	# This is the target state, this has been shown to be the best move. 
	maskedTargetState = np.ma.array(results[1].state, mask=np.isnan(results[1].state))
	maskedTargetState = np.ma.filled(maskedTargetState, 2)

	# Check each move until finding the one that matches the best move
	for m in moves:
		b = copy.deepcopy(board)
		b.perform_move(m, board.whosTurn)
		
		maskedTestState = np.ma.array(b.state, mask=np.isnan(b.state))
		maskedTestState = np.ma.filled(maskedTestState, 2)

		# Mask the two arrays to ignore the Nan values. Allows comparison1
		if np.ma.all(maskedTestState == maskedTargetState):
			return m

	# If the code reaches here, the board state was returned in tensorformat
	#print("STATE: ", results[1].state)

	for m in moves:
		b = copy.deepcopy(board)
		b.perform_move(m, board.whosTurn)
		
		#maskedTestState = np.ma.array(b.state, mask=np.isnan(b.state))
		#maskedTestState = np.ma.filled(maskedTestState, 2)

		#print(maskedTestState)

		test = convert_board_to_tensorinput(b)

		print(test)
		if np.ma.all(test == results[1].state):
			return m

		
def evaluate_state(state, whosTurn, availableMoves = None, print_messages = True):
	"""
	Hand-tuned state evaluation. 
	This is/will be replaced by a deep CNN.
	"""
	start = time.time()
	evalScore = 0

	if availableMoves == None:
		availableMoves = get_available_moves(state, whosTurn)
	# Number of potential free moves for each player, each potential move adds +1
	myScore = len(availableMoves)
	enemyScore = len(get_available_moves(state, not whosTurn))
	""" # temporarily disabled to increase clarity of debugging

	myScore = 0
	enemyScore = 0"""

	for row in AXES:
		for col in AXES:
			thisPieceTeam = state[row][col]
			if np.isnan(thisPieceTeam):
				continue

			scoreWeight = 1

			# Adjacent to corner
			if [row,col] in TILES_ADJACENT_TO_CORNER:
				# Which corner?
				if row < 4:
					if col < 4: #Upper-left
						relevantCorner = [0,0]
					else: #Upper-right
						relevantCorner = [0,7]
				else: 
					if col < 4: # Lower-left
						relevantCorner = [7,0]
					else: # Lower-Right
						relevantCorner = [7,7]
				if np.isnan(state[relevantCorner[0]][relevantCorner[1]]):
					# The relevant corner is empty, you're gonna have a bad time. 
					if print_messages:
						if thisPieceTeam == 1:
							print("\t\t", end='')
						print("{} Is adjacent to an empty corner (-7)".format([row+1,col+1]))
					scoreWeight = EMPTY_CORNER_ADJACENCY_PENALTY
				else:
					if print_messages:
						if thisPieceTeam == 1:
							print("\t\t", end='')
						print("{} Is adjacent to a filled corner empty corner (+5)".format([row+1,col+1]))
					scoreWeight = WALL_BONUS

			# Next to at least 1 wall
			elif row in WALLNUMBERS or col in WALLNUMBERS:
				# Corner
				if row in WALLNUMBERS and col in WALLNUMBERS:
					scoreWeight = CORNER_BONUS
					if print_messages:
						if thisPieceTeam == 1:
							print("\t\t", end='')
						print("{} Is in a corner (+10)".format([row+1,col+1]))

				# Adjacent to a wall
				else:
					scoreWeight = WALL_BONUS
					if print_messages:
						if thisPieceTeam == 1:
							print("\t\t", end='')
						print("{} Is adjacent to a wall (+5)".format([row+1,col+1]))
					
			if thisPieceTeam == whosTurn:
				evalScore += 1*scoreWeight
				myScore += 1*scoreWeight
			else:
				evalScore -= 1*scoreWeight
				enemyScore += 1*scoreWeight
	if print_messages:
		print("\tmyScore: {}".format(myScore))
		print("\tenemyScore: {}".format(enemyScore))

		print("Took {} sec".format(round(time.time() - start, 4)))

	return myScore

### Functions declared outside of classs to keep gameBoard lightweight
def who_wins(board):
	num0 = 0
	num1 = 0

	for i in board.state:
		for j in i:
			if j == 0:
				num0 = num0 + 1
			elif j == 1:
				num1 = num1 + 1

	print("Player 0: {}".format(num0))
	print("Player 1: {}".format(num1))

	if num0 > num1:
		print("Human player wins!")
	elif num1 > num0:
		print("AI Wins!")
	else:
		print("We have a draw!")
	input()
	exit()

def get_square_moves(row, col, state, whosTurn):
	"""
	Analyse a given square, is this a valid move? If so, in which directions will pieces be taken?.
	NOTE: This is the crux of the computational demand. This is a relatively expensive process to undertake, stepping along each direction.
	The upside is that once this has been calculated, the results can just be referenced when the move is taken, it is only done once. 
	"""
	curSquare = np.array([row,col])
	content = state[curSquare[0],curSquare[1]]

	squareMoves = []

#    [(),(),()]

	# square is taken
	if not np.isnan(content):
		return None

	for dir in DIRECTIONS:
		# The moving square will move one step in a given direction at a time
		movingSquare = np.array([row,col])

		# Move 1 step in the given direction
		movingSquare = np.add(movingSquare, dir)

		try:
			# We find out what is in the first square in this direction
			movingContent = state[movingSquare[0],movingSquare[1]]

			if np.isnan(movingContent): # empty square
				continue
			elif movingContent == whosTurn: # piece in adjacent square matches this player's colour
				continue
			else: # Enemy piece in square
				steps = 0
				# Continue stepping in this direction to find out if this direction yields a valid move
				while True:
					steps = steps + 1
					movingSquare = np.add(movingSquare, dir)

					# If moved out of range
					if movingSquare[0] not in AXES or movingSquare[1] not in AXES:
						break

					# This may step out of range, in which case the loop will continue
					movingContent = state[movingSquare[0], movingSquare[1]]

					if np.isnan(movingContent): # empty square
						# Not fruitful, there was an enemy, but no friend to capture with
						break
					elif not movingContent == whosTurn: # Another friend
						continue
					elif movingContent == whosTurn:# Found friend to capture with
						squareMoves.append((dir, steps))
						break

		except(IndexError):
			continue

	if len(squareMoves) == 0:
		return None
	return squareMoves

def get_available_moves(state, whosTurn):
	"""
	Analyse the board and get a list of available moves.
	"""
	availableMoves = []
	for row in range(8):
		for col in range(8):

			thisSquareMoves = get_square_moves(row, col, state, whosTurn)
			if not thisSquareMoves == None:
				availableMoves.append([[row, col], thisSquareMoves])

	return availableMoves

def select_move(availableMoves):
	"""
	Interface: Allow the player to choose a move. 
	"""
	moveNumbers = range(len(availableMoves))

	print("Select your move!")
	for i in moveNumbers:
		print("\t{}:\t{}{}".format((i+1), chr(availableMoves[i][0][1]+1 + 64), availableMoves[i][0][0]+1))
	
	choice = ""
	while not choice in moveNumbers:
		print("Select your move! Please input a number from 1 - {}!...".format(len(availableMoves)))
		try:
			choice = int(input()) - 1
		except:
			print("Please input an integer!")

	print("You have selected {}: {}".format(choice + 1, availableMoves[choice][0]))
	return availableMoves[choice]

def get_winner_and_scores(moveList):
	gb = GameBoard(None)
	whosTurn = 0
	
	for mv in moveList:
		mv = mv.lower()
		# Convert "A8" input to [7,0] == [row, col]
		mv = [int(mv[1]) - 1, ord(mv[0]) - 97]

		moves = get_available_moves(gb.state, whosTurn)
		
		if len(moves) == 0:
			gb.whosTurn = not gb.whosTurn
		else:
			for move in moves:
				if move[0] == mv:
					gb.perform_move(move, whosTurn)
					whosTurn = not whosTurn
					break
	# All moves performed
	blackScore = 0
	whiteScore = 0

	for i in gb.state:
		for j in i:
			if j == 0:
				blackScore = blackScore + 1
			elif j == 1:
				whiteScore = whiteScore + 1

	if blackScore > whiteScore:
		winner = "Black"
	elif whiteScore > blackScore:
		winner = "White"
	else:
		winner = "Draw"

	return blackScore, whiteScore, winner



def get_artificial_boards(listMoves, howManyStates):
	"""
	Artificially replay through a game, returning a board for every state in the game
	
	Returns: matrix of each turn of the game
	"""
	gb = GameBoard(None)
	whosTurn = 0
	states = []

	# Pick n random turns to actually save from this game
	turnsToSave = random.sample(range(0, len(listMoves)), howManyStates)
	
	counter = 0
	for mv in listMoves:

		mv = mv.lower()
		# Convert "A8" input to [7,0] == [row, col]
		mv = [int(mv[1]) - 1, ord(mv[0]) - 97]

		moves = get_available_moves(gb.state, whosTurn)
		
		if len(moves) == 0:
			gb.whosTurn = not gb.whosTurn
		else:
			for move in moves:
				if move[0] == mv:
					gb.perform_move(move, whosTurn)
					
					# Only save this state if it is specified
					if counter in turnsToSave:
						states.append(copy.deepcopy(gb.state))
					whosTurn = not whosTurn
					done = True
					break
			'''if done == False:
				print(len(moves))
				#print("PRBLEM - THIS INDICATES THAT THE PLAYER HAD NO MOVES TO MAKE, MUST IMPLEMENT 'SKIP MOVE' ")
				print("PRBLEM")'''
		counter = counter + 1
		
	return states

def convert_board_to_tensorinput(board):
	tensorinput = board.state

	for y in range(8):
		for x in range(8):

			# Black
			if tensorinput[x, y] == 0:
				tensorinput[x, y] = -1
			# Not white (empty)
			elif not tensorinput[x, y] == 1:
				tensorinput[x, y] = 0
	return tensorinput



def evaluate_board_cnn(board):
	"""
	board.whosTurn encoding
		0 = Black team
		1 = White team
	
	result encoding (sorry)
		[1,0] = White win
		[0,1] = Black win

	relevantIndex = 1 - board.whosTurn
	enemyScoreIndex = board.whosTurn
	""" # This is implied, and commented out for efficiency

	tInput = np.reshape(dbGen.convert_state_to_tensorformat(board.state), (100))

	r = sess.run(predictions, feed_dict={gameboards:[tInput], keep_prob:1.0})
	# Take the prediction of a friendly win, and subtract the prediction for an enemy win
	return r[0][1 - board.whosTurn] - r[0][board.whosTurn]

#def initialise_model():
	

"""
def evaluate_board(board):
	if board.depth == maxDepth:

		val = evaluate_state(board.state, board.whosTurn, print_messages = False)
		return val 
	else:
		children = get_child_boards(board)
		children_with_values = {}

		# Recursively recall this function until hitting the floor (maxDepth)
		for child in children:
			# NOTE - THIS CURRENTLY IS OVERWRITING WHEN THERE ARE MULTIPLE CHILDREN WITH THE SAME SCORE
			children_with_values[evaluate_board(child)] = child

		# This player tries to maximise the score
		if board.whosTurn == whosTurnMinimax:
			print("d: {}, MAX:  ".format(board.depth), children_with_values)

			# If on the final level, return not only the score, but the board itself.
			if board.depth == 0:
				return max(children_with_values.keys()), children_with_values[max(children_with_values.keys())]
			return max(children_with_values.keys())
		# Simulate enemy player trying to minimise
		else:
			print("d: {}, MIN: ".format(board.depth), children_with_values)
			return min(children_with_values.keys()), 
			"""