import time
import game as g
import numpy as np
#import dbGeneration as dbGen
#import convert

import tensorflow as tf

if __name__ == "__main__":
	print("Program Begin.")
	start = time.time()
	#dbGen.establish_game_database()

	gb = g.GameBoard(None)
	g.play_game(gb, maxSearchDepth = 4)

	print("Total time taken: %s sec" % round((time.time() - start), 10))
