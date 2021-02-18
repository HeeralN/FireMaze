import random
import math

from queue import Queue
from heapq import heappush, heappop

import time

############################################################################################################## Maze 
class Maze:
    def __init__(self, dimension, p, flamibilityRate = 0):
        self.dim = dimension
        self.maze = []
        self.p = p
        self.flamibilityRate = flamibilityRate
    
    # assign each cell in the grid a random value between 1 and 10 and randomly select p distinct numbers to block
    # WHAT IF p IS A LONG DECIMAL PROBABILITY?
    def generateMaze(self):
        self.maze = [[random.randint(0,9) for i in range(self.dim)] for j in range(self.dim)]
        
        # generates p*10 random ints
        randomInts = []
        while len(randomInts) < self.p*10:
            x = random.randint(0,9)
            if x not in randomInts:
                randomInts.append(x)
    
        for i in range(self.dim):
            for j in range(self.dim):
                if i == 0 and j == 0:
                    self.maze[i][j] = "S"
                elif i == self.dim-1 and j == self.dim-1:
                    self.maze[i][j] = "G"
                elif self.maze[i][j] in randomInts:
                    self.maze[i][j] = "X"
                else:
                    self.maze[i][j] = "-"


    def updateMaze(self, row, col, newState):
        self.maze[row][col] = newState


    # chooses random fire start cell, returns index tuple of cell
    def startFire(self): 
        i, j = 0
        # can't start fire on start, goal, or blocked node
        while (i == 0 and j == 0) or (i == self.dim-1 and j == self.dim-1) or self.maze[i][j] == 'X':
            i = random.randInt(0,self.dim-1)
            j = random.randInt(0,self.dim-1)

        self.maze.updateMaze(i,j,"F")
        return (i,j)


    def advanceFire(self, q): # returns list of cells on fire [ (row,col), ...]
        # NOTESSSSSS:  so fireCells.append(advanceFire) will add all new fire cells
        cellsOnFire = []
        for i in range(self.dim):
            for j in range(self.dim):
                if self.maze[i][j] != "F":
                    # count number of neighbors on fire
                    k = 0
                    if i < self.dim - 1 and self.maze[i+1][j] == "F":
                        k+=1
                    if self.maze[i-1][j] == "F":
                        k+=1
                    if j < self.dim - 1 and self.maze[i][j+1] == "F":
                        k+=1
                    if self.maze[i][j-1] == "F":
                        k+=1
                    
                    prob = 1 - (1-q) ** k
                    if random.random() <= prob:
                        self.maze.updateMaze(i, j, "F")
                        cellsOnFire.append(i,j) # appends new cell on fire to list
        return cellsOnFire


    def closestFireCell(self, curr, fire): # curr is (row,col), fire is list of indexes on fire
        min = float('inf')
        for i,j in fire:
            distance = euclideanDistance(curr, (i,j))
            if distance < min:
                min = distance
        return min


    def displayMaze(self):
        for a in range(self.dim):
            for b in range(self.dim):
                print(self.maze[a][b], end =" ")
            print()





############################################################################################################## Data Structures- Stack, Node, Queue 
class Node:
    def __init__(self, index, parent, state = None, currCost = float('inf'), heuristic = float('inf')):
        self.row = index[0] # index is a tuple (row,col)
        self.col = index[1]
        self.state = state  # string possibilities: S,G,-,F,X
        self.parent = parent  # parent is a Node
        self.heuristic = heuristic  # cost from node to goal
        self.currCost = currCost # cost to get to this node
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __lt__(self, other):
        return self.heuristic < other.heuristic

    def tostring(self):
        return "(" + self.row + "," + self.col + ")"


class Stack:
    def __init__(self):
        self.fringe = []

    def is_empty(self):
        return not self.fringe

    def push(self, cell): # cell is a tuple (row,col)
        self.fringe.append(cell)

    def pop(self):  # removes & returns value of last item added to fringe
        return self.fringe.pop()
    
    def __repr__(self):  #  prints the stack
        return repr(self.fringe)


class PriorityQueue:
    def __init__(self):
        self.pqueue = []

    def is_empty(self):
        return not self.pqueue
    
    def get(self):
        return heappop(self.pqueue)[1]
    
    def put(self, priority, node):
        heappush(self.pqueue,(priority, node))
    
    def __str__(self):
        return str(self.pqueue)






############################################################################################################## Helper Methods
# checks if cell at index is a valid open cell to visit
# index is a tuple, visited is a set
def isAvailable(maze, index, visited):
    row,col = index
    return 0 <= row < maze.dim and 0 <= col < maze.dim and maze.maze[row][col] != 'X' and maze.maze[row][col] != 'F' and index not in visited


# returns # of explored nodes
def updateMazePath(curr, maze, visited):  # curr is the goal node
    path = [(curr.row, curr.col)]  # list of tuples (row, col)

    # shortest path denoted by *
    while curr.parent is not None:
        curr = curr.parent
        path.append((curr.row, curr.col))
        if not curr.state == "S":
            maze.updateMaze(curr.row, curr.col, '*')
    
    # all visited nodes not in shortest path denoted by %
    for i in range(maze.dim):
        for j in range(maze.dim):
            if (i,j) in visited and (i,j)not in path:
                maze.updateMaze(i, j, ".")
    
    return len(visited)


# prints all nodes currently in fringe
def printFringe(fringe): 
    print("fringe: ")
    while fringe is not is_empty:
        print(fringe.pop())


# computes euclideanDistance from (row1,col1) to (row2,col2)
def euclideanDistance(curr, goal):  # curr, goal are tuples (row, col)
    print (math.sqrt((curr[0] - goal[0] - 1)**2 + (curr[1] - goal[1]) ** 2))


# same functionality as isAvailable except does not check that index is not in visited
def isValid(maze, index):
    row,col = index
    return 0 <= row < maze.dim and 0 <= col < maze.dim and maze.maze[row][col] != 'X' and maze.maze[row][col] != 'F'





############################################################################################################## DFS, BFS, A* 
# DFS
def dfs(maze, sRow, sCol, gRow, gCol):
    fringe = Stack() # things to visit
    fringe.push(Node((sRow, sCol), None, 'S')) # adds start node to fringe
    visited = set() # holds tuples (row, col) of visited cells 

    while not fringe.is_empty():
        curr = fringe.pop()
        
        if curr.row == gRow and curr.col == gCol: # if current node is goal
            return (curr, visited)
        #print("visited: ", curr.row, curr.col)
        visited.add((curr.row, curr.col))

        # if it is an open, valid node and has not already been visited
        if helperMethods.isAvailable(maze, (curr.row-1, curr.col), visited):  # left 
            fringe.push(Node((curr.row-1, curr.col), curr, maze.maze[curr.row-1][curr.col]))

        if helperMethods.isAvailable(maze, (curr.row, curr.col - 1), visited):  # top
            fringe.push(Node((curr.row, curr.col - 1), curr, maze.maze[curr.row][curr.col - 1]))

        if helperMethods.isAvailable(maze, (curr.row, curr.col+1), visited):  # down
            fringe.push(Node((curr.row, curr.col+1), curr, maze.maze[curr.row][curr.col+1]))

        if helperMethods.isAvailable(maze, (curr.row+1, curr.col), visited):  # right
            fringe.push(Node((curr.row+1, curr.col), curr, maze.maze[curr.row+1][curr.col]))
    return None,None  # goal was never found


# BFS
def bfs(maze, sRow, sCol, gRow, gCol):
    fringe = Queue() # things to visit
    fringe.put(Node((sRow, sCol), None, 'S')) # adds start node to fringe
    visited = set() # holds tuples (row, col) of visited cells 

    while not fringe.empty():
        curr = fringe.get()
        
        if curr.row == gRow and curr.col == gCol: # if current node is goal
            return curr,visited

        #print("visited: (", curr.row, ",", curr.col, ")")
        visited.add((curr.row, curr.col))

        # if it is an open, valid node and has not already been visited
        if helperMethods.isAvailable(maze, (curr.row-1, curr.col), visited):  # left 
            fringe.put(Node((curr.row-1, curr.col), curr))

        if helperMethods.isAvailable(maze, (curr.row, curr.col - 1), visited):  # top
            fringe.put(Node((curr.row, curr.col - 1), curr))

        if helperMethods.isAvailable(maze, (curr.row, curr.col+1), visited):  # down
            fringe.put(Node((curr.row, curr.col+1), curr))

        if helperMethods.isAvailable(maze, (curr.row+1, curr.col), visited):  # right
            fringe.put(Node((curr.row+1, curr.col), curr))
        
    return None,None  # goal was never found


# A*
def aStar(maze, sRow, sCol, gRow, gCol):
    pass





############################################################################################################## ALL TESTING
if __name__ == "__main__":
    probability = 0.3
    dimension = 10
    q = 0.3

    maze = Maze(dimension, probability)
    maze.generateMaze()

    cellsOnFire = maze.advanceFire(q)
    for cell in cellsOnFire:
        print("(" , cell[0], "," , cell[1] , ")")


    ############################################ PROBLEM 2 TESTING (dfs)
    '''
    probability = 0.0
    dimension = 100

    solvableMazesCount = 0
    
    for i in range(100): # run DFS on 100 mazes
        maze = Maze(dimension, probability)
        maze.generateMaze()
        curr1, visited1 = dfs(maze, 0, 0, dimension-1, dimension-1)
        
        if curr1 != None and visited1 !=  None:
            solvableMazesCount += 1
 
    print("number of solvable mazes out of 100 is: ", solvableMazesCount)
    '''


    ############################################ PROBLEM 3 TESTING (bfs and a*)
    '''
    probability = 0.0
    dimension = 50

    BFSnumExploredNodes = 0
    AStarnumExploredNodes = 0

    numSolvableMazes = 1
    
    for i in range(1): # run BFS on 5 mazes
      maze = Maze(dimension, probability)
      maze.generateMaze()
      curr, visited = bfs(maze, 0, 0, dimension-1, dimension-1)
  
      if curr == 0 and visited == 0:
          numSolvableMazes -= 1
      else:
          BFSnumExploredNodes += len(visited)

    avgBFSnumExploredNodes = BFSnumExploredNodes / numSolvableMazes
    print("total num nodes explored by BFS: ", BFSnumExploredNodes)
    print("avg num nodes explored by BFS", avgBFSnumExploredNodes)
    
    
    numSolvableMazes = 5
    
    for i in range(5): # run A* on 5 mazes
        maze = Maze(dimension, probability)
        maze.generateMaze()
        curr, visited = A*(maze, 0, 0, dimension-1, dimension-1)
  
        if curr is 0 and visited is 0:
            numSolvableMazes -= 1
        else:
            AStarnumExploredNodes += len(visited)

    avgAStarnumExploredNodes = AStarnumExploredNodes / numSolvableMazes
    print("total num nodes explored by A*: ", AStarnumExploredNodes)
    print("avg num nodes explored by A*", avgAStarnumExploredNodes)
    '''



############################################ PROBLEM 4 TESTING (dfs, bfs, and a*)
    '''
    probability = .5
    dimension = 10

    time = 60
    maze = Maze(dimension, probability)
    maze.generateMaze()
    #maze.displayMaze()
    #print()
    
    curr1, visited1 = dfs(maze, 0, 0, dimension-1, dimension-1)
    # curr1, visited1 = bfs(maze, 0, 0, dimension-1, dimension-1)
    # curr1, visited1= aStar(maze, 0, 0, dimension-1, dimension-1)
    if curr1 is not 0 and visited1 is not 0:
        numExploredNodes = updateMazePath(curr1, maze, visited1)
    maze.displayMaze()
    '''