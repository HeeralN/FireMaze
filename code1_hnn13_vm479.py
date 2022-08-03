import random
import math
from queue import Queue
from heapq import heappush, heappop
import time


############################################################################################################## Maze 
'''
Maze Class:
    - Generates a 2D maze by randomly assigning walls and empty spaces that are guaranteed to have a path from the start to the goal
    - Starts the fire in a random location and as randomly spreads the fire in random directions
'''
class Maze:
    def __init__(self, dimension, p, flamibilityRate = 0):
        self.dim = dimension
        self.maze = []
        self.p = p
        self.flamibilityRate = flamibilityRate
    
    # assign each cell in the grid a random value between 1 and 10 and randomly select p distinct numbers to block
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
        i = 0
        j = 0
        # can't start fire on start, goal, or blocked node
        while (i == 0 and j == 0) or (i == self.dim-1 and j == self.dim-1) or self.maze[i][j] == 'X':
            i = random.randint(0,self.dim-1)
            j = random.randint(0,self.dim-1)

        self.updateMaze(i,j,"F")
        return (i,j)


    def advanceFire(self): # returns list of cells on fire [ (row,col), ...]
        fireCells = []
        for i in range(self.dim):
            for j in range(self.dim):
                if self.maze[i][j] != "F" and self.maze[i][j] != "S" and self.maze[i][j] != "G" and self.maze[i][j] != "X":
                    # count number of neighbors on fire
                    k = 0
                    if i+1 < self.dim and self.maze[i+1][j] == "F":
                        k+=1
                    if i - 1 > 0 and self.maze[i-1][j] == "F":
                        k+=1
                    if j + 1 < self.dim and self.maze[i][j+1] == "F":
                        k+=1
                    if j - 1 > 0 and self.maze[i][j-1] == "F":
                        k+=1
                    
                    prob = 1 - (1 - self.flamibilityRate) ** k
                    if random.random() <= prob:
                        fireCells.append((i,j)) # returns new cell on fire
        for i,j in fireCells:
            self.updateMaze(i, j, "F")

        return fireCells

    def closestFireCell(self, curr, fireCells): # curr is (row,col), fire is list of indexes on fire
        min = float('inf')
        for (i,j) in fireCells:
            distance = euclideanDistance(curr, (i,j))
            if distance < min:
                min = distance
        return min

    def displayMaze(self):
        for a in range(self.dim):
            for b in range(self.dim):
                print(self.maze[a][b], end =" ")
            print()

    def clearMaze(self, row, col):
        for i in range(self.dim):
            for j in range(self.dim):
                if self.maze[i][j] != 'S' and self.maze[i][j] != 'G' and self.maze[i][j] != 'X' and i != row and i != col :
                    self.maze[i][j] = "-"



############################################################################################################## Data Structures- Stack, Node, Queue 
class Stack:
    def __init__(self):
        self.fringe = []

    def is_empty(self):
        return not self.fringe

    def push(self, cell): # cell is a tuple (row,col)
        self.fringe.append(cell)

    def pop(self):  # removes & returns value of last item added to fringe
        return self.fringe.pop()
    

class Node:
    def __init__(self, index, parent, state = None, currCost = float('inf'), heuristic = float('inf')):
        self.row = index[0] # index is a tuple (row,col)
        self.col = index[1]
        self.state = state  # string possibilities: S,G,-,F,X
        self.parent = parent  # parent is a Node
        self.heuristic = heuristic  # cost from node to goal
        self.currCost = currCost # cost to get to this node
    

class PriorityQueue:
    def __init__(self):
        self.pqueue = []

    def is_empty(self):
        return not self.pqueue
    
    def get(self):
        return heappop(self.pqueue)[1]
    
    def put(self, priority, node):
        heappush(self.pqueue,(priority, node))




############################################################################################################## Helper Methods
def findPath(curr, maze):  # curr is the goal node
    path = [(curr.row, curr.col)]  # list of tuples (row, col)
    # shortest path denoted by *
    while curr.parent is not None:
        curr = curr.parent
        path.insert(0,(curr.row, curr.col))
    return path  # returns a list of path cells, not including the start cell

# returns # of explored nodes
def updateMazePath(maze, path, fireCells = None):  
    # shortest path denoted by *
    if fireCells is not None:
        for i,j in fireCells:
            maze.maze[i][j]='F'
    for i,j in path:
        if maze.maze[i][j]== "F":
            break
        if not maze.maze[i][j]== "S" and not maze.maze[i][j]== "G":
            maze.maze[i][j]='*'
    return path

# used for A*
def euclideanDistance(curr, goal):  # curr, goal are tuples (row, col)
    return math.sqrt((curr[0] - goal[0] - 1)**2 + (curr[1] - goal[1]) ** 2)
    

############################################################################################################## Searches (dfs, bfs, a*)
# DFS
def dfs(maze, sRow, sCol, gRow, gCol):
    fringe = Stack() # things to visit
    fringe.push(Node((sRow, sCol), None, 'S')) # adds start node to fringe
    dim = gRow + 1
    visited = [[0 for i in range(dim)] for j in range(dim)] # dim x dim grid: 0 = not visited, 1 = visited
    exploredCount = 0

    while not fringe.is_empty():
        curr = fringe.pop()
        
        if curr.row == gRow and curr.col == gCol: # if current node is goal
            return curr, exploredCount
        
        visited[curr.row][curr.col] = 1

        left = (curr.row, curr.col-1)
        top = (curr.row-1, curr.col)
        down = (curr.row+1, curr.col)
        right = (curr.row, curr.col+1)

        for row,col in [left, top, down, right]:
            # if it is an open, valid node and has not already been visited
            if (0 <= row < maze.dim and 0 <= col < maze.dim and maze.maze[row][col] != 'X' and visited[row][col]==0):
                fringe.push(Node((row,col), curr))

    return None,exploredCount  # goal was never found

# BFS
def bfs(maze, sRow, sCol, gRow, gCol):
    fringe = Queue() # things to visit
    fringe.put(Node((sRow, sCol), None, 'S')) # adds start node to fringe
    exploredCount = 0 # have not explored any cell's children yet
    dim = gRow + 1
    visited = [[0 for i in range(dim)] for j in range(dim)] # dim x dim grid: 0 = not visited, 1 = visited

    while not fringe.empty():
        curr = fringe.get()
        if curr.row == gRow and curr.col == gCol: # if current node is goal
            return curr, exploredCount

        exploredCount += 1 # curr is not goal, so will explore its children
        visited[curr.row][curr.col] = 1 # mark curr as visited

        # possible children
        left = (curr.row, curr.col-1)
        top = (curr.row-1, curr.col)
        down = (curr.row+1, curr.col)
        right = (curr.row, curr.col+1)

        for row,col in [left, top, down, right]:
            # if this is a valid child
            if 0 <= row < maze.dim and 0 <= col < maze.dim and maze.maze[row][col] != 'X':
                if visited[row][col]==0: # if this child has not been visited
                    fringe.put(Node((row,col), curr))
                    visited[row][col] = 1 # mark this child as visited


    return None,exploredCount  # goal was never found

# A*
def aStar(maze, sRow, sCol, gRow, gCol):
    fringe = PriorityQueue() # list of Nodes to visit
    fringe.put(0, Node((sRow, sCol), None, 'S', 0, 0)) # adds start node -> fringe
    exploredCount = 0 # have not explored any cell's children yet
    visitedCosts = {(sRow,sCol):0} # key = (row,col) : value = cost so far
    dim = gRow + 1
    visited = [[0 for i in range(dim)] for j in range(dim)] # dim x dim grid: 0 = not visited, 1 = visited
    
    while not fringe.is_empty():
        curr = fringe.get() 
        if curr.row == gRow and curr.col == gCol: # if current node is goal
            return curr, exploredCount 

        exploredCount += 1 # curr is not goal, so will explore its children
        visited[curr.row][curr.col] = 1 # mark curr as visited

        left = (curr.row, curr.col-1)
        top = (curr.row-1, curr.col)
        down = (curr.row+1, curr.col)
        right = (curr.row, curr.col+1)
        goal = (gRow, gCol)

        for row,col in [left, top, down, right]:
            # if it is an open, valid node 
            if 0 <= row < maze.dim and 0 <= col < maze.dim and maze.maze[row][col] != 'X' and maze.maze[row][col] != 'F':
                cost = curr.currCost + 1  
                heuristic = euclideanDistance((row,col), goal)
                totalIndexCost = cost + heuristic
                # if node is not already visited or path found with less cost from start -> node
                if visited[row][col]==0 or cost < visitedCosts[(row,col)]:
                    visited[row][col] = 1
                    fringe.put(totalIndexCost, Node((row,col), curr, '-', cost, heuristic))
                    visitedCosts[(row,col)] = cost
                
    return None,exploredCount  # goal was never found
    

############################################################################################################## Strategy 3: A* with Fire Heuristic
def aStarOnFire3(maze, sRow, sCol, gRow, gCol, fireCells):
    fringe = PriorityQueue() # list of Nodes to visit
    fringe.put(0, Node((sRow, sCol), None, 'S', 0, 0)) # adds start node -> fringe
    dim = gRow + 1
    visited = [[0 for i in range(dim)] for j in range(dim)] # dim x dim grid: 0 = not visited, 1 = visited
    exploredCount = 0

    while not fringe.is_empty():
        curr = fringe.get()
        exploredCount += 1
        if curr.row == gRow and curr.col == gCol: # if current node is goal
            return curr, visited, exploredCount
        
        left = (curr.row, curr.col-1)
        top = (curr.row-1, curr.col)
        down = (curr.row+1, curr.col)
        right = (curr.row, curr.col+1)
        goal = (gRow, gCol)

        for row,col in [left, top, down, right]:
            # if it is an open, valid node
            if 0 <= row < maze.dim and 0 <= col < maze.dim and maze.maze[row][col] != 'X' and maze.maze[row][col] != 'F':
                cost = curr.currCost + 1
                weight = -0.3   # importance of fire distance
                fireDistance = maze.closestFireCell((row,col), fireCells)  # closest fire cell distance
                heuristic = euclideanDistance((row,col), goal) + weight * fireDistance  
                
                # if node is not already visited 
                if visited[row][col]==0 : 
                    # heuristic is based on how far the fire is from current node

                    fringe.put(heuristic, Node((row,col), curr, None, cost, euclideanDistance((row,col), goal)))
                    #visitedCosts[(row,col)] = cost
                visited[row][col] = 1 
    
    return 0,0,exploredCount  # goal was never found


############################################################################################################## ALL TESTING
if __name__ == "__main__":

    while True:
        print("Welcome to the Maze on Fire Solver! The maze will be randomly generated and contains the following symbols: " +
                "\n * = Path Taken \n X = Wall \n S = Start \n F = Fire \n G = Goal \n")
        num = int(input("Enter which type of algorithm to test: 1 for BFS, 2 for DFS, 3 for A*, 4 for A* with Fire Heuristic OR 0 to exit:\n"))
        if num == 0:
            break

        probability = float(input("Enter the probability of fire spreading (try 0.3): \n"))
        dimension = int(input("Enter the dimension of the maze to test (try 30 or 100): \n"))

        if num == 1: # BFS
            maze = Maze(dimension, probability)
            maze.generateMaze()

            curr1, numExploredNodes = bfs(maze, 0, 0, dimension-1, dimension-1)
            if curr1 is not None:
                path = findPath(curr1, maze)
                updateMazePath(maze, path)
            maze.displayMaze()
            # print("numExploredNodes: ", numExploredNodes)


        if num == 2:  # DFS
            maze = Maze(dimension, probability)
            maze.generateMaze()
            
            curr1, visited1 = dfs(maze, 0, 0, dimension-1, dimension-1)
            if curr1 != 0 and visited1 != 0:
                path = updateMazePath(maze, findPath(curr1, maze))
            maze.displayMaze()

        if num == 3: # A*
            maze = Maze(dimension, probability)
            maze.generateMaze()
            
            curr1, numExploredNodes = aStar(maze, 0, 0, dimension-1, dimension-1)
            if curr1 is not None:
                path = findPath(curr1, maze)
                updateMazePath(maze, path)
            maze.displayMaze()
            # print("numExploredNodes: ", numExploredNodes)

        if num == 4: # A* with Fire Heuristic
            # probability = 0.3
            # dimension = 10
            # q = 0.2  # flammability rate
            q = input("Enter the flammability rate (suggestion is to try 0.2): \n")
            strategy = input("Enter the strategy to use (descriptions found in the README): 1, 2 or 3 \n")
            count = 0

            while count < 30:
                # initialize maze
                maze = Maze(dimension, probability, q)
                maze.generateMaze()
                
                # start fire and make sure it can reach agent's start position
                (startFireRow,startFireCol) = maze.startFire() # initial fire cell
                print("Fire starts at (",startFireRow,", ", startFireCol,")")
                curr1, visited1 = aStar(maze, startFireRow, startFireCol, 0, 0)
                if visited1 is None:
                    print("Fire CAN'T reach agent")
                else:
                    # run one of the strategies
                    path = []
                    
                    if strategy == 1 or strategy == 2:
                        curr1, = aStar(maze, 0, 0, dimension-1, dimension-1)
                        if curr1 is None:
                            print("Unsolvable maze.")
                        else:
                            path = findPath(curr1, maze)   # in order [(row,col),...] from start to goal

                            if strategy == 1: # STRATEGY 1
                                onFireCells = []
                                for i,j in path:
                                    onFireCells += maze.advanceFire()
                                    if maze.maze[i][j] == "F":
                                        print("Strategy 1: Caught on fire.")
                                        break
                                updateMazePath(maze, path, onFireCells)
                                count += 1 # agent can reach fire
                                #print("Maze dim:", dimension, ", p = ", probability, ", q = ", q)
                                #maze.displayMaze()
                                #print()
                                

                            elif strategy == 2: # STRATEGY 2
                                # while there is still nodes left in path, advance one step
                                while len(path) > 0:
                                    path.pop(0)  # move forwards one step
                                    if len(path) == 0:  # goal is found
                                        print("Strategy 2: Goal found!")
                                        break
                                    i,j = path[0]  # explore this cell
                                    if not maze.maze[i][j] == 'G':
                                        maze.updateMaze(i,j,"*")
                                    maze.advanceFire()   # advance fire
                                    if maze.maze[i][j] == 'F':
                                        print("Strategy 2: Caught on fire.")
                                        break
                                    
                                    # recompute path
                                    curr1, visited1 = aStar(maze, 0, 0, dimension-1, dimension-1)
                                    if curr1 != 0 and visited1 != 0:
                                        path = findPath(curr1, maze)
                                    
                                print("Maze dim:", dimension, ", p = ", probability, ", q = ", q)
                                maze.displayMaze()
                        
                        #maze.clearMaze(startFireRow, startFireCol)


                    if (strategy == 3): # STRATEGY 3
                        advanceNumSteps = 0
                        fireCells = [(startFireRow, startFireCol)] # initialize list of fire cells with starting index
                        curr2, visited2 = aStarOnFire3(maze, 0, 0, dimension-1, dimension-1, fireCells)
                        if curr2 == 0 and visited2 == 0:
                            print("Unsolvable maze.")
                        else:
                            path = findPath(curr2, maze)  # in order list of path taken
                            while len(path) > 0:
                                #print(fireCells)
                                distanceFromFire = maze.closestFireCell(path[0], fireCells)
                                advanceNumSteps = int(distanceFromFire / 2)
                                lastNodeState = ""
                                # advance some # of steps based on how far the fire is
                                for a in range(advanceNumSteps):
                                    i,j = path.pop(0)
                                    if len(path) == 0:  # if goal has just been popped
                                        print("Strategy 3: Goal found!")
                                        break
                                    i,j = path[0]  # explore this cell
                                    lastNodeState = maze.maze[i][j]
                                    if not lastNodeState == 'G':
                                        maze.updateMaze(i,j,"*")
                                    fireCells += maze.advanceFire()   # advance fire
                                    if lastNodeState == 'F':
                                        print("Strategy 3: Caught on fire.")
                                        break
                                if lastNodeState == "F":
                                    break
                                if len(path) == 0:
                                    break
                                i,j = path[0]
                                # make new path
                                curr2, visited2= aStarOnFire3(maze, i, j, dimension-1, dimension-1, fireCells)
                                if curr2 != 0 and visited2 != 0 and fireCells != 0:
                                    path = findPath(curr2, maze)
                        
                            print("Maze dim:", dimension, ", p = ", probability, ", q = ", q)
                            maze.displayMaze()
                    
            print("count is: " , count)