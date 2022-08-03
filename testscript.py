
############################################ PROBLEM 2 TESTING (DFS)
'''
    probability = 0.3
    dimension = 100
    solvableMazesCount = 0
    
    for i in range(100): # run DFS on 100 mazes
        maze = Maze(dimension, probability)
        maze.generateMaze()
        curr1, visited1 = dfs(maze, 0, 0, dimension-1, dimension-1)
        
        if curr1 is not None and visited1 is not None:
            solvableMazesCount += 1

    print("number of solvable mazes out of 100 is: ", solvableMazesCount)
'''
############################################ PROBLEM 3 TESTING (BFS AND A*)
'''
probability = 1.0
dimension = 100
BFSnumExploredCount = 0
AStarnumExploredCount = 0
numMazes = 30

while(probability >= 0):
    for i in range(numMazes): # run BFS on 30 mazes
        maze = Maze(dimension, probability)
        maze.generateMaze()
        curr, exploredCount = bfs(maze, 0, 0, dimension-1, dimension-1)
        BFSnumExploredCount += exploredCount

        curr, exploredCount = aStar(maze, 0, 0, dimension-1, dimension-1)
        AStarnumExploredCount += exploredCount
    
    avgBFSnumExploredNodes = BFSnumExploredCount / numMazes
    avgAStarnumExploredNodes = AStarnumExploredCount / numMazes
    print("FOR PROBABILITY OF ", probability)
    print("...avg num nodes explored by BFS", avgBFSnumExploredNodes)
    print("...avg num nodes explored by A*", avgAStarnumExploredNodes)
    print()
    
    BFSnumExploredCount = 0
    AStarnumExploredCount = 0

    probability -= 0.1
'''

############################################ PROBLEM 4 TESTING (dfs, bfs, and a*)
'''
probability = 0.3
dimension = 1911

maze = Maze(dimension, probability)
maze.generateMaze()
#maze.displayMaze()
print()

# Problem 4 FIND number of nodes
maze = Maze(dimension, probability)
maze.generateMaze()
startTime = time.time()

curr1, visited1 = aStar(maze, 0, 0, dimension-1, dimension-1)
if curr1 is not 0 and visited1 is not 0:
    numExploredNodes = updateMazePath(maze, findPath(curr1, maze))
endTime = time.time()
        
if visited1 is not 0 and endTime-startTime < 60:
    print(dimension, " finished in ",endTime-startTime," seconds, explored ", numExploredNodes," nodes")
elif visited1 == 0:
    print("no path found")
else:
    print(dimension," didn't finish in 1 minute")

#maze.displayMaze()
'''

############################################ PROBLEM 6 TESTING (Strategies 1,2, and 3)
'''
probability = 0.3
dimension = 10
q = 0.2
strategy = 1
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
                        if curr1 is not 0 and visited1 is not 0:
                            path = findPath(curr1, maze)
                        
                    print("Maze dim:", dimension, ", p = ", probability, ", q = ", q)
                    maze.displayMaze()
                
            #maze.clearMaze(startFireRow, startFireCol)


        if (strategy == 3): # STRATEGY 3
            advanceNumSteps = 0
            fireCells = [(startFireRow, startFireCol)] # initialize list of fire cells with starting index
            curr2, visited2 = aStarOnFire3(maze, 0, 0, dimension-1, dimension-1, fireCells)
            if curr2 is 0 and visited2 is 0:
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
                    if curr2 is not 0 and visited2 is not 0 and fireCells is not 0:
                        path = findPath(curr2, maze)
            
                print("Maze dim:", dimension, ", p = ", probability, ", q = ", q)
                maze.displayMaze()
        
print("count is: " , count)
'''
############################################ DFS, BFS, A* GENERAL TESTING
'''
probability = 0.2
dimension = 50
maze = Maze(dimension, probability)
maze.generateMaze()
'''
'''
curr1, visited1 = dfs(maze, 0, 0, dimension-1, dimension-1)
if curr1 is not 0 and visited1 is not 0:
    path = updateMazePath(maze, findPath(curr1, maze))
maze.displayMaze()
'''
'''
curr1, numExploredNodes = bfs(maze, 0, 0, dimension-1, dimension-1)
if curr1 is not None:
    path = findPath(curr1, maze)
    updateMazePath(maze, path)
maze.displayMaze()
print("numExploredNodes: ", numExploredNodes)
'''
'''
curr1, numExploredNodes = aStar(maze, 0, 0, dimension-1, dimension-1)
if curr1 is not None:
    path = findPath(curr1, maze)
    updateMazePath(maze, path)
maze.displayMaze()
print("numExploredNodes: ", numExploredNodes)
'''