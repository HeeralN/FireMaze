import random

def generateMaze(dim, p):
    maze = []
    #generate maze row by row
    for x in range(dim):
        currRow = []
        for y in range(dim):
            if(x==0 and y==0):
                currRow.append("S")
            elif(x==dim-1 and y==dim-1):
                currRow.append("G")
            else: 
                r = random.random()
                if(r < p):
                    currRow.append("|")
                elif(r >= p):
                   currRow.append(":")
        maze.append(currRow)

    #print maze
    for a in range(dim):
        for b in range(dim):
            print(maze[a][b], end =" ")
        print()


generateMaze(5, 0.25)





