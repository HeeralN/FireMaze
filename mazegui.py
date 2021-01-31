def generateMaze(dim, p):
    maze = []
    for x in range(dim):
        currRow = []
        for y in range(dim):
            if(x==0 and y==0):
                print("S", end =" ")
                continue
            elif(x==dim-1 and y==dim-1):
                print("G", end =" ")
            else: 
                #if():
                    currRow.append("|")
                    print("|", end =" ")
                #elif():
                #    print("Empty")
        maze.append(currRow)
        print()


generateMaze(5, 0.25)




