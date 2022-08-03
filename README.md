# FireMaze
## Rutgers University, Intro to AI 440 Project 1
### Team members: Vidya Mannava, Heeral Narkhede

This project is to find a path from a starting position to the goal for the agent to follow. Along the way there will be obstacles such as walls and a fire that starts in a random initial position and spreads with some probability in all directions. We have solved this by using:
- BFS
- DFS
- A*
- A* with a cost heuristic (best performing algorithm)

In order to find the best strategy to exit the maze, we tried several different strategies: 
- Strategy 1: Find the shortest path and follow this path until you exit the maze or burn using the A* algorithm.
- Strategy 2: Recompute the shortest path after every step forwards based on the current state of the maze and fire using A*.
- Strategy 3: We created an A* algorithm that was a combination of Strategies 1 and 2 but is a more efficient algorithm by adding a cost heuristic to when the path should be recomputed. It also adds a weight predict where the fire is likely to spread in the future so that path is less favored by the agent.


Improvements in the future include a time constraint for when the next step should be taken in to simulate a real life scenario.
