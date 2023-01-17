
import numpy as np


# Function to create canonical pattern or 1-factorization
def findCanonicalPattern(numberOfTeams, numberOfRounds): 
       
    x = numberOfRounds//2
    y = numberOfTeams//2
    z = 2    
    
    # Creates 3-dimensional array. For example, for 4 teams: x = 2, y = 3, z = 2. So E = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    E = np.zeros((x,y,z)) 
    
    for i in range(numberOfRounds//2): 
        
        E[i][0][:]=[numberOfTeams,i + 1]        # The first edge of a round is the last team (e.g. team 4) playing team i + 1 
        
        for k in range(numberOfTeams//2-1):      # Then to fill the last edges, use functions F1 and F2
            
            E[i][k+1][:]=[F1(i + 1, k + 1, numberOfTeams), F2(i + 1, k + 1, numberOfTeams)] 
    np.random.shuffle(E)
    return(E) 

    
# Defines F1 used to find the canonical pattern   
def F1(i,k,numberOfTeams):
    
    if i + k < numberOfTeams:
        
        return(i + k)
        
    else:
        
        return(i + k - numberOfTeams + 1)
    
    
# Defines F2 used to find the canonical pattern     
def F2(i,k,numberOfTeams):
    
    if i - k > 0:
        
        return(i - k)
        
    else:
        
        return(i - k + numberOfTeams - 1)

print(findCanonicalPattern(4,6)) 

# Defines function to get initial solution for Simulated Annealing
def getInitialSolution(numberOfTeams,numberOfRounds):
    
    # The solution will be a 2-dimensional array (a.k.a. the schedule)
    solution = np.zeros((numberOfTeams,numberOfRounds), dtype=int)
    
    # Finds canonical pattern to creat a feasible single round robin schedule
    games = findCanonicalPattern(numberOfTeams, numberOfRounds)
    
    # Creates first half of the tournament
    for i in range(numberOfRounds//2):
        
        for k in range(numberOfTeams//2):
            
            # Every edge of the canonical pattern is a game between the two nodes
            edge = games[i][k]
            
            teamA = int(edge[0])
            teamB = int(edge[1])
            
            # One team plays at home, one team plays away
            solution[teamA - 1][i] = teamB
            solution[teamB - 1][i] = - teamA
    
    # To create second half, mirror the first half inverting the signs
    temp = solution.copy()
    temp = -1*np.roll(temp, numberOfRounds//2, axis=1)
    solution = solution+temp

    return(solution)
print( getInitialSolution(4,6))
distanceS = getInitialSolution(4,6)
totaldistance = 0
for x in range(4): 

    # If the entry is positive, team x is playing at home. So make the entry equal to team x. Else, take the absolute of the entry in the schedule.
    distanceS[x] = [x + 1 if i > 0 else abs(i) for i in distanceS[x]]

    # Starts by adding distance from team x to first entry in the schedule. If first entry is team x, you're playing at home and the distance added is 0.
    print(distanceS[x])
