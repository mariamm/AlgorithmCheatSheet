// AlgorithmCheatSheet.cpp  
#include "Common.h"
#include "AlgorithmCheatSheet.h"
#include "recursion.h"
#include "DP.h"

int main()
{ 
    vector<vector<int>> adjList = { {1, 2}, {3, 4}, {5, 6}};
    vector<vector<int>> rotated = rotateImage2(adjList, true);
    print2dVector(rotated);
    int x;
    cin >> x;
} 