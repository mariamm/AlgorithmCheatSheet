#include "Common.h"

/*Recursion (aka Backtracking) Problems*/

/* Basic sudoku solver
 * Assumes input is valid and has a valid solution, 
 * otherwise the same board is the result
 * Empty cells to be filled have value 0
 */
struct SudokuSolver
{
    vector<vector<int>> board;
    vector<unordered_set<int>> rows;  //set for every row in board
    vector<unordered_set<int>> cols;  //set for every column in board
    vector<unordered_set<int>> boxes; //set for every box
    vector<pair<int, int>> todo;      //empty cells to be filled

    SudokuSolver()
    {
        //demo board
        vector<vector<int>> board = {
            {7,5,0,2,8,0,0,3,0},
            {8,1,0,0,0,9,7,2,5},
            {3,0,2,7,1,5,0,8,9},
            {0,2,1,0,0,8,0,4,0},
            {5,0,8,0,0,0,2,0,0},
            {4,3,0,0,2,0,0,0,8},
            {0,0,0,0,0,0,8,0,0},
            {0,8,0,0,4,1,9,5,2},
            {0,0,0,8,0,2,0,0,0}
        };
    }
    SudokuSolver(vector<vector<int>>& b) {
        board = b;
        rows.resize(9);
        cols.resize(9);
        boxes.resize(9);

        for (int i = 0; i <9; i++){
            for (int j = 0; j <9; j++){
                if (b[i][j] == 0)
                    todo.push_back(make_pair(i, j));
                else{
                    rows[i].insert(b[i][j]);
                    cols[j].insert(b[i][j]);
                    boxes[idx(i, j)].insert(b[i][j]);
                }
            }
        }
    }
    int idx(int r, int c)
    {
        //box 0: rows 012, cols 012 --> 0 * 3 + 0
        //box 1: rows 012, cols 345 --> 0 * 3 + 1
        //box 2: rows 012, cols 678 --> 0 * 3 + 2
        //box 3: rows 345, cols 012 --> 1 * 3 + 0
        //box 4: rows 345, cols 345 --> 1 * 3 + 1
        //box 5: rows 345, cols 678 --> 1 * 3 + 2
        //box 6: rows 678, cols 012 --> 2 * 3 + 0
        //box 7: rows 678, cols 345 --> 2 * 3 + 1
        //box 8: rows 678, cols 678 --> 2 * 3 + 2

        return (r / 3) * 3 + c / 3;
    }
    bool validEntry(int r, int c, int num)
    {
        if (rows[r].find(num) != rows[r].end())
            return false;

        if (cols[c].find(num) != cols[c].end())
            return false;

        if (boxes[idx(r,c)].find(num) != boxes[idx(r, c)].end())
            return false;

        return true;
    }

    bool recurseSolve(int todoIndex)
    {
        if (todoIndex == todo.size())
            return true;

        int r = todo[todoIndex].first;
        int c = todo[todoIndex].second;
        for (int i = 1; i <= 9; i++)
        {
            if (validEntry(r, c, i))
            {
                board[r][c] = i;
                rows[r].insert(i);
                cols[c].insert(i);
                boxes[idx(r,c)].insert(i);
                if (recurseSolve(todoIndex + 1))
                    return true;
                else
                {
                    board[r][c] = 0;
                    rows[r].erase(i);
                    cols[c].erase(i);
                    boxes[idx(r, c)].erase(i);
                }
            }
        }
        return false;
    }
    vector<vector<int>> solve()
    {
        recurseSolve(0);
        return board;
    }

    void prettyPrint(vector<vector<int>> board)
    {
        for (int i = 0; i < 9; i++)
        {
            if (i % 3 == 0)
                cout << "---------------------" << endl;
            
            for (int j = 0; j < 9; j++)
            {
                if (j % 3 == 0) 
                    cout << "|";
                
                cout << board[i][j] << " ";
            }
            cout << "|" << endl;
        }
        cout << "---------------------" << endl;
    }

    /*
    
        */
};


////////////////////////////////////////////
////////////   Backtracking   //////////////
////////////////////////////////////////////

/* Graph Coloring: 1. Check if graph is Bipartite (split in two groups connected to the other only)
 * Solution: Check if graph can be colored with 2 colors
 * @param graph: graph presented in adjacency list
 * @param v: current vertix
 * @param color: color of current vertix
 * @param visited: visited vertices set
 * @return: bool if graph can be colored
 */

bool colorGraph(vector<vector<int>>& graph, int v, int color, unordered_map<int, int>& visited)
{
    for (int n : graph[v])
    {
        if (visited.find(n) == visited.end())
        {
            visited.insert({ n, -color });
            if (!colorGraph(graph, n, -color, visited))
                return false;
        }
        else
        {
            if (visited[n] == color)
                return false;
        }
    }
    return true;
}

bool isBipartite(vector<vector<int>>& graph) {
    unordered_map<int, int> visited;

    for (int i = 0; i < graph.size(); i++)
    {
        if (visited.find(i) == visited.end())
        {
            visited.insert({ i, 1 });
            if (!colorGraph(graph, i, 1, visited))
                return false;
        }
    }
    return true;
}
/* Graph Coloring: 2.Check if graph can be colored with 3 colors.
 * @param graph: graph presented in adjacency list
 * @param v: current vertix
 * @param colors: input vector of colors
 * @param color: color of current vertix
 * @param visited_color: visited vertices map and their color
 * @return: bool if graph can be colored
 */

bool colorGraph2(vector<vector<int>>& graph, int v, vector<int> colors, int c, unordered_map<int, int> visited_color)
{
    for (int n : graph[v])
    {
        if (visited_color.find(n) == visited_color.end()) //neighbor not visited yet
        {
            for (int i = 0; i < colors.size(); i++)
            {
                if (i == c)
                    continue; //skip same color
                visited_color[n] = i;
                if (colorGraph2(graph, n, colors, i, visited_color) == false)
                {
                    //remove set color (to try next color)
                    visited_color.erase(n);
                }
                else
                {
                    break;
                }
            }
        }
        else if (visited_color[n] == c)
            return false;

    }
    return true;
}
bool colorGraph2Main(vector<vector<int>>& graph, vector<int> colors)
{
    //Assuming a connected graph
    unordered_map<int, int> visited_color;
    for (int i = 0; i < graph.size(); i++)
    {
        //color not visited nodes
        if (visited_color.find(i) == visited_color.end())
        {
            for (int c = 0; c < colors.size(); c++)
            {
                visited_color[i] = c;
                if (colorGraph2(graph, i, colors, c, visited_color))
                {
                    break; //found correct color
                }
                else
                {
                    visited_color.erase(i);
                }
            }
        }

    }
    return visited_color.size() == graph.size();
}
/*3 Variant Examples for permutation function :
 * Given a collection of numbers, return all possible permutations. Aka Heap's algorithm.
 * @param num : initial vector
 * @param start : start index for the recursive call
 * @param result : output list of vector permutations
*/
//Variant 1
void permute(vector<int>& num, int start, vector<vector<int> >& result) {
    if (start == num.size() - 1) {
        result.push_back(num);
        return;
    }
    for (size_t i = start; i < num.size(); i++) {
        swap(num[start], num[i]);
        permute(num, start + 1, result);
        swap(num[start], num[i]);
    }
}
/*Variant 2
 *@param num : initial vector
* @param curr : current vector for recursive call
* @param result : output list of vector permutations
* Not really a good solution considering that we have to modify nums with INT_MAX to not be reused, which limits the
* possibility of having INT_MAX as an actual value in the input
*/
void permute(vector<int>& nums, vector<int>& curr, vector<vector<int>>& result) {
    if (curr.size() == nums.size()) {
        result.push_back(curr);
        return;
    }
    for (size_t i = 0; i < nums.size(); i++)
    {
        if (nums[i] != INT_MAX)
        {
            curr.push_back(nums[i]);
            int temp = nums[i];
            nums[i] = INT_MAX;
            permute(nums, curr, result);
            nums[i] = temp;
            curr.pop_back();
        }
    }
}
//Variant 3 (using std::next_permutation)
void permute(vector<int>& nums, vector<vector<int>>& result)
{
    sort(nums.begin(), nums.end()); //must be sorted in the beginning, otherwise permutations are missing
    do {
        result.push_back(nums);
    } while (next_permutation(nums.begin(), nums.end()));
}
