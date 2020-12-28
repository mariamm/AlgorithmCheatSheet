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