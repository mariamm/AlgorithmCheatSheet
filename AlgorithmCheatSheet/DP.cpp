#include "Common.h"

/*
 * Longest increasing subsequence in an array
 * Example {1, 2, 3, 0, 4, 1, 5, 3} -> 1, 2, 3, 4, 5 -> 5
 */
int longestIncreasingSubsequenceLength(const vector<int>& A)
{
    vector<int> dp(A.size(), 1);

    for (int i = 1; i < A.size(); i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (A[i] > A[j])
                dp[i] = max(dp[i], dp[j] + 1);
        }

    }
    return dp[A.size() - 1];
}
/*
vector<int> longestIncreasingSubsequence(const vector<int>& A)
{
    vector<int> dp(A.size(), 1);

    for (int i = 1; i < A.size(); i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (A[i] > A[j])
                dp[i] = max(dp[i], dp[j] + 1);
        }
    }
    
    vector<int> answer(dp[A.size()-1]);
    int index = answer.size() - 1;
    for (int i = dp.size() - 2; i >= 0; i--)
    {
        for (int j = dp.size() - 1; j > i; j--)
        {
            if (dp[i] == dp[j] + 1) // included item
                answer[index--] = A[i];
        }
    }
}*/

//find the longest common subsequence between two strings
int longestCommonSubsequence(string a, string b)
{
    vector<vector<int>> dp(a.length() + 1, vector<int>(b.length() + 1));

    for (int i = 0; i < a.length(); i++)
    {
        for (int j = 0; j < b.length(); j++)
        {
            if (a[i] == b[j])
                dp[i + 1][j + 1] = dp[i][j] + 1;
            else
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);
        }
    }
    return dp[a.length()][b.length()];
}
/*
//find the minimum cost to multiply a chain of matrices.
int matrixChainMultiplication(const vector<vector<int>>& matricesDimensions)
{

}
*/
//find if sum can be reached given following elements (not the same as coin problem, where coins are unlimited)
bool subsetsum(vector<int> elements, int sum)
{
    vector<vector<bool>> dp(elements.size(), vector<bool>(sum + 1));

    for (vector<bool>& v : dp)
        v[0] = true;

    for (int i = 0; i < elements.size(); i++)
    {
        for (int j = 1; j <= sum; j++)
        {
            if (elements[i] == sum)
                dp[i][j] = true;
            else if (elements[i] < sum)
            {
                dp[i][j] = (i > 0 && j >= elements[i]) ? dp[i - 1][j - elements[i]] : 0;
            }
            else
                dp[i][j] = i > 0 ? dp[i - 1][j] : false;
        }
    }

    return dp[elements.size() - 1][sum];
}

//find the minimum number of coins for change
int coinchange1(vector<int> coins, int change)
{
    vector<int> dp(change + 1, INT_MAX); //dp[sum] = numcoins
    dp[0] = 0;
    for (int i = 1; i <= change; i++)
    {
        for (int c : coins)
        {
            if (i == c)
                dp[i] = 1;
            else if (i >= c)
            {
                dp[i] = min(dp[i], dp[i - c] + 1);
            }
        }
    }

    return dp[change] == INT_MAX ? -1 : dp[change];
}

//find the minimum edit distance between two strings
int editDistance(string a, string b)
{
    vector<vector<int>> dp(a.length() + 1, vector<int>(b.length() + 1));

    /*     a b c d e
         0 1 2 3 4 5 //if the same, take diagonal, else, minimum of all three + 1
       c 1 1 2 2 3
       d 2
       b 3
    */

    for (int i = 0; i < a.length(); i++)
    {
        for (int j = 0; j < b.length(); j++)
        {
            if (a[i] == b[j])
                dp[i + 1][j + 1] = dp[i][j];
            else
                dp[i + 1][j + 1] = min({ dp[i][j], dp[i + 1][j], dp[i][j + 1] }) + 1;
        }
    }
    return dp[a.length()][b.length()];
}

int longestIncreasingSubsequence(const vector<int>& A)
{
    vector<int> dp(A.size(), 1);

    for (int i = 1; i < A.size(); i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (A[i] > A[j])
                dp[i] = max(dp[i], dp[j] + 1);
        }

    }
    return dp[A.size() - 1];
}

/* Classic 0/1 knapsack problem
 * Maximize values in knapsack with weights less or equal to capacity. Items are used only once!
 */
int knapsack(vector<int> values, vector<int> weights, int capacity)
{
    vector<vector<int>> dp(values.size() + 1, vector<int>(capacity + 1));

    for (size_t i = 0; i < values.size(); i++)
    {
        int w = weights[i];
        int v = values[i];

        for (size_t j = 1; j <= capacity; j++)
        {
            dp[i + 1][j] = dp[i][j];

            if (j + w <= capacity)
            {
                dp[i + 1][j + w] = max(dp[i + 1][j] + v, dp[i + 1][j + w]);
            }
        }
    }
    return dp[values.size()][capacity];
}
/* Reconstruct items taken in the knapsack 
 * 
 */
vector<int> reconstructKnapsack(vector<vector<int>> dp, vector<int> values, vector<int> weights)
{
    vector<int> items;

    int r = dp.size() - 1;
    int c = dp[0].size() - 1;

    while (r > 0 && c > 0)
    {
        if (dp[r - 1][c] == dp[r][c])
        {
            r--;
        }
        else
        {
            items.push_back(r);
            c = c - weights[r - 1];
            r--;
        }
    }
    
    return items;
}

/* 1D Knapsack no repetition
 */
int knapsack1DnoRep(vector<int> values, vector<int> weights, int capacity)
{
    vector<int> dp(capacity + 1);

    for (int i = 0; i < values.size(); i++)
    {
        int v = values[i];
        int w = weights[i];

        // iterate in reverse to not include item several times
        for (int j = capacity - w; j >= 0; j--)
            dp[j+w] = max(dp[j]+v, dp[j + w]);
    }

    return dp[capacity];
}

/* 1D Knapsack with repetition (like coin change problem)
 */
int knapsack1DwithRep(vector<int> values, vector<int> weights, int capacity)
{
    vector<int> dp(capacity + 1);

    for (int i = 0; i < values.size(); i++)
    {
        int v = values[i];
        int w = weights[i];

        // iteratation includes repetition
        for (int j = 0; j <= capacity - w; j++)
            dp[j+w] = max(dp[j]+v, dp[j + w]);
    }

    return dp[capacity];
}
/* Knapsack but with weights up to 1e9
* The idea is to flip the dp[capacity] = max_value to dp[value] = min_capactiy
*/
int knapsackWeights(vector<int> values, vector<int> weights, int capacity)
{
    int sum_values = 0;
    for (int i : values)
        sum_values += i;

    vector<int> dp(sum_values + 1, 1e9+1);

    for (int i = 0; i < values.size(); i++)
    {
        int v = values[i];
        int w = weights[i];

        // iterate in reverse to not include item several times
        for(int j = sum_values - v; j >= 0; j--)
            dp[j+v] = min(dp[j]+w, dp[j + v]);
    }

    for (int i = dp.size() - 1; i >= 0; i--)
    {
        if (dp[i] <= capacity)
            return i;
    }
    return 0;
}