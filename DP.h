#include "Common.h"

/* Random collection of DP problems*/
namespace DP
{
    // Geeks for geeks list

    /*1. Ugly numbers
     * Ugly numbers are numbers whose only prime factors are 2, 3 or 5
     */
    int uglyNumbers(int n)
    {
        set<int> ugly;
        ugly.insert(1);
        while (n)
        {
            int first = *ugly.begin();

            ugly.erase(first);

            ugly.insert(first * 2);
            ugly.insert(first * 3);
            ugly.insert(first * 5);
            n--;
        }
        return *ugly.begin();
    }
    /* 2. Fibonacci 
    * return nth fibonacci number
    */ 
    int fib(int n) { 
        if (n < 2)
            return n;
        int d1 = 0;
        int d2 = 1;

        for (int i = 2; i <= n; i++)
        {
            int temp = d2;
            d2 += d1;
            d1 = temp;
        }
        return d2;
    }

    /*
    * Catalan number:
    * E.g. number of paths, number of trees
        C0=1 \ and \ Cn+1= sum{i:n}Ci * Cn-i  for n >= 0;    
    */
    int catalanNumber(int n)
    {
        if (n < 2)
            return n;
        vector<int> dp(n + 1);
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                dp[i] += dp[j] * dp[i-j-1];
            }
        }
        return dp[n];
    }

    /* Bell number
    * E.g.: Number of ways to partition a set
    */
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
    string longestCommonSubsequenceReconstructed(string a, string b)
    {
        string s = ""; 
         
        return s;
    }

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

    int longestIncreasingSubsequence(const vector<int>& A, vector<int> &answer)
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
        answer.resize((dp[A.size() - 1]));
        int index = (int)answer.size() - 1;
        for (int i = (int)dp.size() - 2; i >= 0; i--)
        {
            for (int j = (int)dp.size() - 1; j > i; j--)
            {
                if (dp[i] == dp[j] + 1) // included item
                    answer[index--] = A[i];
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
        //i-->items (processed)
        //j--> used capacity
        //dp[i][j] maximum value considering i items so far and using j capacity

        for (size_t i = 1; i <= values.size(); i++)
        {
            int w = weights[i-1];
            int v = values[i-1];

            for (size_t j = 1; j <= capacity; j++)
            { 
                if ( j>=w)
                {
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-w]+v);
                }
                else
                    dp[i][j] = dp[i - 1][j];
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

        int r = (int) dp.size() - 1;
        int c = (int) dp[0].size() - 1;

        while (r > 0 && c > 0)
        {
            if (dp[r - 1][c] == dp[r][c]) //we didn't include this item, we took the value of the previous row
            {
                r--;
            }
            else
            {
                items.push_back(r); // we took this item, now subtract it's weight to find the (previous) taken item 
                c = c - weights[r - 1];
                r--;
            }
        }
    
        return items;
    }

    /* 1D Knapsack no repetition*/
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
        size_t sum_values = 0;
        for (int i : values)
            sum_values += i;

        int INF = (int) 1e9 + 1;
        vector<int> dp(sum_values + 1, INF);

        for (size_t i = 0; i < values.size(); i++)
        {
            int v = values[i];
            int w = weights[i];

            // iterate in reverse to not include item several times
            for(size_t j = sum_values - v; j-- > 0;)
                dp[j+v] = min(dp[j]+w, dp[j + v]);
        }

        for (size_t i = dp.size() - 1; i-- > 0;)
        {
            if (dp[i] <= capacity)
                return (int)i;
        }
        return 0;
    }

    /* modiefied DFS with dp for max distances to a node*/
    void longestDistanceDfs(const vector<vector<int>>& adjList, vector<bool>& visited, int v, vector<int>& maxdis)
    {
        if (visited[v] == true)
            return;

        visited[v] = true;
        for (int n : adjList[v])
        {
            if (!visited[n])
                longestDistanceDfs(adjList, visited, n, maxdis);

            maxdis[v] = max(maxdis[v], maxdis[n] + 1);
        }
    }
    /*
    * Longest Path in a DAG
    * Using dynamic programming to store longest distance (from) a node.
    */
    int longestDistance(const vector<vector<int>> &adjList, int N)
    {   
        vector<int>maxdis(N + 1);
        vector<bool> visited(N + 1);
        for (int i = 1; i <= N; i++)
        {
            if (!visited[i])
                longestDistanceDfs(adjList, visited, i, maxdis);
        }

        int longest = 0;
        for (int i : maxdis)
            longest = max(longest, i);

        return longest;
    }

    /* Probabilty of coin tosses (more heads than tail)
    * Given the probability of N bias coins, return the probabilty of tossing more heads than tails
    * DP solution: i--> coins tossed so far, j--> heads tossed so far
    */
    double coinsMoreHeads(vector<double> probabilities)
    {
        size_t N = probabilities.size() + 1;
        vector<vector<double>> dp(N, vector<double>(N)); //i index j heads so far
        dp[0][0] = 1.0; //initial probability of having 0 heads with 0 tosses is 1.0
        for (size_t i = 1; i < N; i++)
        {
            double pHead = probabilities[i - 1];
            double pTail = 1.0 - pHead;
            dp[i][0] = dp[i - 1][0] * pTail; // j=0 heads means toss at i is tails
            for (size_t j = 1; j <= i; j++)
            {
                dp[i][j] = dp[i - 1][j - 1] * pHead + dp[i - 1][j] * pTail;
            }

        }
        double sum = 0.;
        for (size_t i = (N + 1) / 2; i < dp.size(); i++)
            sum += dp[N][i];

        return sum;
    }

    /*Cracking the coding interview questions (Chapter 8)*/

    //8.1 Triple jump
    int waysToReach(int n)
    {
        //push dp or pull dp?
        vector<int> dp(n + 1);
        dp[0] = 1;
        for (int i = 0; i < n; i++)
        {
            for (int j = 1; j <= 3; j++)
            {
                if (i + j < dp.size())
                    dp[i + j] += dp[i];
            }
        }
        return dp[n];
    }
    //8.2 Robot grid
    string robotPath(vector<vector<bool>>& grid)
    {
        //assuming start and end are valid points
        int rows = grid.size();
        int cols = grid[0].size();
        vector<vector<bool>> dp(rows + 1, vector<bool>(cols + 1));
        dp[rows][cols - 1] = true;
        dp[rows - 1][cols] = true;
        //robot starts at 0,0 and wants to arrive at r,c

        for (int i = rows - 1; i >= 0; i--)
        {
            for (int j = cols - 1; j >= 0; j--)
            {
                if (grid[i][j] == true)
                {
                    dp[i][j] = dp[i + 1][j] || dp[i][j + 1];
                }
                else
                    dp[i][j] = false;
            }
        }
        int i = 0, j = 0;
        string path = "";
        while (i < rows && j < cols)
        {
            if (dp[i + 1][j])
            {
                path += "D";
                i++;
            }
            else
            {
                path += "R";
                j++;
            }
        }
        return path;
    }
    //8.3 Magic index (don't understand why dp, binary search?)
    int magicIndex(vector<int>& A)
    {
        int low = 0;
        int high = A.size() - 1;

        while (low <= high)
        {
            int mid = low + (high - low) / 2;

            if (A[mid] == mid)
                return mid;
            //01234
            //12344
            if (A[mid] > mid)
            {
                low = mid + 1;
            }
            else
                high = mid - 1;
        }
        return -1;
    }

    //8.4 power set (return subsets of set)
    void helper(vector<vector<int>>& ans, vector<int>& current, vector<int>& nums, int idx)
    {
        ans.push_back(current);

        for (int i = idx; i < nums.size(); i++)
        {
            current.push_back(nums[i]);
            helper(ans, current, nums, i + 1);
            current.pop_back();
        }
    }
    vector<vector<int>> powerset(vector<int>& nums)
    {
        vector<vector<int>> ans;
        vector<int> current = {};
        helper(ans, current, nums, 0);
        return ans;
    }

    //8.5 Recursive multiply
    int recursiveMultiply(int x, int y)
    {
        if (x > y)
            swap(x, y);
        if (x == 1)
            return y;
        return recursiveMultiply(x - 1, y) + y;
    }

    //8.6 Towers of hanoi


    //Algorithms 1st Edition by Sanjoy Dasgupta, chapter 6
    /* 6.1.Maximum sum in a continious substring (Kadane's alg.) */
    int maxsumcontinuous(vector<int>& arr)
    {
        int sum = 0;
        int maxsum = 0;

        for (int i : arr) {
            sum += i;
            sum = max(sum, 0);
            maxsum = max(sum, maxsum);
        }
        return maxsum;
    }

    /* 6.2 Hotel stops
    * You are going on a long trip. You start on the road at mile post 0. 
    * Along the way there are n hotels, at mile posts a1,a2,...an, where each ai is measured from the starting point. 
    * You can choose which of the hotels you stop at. You must stop at the final hotel (at distance an), which is your destination.
    * You’d ideally like to travel 200 miles a day, but may not be possible-
    * If you travel x miles during a day, the penalty for that day is (200 − x)^2
    * You want to minimize the total penalty over all travel days, of the daily penalties.
    * Give an efficient algorithm that determines the optimal sequence of hotels at which to stop.
    */
    vector<int> optimalstops(const vector<int>& hotels)
    { 
        int n = hotels.size();
        vector<int> dp(n, INT_MAX);
        dp[0] = 0;
        vector<int> predecessor(n);

        for (int i = 1; i <n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                int pen = pow(200 - (hotels[i] - hotels[j]), 2);
                if (pen + dp[j] < dp[i])
                {
                    dp[i] = dp[j] + pen;
                    predecessor[i] = j;
                }
            }
        }
        vector<int> path;
        path.push_back(n-1);

        int prev = predecessor[hotels.size() - 1];
        while (prev != 0)
        {
            path.push_back(prev);
            prev = predecessor[prev];
        }
        reverse(path.begin(), path.end());
        return path;
    }



    /* 6.3
    *Yuckdonald’s is considering opening a series of restaurants.
    *The n possible locations are along a straight line
    *the distances of these locations to the start are in miles and in increasing order, m1, m2,...,mn.
    *The constraints are as follows: At each location, Yuckdonald’s may open at most one restaurant. 
    *The expected profit from opening a restaurant at location i is pi, where pi > 0 and i = 1, 2,..., n.
    *Any two restaurants should be at least k miles apart, where k is a positive integer.
    Give an efficient algorithm to compute the maximum expected total profit subject to the given
    constraints.
    */

    int maxRestaurantProfit(vector<int>& locations, vector<int>& profit, int k)
    {
        // dp(i) = maxprofit with restaurant at ith location + max(dp(j), where locations[j]+k < locations[i])
        int n = locations.size();
        vector<int> dp(n);
        int maxprofit = 0;
        for (int i = 0; i < n; i++)
        {
            dp[i] = profit[i];
            for (int j = 0; j < i; j++)
            {
                int dist = locations[i] - locations[j];
                if (dist >= k)
                {
                    dp[i] = max(dp[i], dp[j] + profit[i]);
                }
            }
            maxprofit = max(maxprofit, dp[i]);
        }
        return maxprofit;
    }

 /*6.4
*You are given a string of n characters s[1...n], which you believe to be a corrupted text document in which all punctuation has vanished ("itwasthebestoftimes").
*You wish to reconstruct the document using a dictionary, which is available in the form of a
Boolean function dict(.): for any string w, dict(w) ={ true if w is a valid word, false otherwise}
*Give a dynamic programming algorithm that determines whether the string s[.] can be
reconstituted as a sequence of valid words.
*The running time should be at most O(n^2),
assuming calls to dict take unit time.
*Reconstruct solution if valid.
*/

    string wordbreak(const unordered_set<string>& dict, string s)
    {
        //string s is valid if all prefix is valid s[0:j-1] is valid && s[j:] is word
        vector<bool> dp(s.size());
        for (int i = 0; i < s.size(); i++)
        {
            string substr1 = s.substr(0, i + 1);
            if (dict.count(substr1))
                dp[i] = true;
            
            for (int j = 0; j < i; j++)
            {
                if (dp[j] == true)
                {
                    int len = i - (j + 1) + 1;
                    string substr = s.substr(j + 1, len);
                    if (dict.count(substr))
                    {
                        dp[i] = true;
                        break;
                    }
                }
            }
        }

        //cats and dogs           dict = cat, cats, and, sand, dog, dogs
        //FFTT FFT FFTT 
        //reconstruct solution:
        vector<string> sol;
        int end = dp.size();
        for (int i = dp.size() - 1; i >= 0; i--)
        {
            if (dp[i] == true)
            {
                //try to break
                string subs = s.substr(i+1, end - i);
                if (dict.count(subs))
                {
                    sol.push_back(subs);
                    end = i;
                }
            }
        }
        string subs = s.substr(0, end+1);
        if (dict.count(subs))
            sol.push_back(subs);

        string ans = "";
        while (!sol.empty())
        {
            ans += sol.back();
            sol.pop_back();
            if (!sol.empty())
                ans += " ";
        }
        return ans;
    }


    //6.11
    /*
    Given two strings x = x1x2...xn and y = y1y2..ym, we wish to find the length of their longest common substring
    *Show how to do this in time O(mn).*/

    int longestCommonSubstring(string x, string y)
    {
        int n = x.size();
        int m = y.size();

        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        int maxlen = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (x[i] == y[j])
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                else
                    dp[i + 1][j + 1] = 0;

                maxlen = max(maxlen, dp[i + 1][j + 1]);
            }
        }
        return maxlen;
    }
};