#include "gtest/gtest.h"
#include "AlgorithmCheatSheet.h"
#include "Common.h"
#include "Geometry.h"
#include "Recursion.h"
#include "DP.h"

TEST(Char_Conversions, UpperCase)
{
    for(int i=0; i<26; i++)
    {
        char c = 'a' + i;
        char C = 'A' + i;
        c = lowertoUpperCase(c);
        EXPECT_EQ(C, c);
    }
}
TEST(Char_Conversions, Lowercase)
{
    for (int i = 0; i < 26; i++)
    {
        char c = 'a' + i;
        char C = 'A' + i;
        C = uppertoLowerCase(C);
        EXPECT_EQ(C, c);
    }
}

TEST(Strings, KMP)
{
    string s = "Hello World";
    string b = "World";
    int i = kmpSubstring(s, b);
    EXPECT_EQ(i, 6);
    int j = kmpSubstring(s, "f");
    EXPECT_EQ(j, -1);
}
TEST(Strings, KMP_SUBSTRING1)
{
    string A = "Hello World";
    string B = "World";
    int index = kmpSubstring(A, B);
    EXPECT_EQ(index, 6);
}
TEST(Strings, KMP_SUBSTRING2)
{
    string A = "Hello World";
    string C = "Zorld";
    int index = kmpSubstring(A, C);
    EXPECT_EQ(index, -1);
}

TEST(DISABLED_Bucket_Sort, Float)
{ 
    float arrSorted[] = { 1.1, 1.25, 2.4, 2.748, 3.0, 3.02, 3.9, 4.59, 94.9 };
    float arr[] = { 4.59, 1.25, 2.748, 2.4, 1.1,3.02, 3.9,  3.0, 94.9 };
    bucketSort(arr, 4); 
}

TEST(Sort, stacksort)
{
    vector<int> values = { 1,4,5,2,3,7,6 };
    vector<int> sol = stacksort(values);
    vector<int> ans = { 7,6,5,4,3,2,1 };
    EXPECT_EQ(ans, sol);
}

TEST(Linked_List, CreateLinkedList)
{
    vector<int> v = { 1, 2, 3, 4, 5 };
    ListNode* head = createLinkedList(v);
    vector<int> vlist = linkedListToVector(head);

    EXPECT_EQ(v.size(), vlist.size());
    for (int i = 0; i < v.size(); i++)
        EXPECT_EQ(v[i], vlist[i]);
}

TEST(Linked_List, ReverseLinkedListIterative)
{
    vector<int> v = { 1, 2, 3, 4, 5 };
    ListNode* head = createLinkedList(v);
    head = reverseListIterative(head);
    vector<int> vlist = linkedListToVector(head);
    reverse(v.begin(), v.end());

    EXPECT_EQ(v.size(), vlist.size());
    for (int i = 0; i < v.size(); i++)
        EXPECT_EQ(v[i], vlist[i]);
}

TEST(Geometry, DotProduct1)
{
    vector<int> p1 = { 1, 2 };
    vector<int> p2 = { 5, 6 };
    int dotp = dotProduct(p1, p2);
    EXPECT_EQ(dotp, 17);
}
TEST(Geometry, DotProduct2)
{
    vector<int> p1 = { 1, 1 };
    vector<int> p2 = { 1, -1 };
    int dotp = dotProduct(p1, p2);
    EXPECT_EQ(dotp, 0);
}


TEST(SegmentTree, MinRangeValidInput)
{
    vector<int> v = { 1, 2, 4, 2, 6, 7, 3, -8, 3, 5, 8, 4,-9, 483, 2, 983, -123}; //17
    vector<pair<int, int>> ranges = { {0,16}, {8,16}, {5,7}, {2,4}, {0,5}, {9,13}, {14, 16} };
    SegmentTree t(v, SegmentTree::Type::Minimum);
    vector<int> min_output;
    vector<int> expected_output = { -123, -123, -8, 2, 1, -9, -123 };
    for (pair<int, int> p : ranges)
    {
        min_output.push_back(t.rangeQuery(p.first, p.second));
    }
    EXPECT_EQ(min_output.size(), expected_output.size());
    for (int i = 0; i < min_output.size(); i++)
    {
        EXPECT_EQ(min_output[i], expected_output[i]);
    }
} 
TEST(SegmentTree, SumRangeValidInput)
{
    vector<int> v = { 1, 2, 4, -2, 6, 7 }; //6
    vector<pair<int, int>> ranges = { {0,5}, {1,4}, {2,3}, {2,4}, {1,5}, {3,5} };
    SegmentTree t(v, SegmentTree::Type::Sum);
    vector<int> sum_output;
    vector<int> expected_output = {18, 10, 2, 8, 17, 11 };
    for (pair<int, int> p : ranges)
    {
        sum_output.push_back(t.rangeQuery(p.first, p.second));
    }
    EXPECT_EQ(sum_output.size(), expected_output.size());
    for (int i = 0; i < sum_output.size(); i++)
    {
        EXPECT_EQ(sum_output[i], expected_output[i]);
    }
}

TEST(Permutations, Variant1)
{
    vector<int> v = { 1, 2, 3 };
    vector<vector<int>> permutations;
    permute(v, 0, permutations);
    sort(permutations.begin(), permutations.end());
    vector<vector<int>> solution = { {1,2,3}, {1,3,2}, {2,1,3}, {2,3,1}, {3,1,2}, {3,2,1} };

    EXPECT_EQ(solution.size(), permutations.size());
    for (int i = 0; i < solution.size(); i++)
    {
        for (int j = 0; j < solution[0].size(); j++)
        {
            EXPECT_EQ(solution[i][j], permutations[i][j]);
        }
    }
}
TEST(Permutations, Variant2)
{
    vector<int> v = { 1, 2, 3 };
    vector<vector<int>> permutations;
    vector<int> current;
    permute(v, current, permutations);

    vector<vector<int>> solution = { {1,2,3}, {1,3,2}, {2,1,3}, {2,3,1}, {3,1,2}, {3,2,1} };

    EXPECT_EQ(solution.size(), permutations.size());
    for (int i = 0; i < solution.size(); i++)
    {
        for (int j = 0; j < solution[0].size(); j++)
        {
            EXPECT_EQ(solution[i][j], permutations[i][j]);
        }
    }
}

TEST(Permutations, Variant3)
{
    vector<int> v = { 1, 2, 3 };
    vector<vector<int>> permutations;
    vector<int> current;
    permute(v, permutations);

    vector<vector<int>> solution = { {1,2,3}, {1,3,2}, {2,1,3}, {2,3,1}, {3,1,2}, {3,2,1} };

    EXPECT_EQ(solution.size(), permutations.size());
    for (int i = 0; i < solution.size(); i++)
    {
        for (int j = 0; j < solution[0].size(); j++)
        {
            EXPECT_EQ(solution[i][j], permutations[i][j]);
        }
    }
}
TEST(ShortestPaths, BiBFS)
{
    vector<vector<int>> adjList = { {1,2}, {0,2,3},{0,1,3},{4},{3} }; 
    int dist = bidirectional_BFS(0, adjList, 4); 
    EXPECT_EQ(dist, 3);

}
TEST(ShortestPaths, Dijkstra)
{
    vector<vector<vector<int>>> adjList= { {{1, 4}, {2, 1}},
                                            {{3,1}},
                                            {{1, 2}, {3,5}},
                                            {{4,3}},
                                            {} };


    int dist = dijkstraShortestPath(adjList, 5);

    EXPECT_EQ(dist, 7);
    
}
TEST(ShortestPaths, BellmanFord)
{ 
    int N = 5;
    vector<vector<int>> edges;
    edges.push_back({ 0,1,6 });
    edges.push_back({ 0,4,7 });
    edges.push_back({ 1,2,5 });
    edges.push_back({ 1,3,-4 });
    edges.push_back({ 1,4,8 });
    edges.push_back({ 2,1,-2 });
    edges.push_back({ 3,2,7 });
    edges.push_back({ 3,0,2 });
    edges.push_back({ 4,3,9 });
    edges.push_back({ 4,2,-3 });
    int dist = bellmanFordShortestPath(edges, 5);

    EXPECT_EQ(dist, 7); 
}  

TEST(ShortestPaths, FloydWarshall)
{
    vector<vector<int>> adjMatrix = {
    { 0, 5, 0, 10 },
    { 0, 0, 3, 0 },
    { 0, 0, 0, 1 },
    { 0, 0, 0, 0 } 
    };

    vector<vector<int>> allpaths = floydWarshallShortestPath(adjMatrix);
}
TEST(Graphs, HierHolzerEuelerianPath)
{
    vector<vector<int>> adjList = {
    {},
    {2, 3},
    {2, 4, 4},
    {1, 2, 5},
    {3, 6},
    {6},
    {3},
    };
    vector<int> solution = hierholzerEulerianPath(adjList, 7);
    vector<int> test = { 1, 3, 5, 6, 3, 2, 4, 3, 1, 2, 2, 4, 6 };

    for (int i = 0; i < solution.size(); i++)
    {
        EXPECT_EQ(solution[i], test[i]);
    }
}

TEST(Geometry, OnDiagonal)
{
    //board 3x3 
    /*  ________
       |__|__|__|
       |__|__|__|
       |__|__|__|
    */
    vector<vector<pair<int, int>>> diagonals = {
        {{0,0},{1,1}}, {{0,0},{2,2}}, {{1,1},{2,2}},
        {{1,0},{2,1}}, {{0,1},{1,0}},
        {{0,1},{1,0}}, {{1,2},{2,1}},
        {{0,2},{1,1}}, {{0,2},{2,0}}, {{1,1},{2,0}},
    };
    vector<vector<pair<int, int>>> non_diagonals = {
        {{0,0},{0,1}}, {{0,0},{1,2}}, {{1,1},{1,0}},
        {{1,0},{1,1}}, {{0,1},{2,2}}, {{1,2},{0,2}}
    };

    for (vector<pair<int, int>> ps : diagonals)
    {
        EXPECT_TRUE(onDiagonal(ps[0], ps[1]));
    }
    for (vector<pair<int, int>> ps : non_diagonals)
    {
        EXPECT_FALSE(onDiagonal(ps[0], ps[1]));
    }
}

TEST(Math, Factorial)
{
     
    vector<unsigned long long int> nums = { 1,2,3,4,5,6 };
    vector<unsigned long long int> ans = { 1, 2, 6, 24, 120, 720 };

    for (int i = 0; i < nums.size(); i++)
    {
        auto n = factorial(nums[i]);
        EXPECT_EQ(n, ans[i]);
    } 
}

TEST(DynamicProgramming, Knapsack1)
{
    vector<int> weights = { 3,1,3,4,2 };
    vector<int> values = { 2,2,4,5,3 };
    int capacity = 7;
    int maxvalue = DP::knapsack(values, weights, capacity);
    int expectedMaxvalue = 10;

    EXPECT_EQ(maxvalue, expectedMaxvalue);
}

TEST(DynamicProgramming, Knapsack2)
{
    vector<int> weights = { 3,1,3,4,2 };
    vector<int> values = { 2,2,4,5,3 };
    int capacity = 7;
    int maxvalue = DP::knapsack1DnoRep(values, weights, capacity);
    int expectedMaxvalue = 10;

    EXPECT_EQ(maxvalue, expectedMaxvalue);
}

TEST(DynamicProgramming, Hotels_Optimal_Stops)
{
    vector<int> hotels = {0, 100, 250, 600, 700, 900};

    vector<int> path = DP::optimalstops(hotels);

    vector<int> opt = {2,3,5};
    EXPECT_EQ(path, opt);
}

TEST(DynamicProgramming, Wordbreak_1)
{
    unordered_set<string> dict = {"cat", "cats", "and", "sand", "dog", "dogs"};
    string s = "catsanddogs";

    string ans = DP::wordbreak(dict, s);
    unordered_set<string> validanswers = { "cats and dogs", "cat sand dogs" };

    EXPECT_EQ(1, validanswers.count(ans));
}
TEST(DynamicProgramming, LongestCommonSubstring)
{
    vector<vector<string>> tests =
    {
        {"abcd", "bcd"},
        {"abced", "bcd"},
        {"abc", "xyz"},
        {"xyabczbcd", "ebcd"}
    };
    vector<int> expected_answers = { 3, 2, 0, 3};
    for (int i=0; i<tests.size(); i++)
    {
        int lcss = DP::longestCommonSubstring(tests[i][0], tests[i][1]);
        EXPECT_EQ(expected_answers[i], lcss);
    }
}

TEST(DynamicProgramming, CoinChangeUnique)
{
    vector<int> denom = { 1,5,10,20 }; 
    EXPECT_TRUE(DP::coinChangePossible2(16, denom));
    EXPECT_TRUE(DP::coinChangePossible2(36, denom));
    EXPECT_FALSE(DP::coinChangePossible2(40, denom));
    vector<int> denom2 = { 19,18,10,5,3,2 };
    EXPECT_TRUE(DP::coinChangePossible2(20, denom2));
}
