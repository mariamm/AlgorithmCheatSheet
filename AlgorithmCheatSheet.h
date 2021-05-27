#pragma once

#include "Common.h"


////////////////////////////////////////////
////////////   Strings        //////////////
////////////////////////////////////////////

//Char conversions
char intToChar(int i)
{
    assert(i >= 0 && i <= 9);
    return i + '0';
}

int charToInt(char c)
{
    assert(c >= '0' && c <= '9');
    return c - '0';
}

char uppertoLowerCase(char UC)
{
    assert(UC >= 'A' && UC <= 'Z');
    return UC - 'A' + 'a';
}

char lowertoUpperCase(char lc)
{
    assert(lc >= 'a' && lc <= 'z');
    return lc - 'a' + 'A';
}

bool isVowel(char c)
{
    string vowels = "AEIOUaeiou";
    return vowels.find(c) != string::npos;
}
/* KMP Algorithm(find substring index)
 * Knuth Morris Pratt
 * If string B is substring of A, return starting index of substring
*/
int kmpSubstring(string A, string B)
{
    int prev_start = 0; 
    size_t i = 0;
    size_t j = 0;
    while (i < A.size() && j < B.size())
    {
        if (A[i] == B[j])
        { 
            i++;
            j++;
        }
        //restart with next pointer
        else
        {
            j = 0;
            i = ++prev_start; 
        }
    }
    if (j == B.size())
        return prev_start;
    return -1;
}

////////////////////////////////////////////
////////////   Sorting        //////////////
////////////////////////////////////////////

/*Bucket sort
* Function to sort arr[] of size n using bucket sort
* Time Complexity: O(n+k) average case, O(n^2) worst case
* Space Complexity: O(n)
*/
void bucketSort(float arr[], int n)
{
    /* 1) Create n empty buckets */
    vector<vector<float>> b(n);
    /* 2) Put array elements in different buckets */
    for (int i = 0; i < n; i++)
    {
        int bi = n * (int)arr[i]; // Index in bucket 
        b[bi].push_back(arr[i]);
    }

    /* 3) Sort individual buckets */
    for (int i = 0; i < n; i++)
        sort(b[i].begin(), b[i].end());

    /* 4) Concatenate all buckets into arr[] */
    int index = 0;
    for (int i = 0; i < n; i++)
        for (size_t j = 0; j < b[i].size(); j++)
            arr[index++] = b[i][j];
}

vector<int> stacksort(vector<int>& values)
{
    vector<int> ans;
   /*
   
    v = 15
    a = 72
    t = 3
   */
    while (!values.empty())
    {
        if (ans.empty() || values.back() < ans.back())
        {
            ans.push_back(values.back());
            values.pop_back();
        }
        else
        {
            int t = values.back();
            values.pop_back();
            do
            {
                values.push_back(ans.back());
                ans.pop_back();
            } while (!ans.empty() && ans.back() < t);
            ans.push_back(t);
        }
    }
    return ans;
}
//Linked List
//Reverse list recursive
ListNode* reverseListRecursive(ListNode* A, ListNode*& head)
{
    if (A == NULL)
        return NULL;
    if (A->next == NULL)
    {
        head = A;
        return A;
    }
    ListNode* newNext = reverseListRecursive(A->next, head);
    newNext->next = A;
    A->next = NULL;

    return A;
}

//Iterative:
ListNode* reverseListIterative(ListNode* currentNode)
{
    ListNode* previousNode = NULL;
    ListNode* nextNode;

    while (currentNode != NULL) {
        nextNode = currentNode->next;
        currentNode->next = previousNode;
        previousNode = currentNode;
        currentNode = nextNode;
    }
    //loop ends when currentNode == NULL, therefore, reveresed head is previousNode. 
    return previousNode;
}


////////////////////////////////////////////
////////////   Binary search   /////////////
////////////////////////////////////////////

/* Binary search in Array
* Find index of given key in a sorted array.
* Time Complexity O(log n)
*/
int findKeyInSortedArray(vector<int>& sortedArray, int value)
{
    int low = 0;
    int high = (int)sortedArray.size() - 1;
    while (low < high)
    {
        int mid = low + (high - low) / 2; //avoid overflow

        if (sortedArray[mid] == value) //found index
            return mid;

        if (sortedArray[mid] > value) //search left
            high = mid - 1;
        else //search right
            low = mid + 1;
    }
    //not found
    return -1;
}

//Binary search sqrt(A)
int sqrt_binarySearch(int A)
{
    if (A == 0) return 0;
    int low = 1;
    int high = A;

    while (low < high)
    {
        long long mid = low + (high - low + 1) / 2;
        if (mid * mid == A)
            return (int)mid;
        if (mid * mid < A)
            low = (int)mid + 1;
        else
            high = (int)mid - 1;
    }
    return low;
}
//Problems
/* Painter Partition problem: minimum time to finish painting.
 * Binary search in rotated array
*/

//Binary search tree
//Delete a node from a BST
TreeNode* findMinimum(TreeNode* root)
{
    TreeNode* current = root;
    while (current->left) {
        current = current->left;
    }
    return current;
}
TreeNode* deleteNode(TreeNode* root, int key)
{
    if (root == NULL)
        return NULL;

    if (root->val > key)
        root->left = deleteNode(root->left, key);
    else if (root->val < key)
        root->right = deleteNode(root->right, key);

    else //root->val == key 
    {
        if (root->left && root->right)
        {
            TreeNode* toDelete = findMinimum(root->right); //or find maximum in left subtree 
            root->val = toDelete->val;
            root->right = deleteNode(root->right, root->val);
        }
        else if (root->left || root->right)
        {
            TreeNode* toDelete = root;
            if (root->left)
                root = root->left;
            else
                root = root->right;
            delete toDelete;
        }
        else
        {
            delete root;
            root = NULL;
        }
    }
    return root;
}

//level order traversal for trees 
void levelOrderTraversal(TreeNode* root)
{
    int current_max_level = 0;
    queue<TreeNode*> q;
    q.push(root);
    int level = 0;
    while (!q.empty())
    {
        level++;
        //loop nodes in this level
        for (size_t i = q.size(); i > 0; --i)
        {
            TreeNode* n = q.front(); q.pop();
            /*do something*/
            //add children to be processed in next level
            if (n->left)  q.push(n->left);
            if (n->right) q.push(n->right);
        }
    }
}

//Diameter
//Diameter of a tree is the maximum length of a path between two nodes

int treeDepth(TreeNode* root) {
    return (root == NULL) ? 0 : treeDepth(root->left) + treeDepth(root->right) + 1;
}

//Get the maximum path(counting edges) between two nodes(diameter) of a tree, 
//not passing through the root necessarily.

pair<int, int> treeDepthAndDiameter(TreeNode* root) {
    if (root == NULL) return { 0,0 };

    pair<int, int> left = treeDepthAndDiameter(root->left);
    pair<int, int> right = treeDepthAndDiameter(root->right);

    int maxdepth = max(left.first, right.first) + 1;
    int maxdiameter = max({ left.second, right.second, maxdepth });

    return { maxdepth , maxdiameter };
}

////////////////////////////////////////////
////////////   Graphs         //////////////
////////////////////////////////////////////

/* DFS Depth first search graph traversal
* AdjList O(V + E) AdjMat O(V ^ 2)
* Time Complexity Time : O(V+E)
* Space Complexity : O(V)
*/
//Recursive call. Can be made to return a bool value if we are checking cycles. 
//Iterative implementation would use a stack!
bool dfs_visit(vector<vector<int>>& adj, int s, vector<bool>& parent, vector<bool>& finished)
{
    //startTime[s] =  //TODO
    for (int v : adj[s]) {
        //Node is a child that is not visited = tree edge
        if (parent[v] == false) {
            //Set parent to true, or s if it's a map
            parent[v] = true;
            //Recursive call to visit v
            if (!dfs_visit(adj, v, parent, finished))
                return false; // for cycles
        }

        //Node is visited and finished = forward edge or cross edge  
        //An edge(s,v) is a cross edge, if startTime[s]>startTime[v]. 
       // else if (finished[v])
            /*optional do something*/
        //Node is visited but not finished = backward edge (cycle)
        //else if (!finished[v] && /* parent[v] != vertex for undirected graphs */)
            /*optional do something*/
    }
    finished[s] = true;
    return true;
}
//Main dfs function that calls every vertex in the set (in case of an unconnected vertex) 
void dfs(vector<vector<int>>& adj, vector<int>VerticesSet) {
    //parent aka visited vector. can be a map to know the parent pointers for
    //topological sort!
    vector<bool> parent(VerticesSet.size(), false);
    //optional finished flag vector used to detect cycles
    vector<bool> finished(VerticesSet.size(), false);
    //optional start time vector used to detect cross edges
    vector<int> startTime(VerticesSet.size()); //TODO

    //loop to visit all vertices
    for (int s : VerticesSet) {
        if (!parent[s]) {
            parent[s] = true;
            dfs_visit(adj, s, parent, finished);
        }
    }
}

/*BFS Breadth first search graph traversal
* AdjList O(V + E) AdjMat O(V ^ 2)
* Time Complexity Time : O(V+E)
* Space Complexity : O(V)
* BFS is used for Shortest Path.
* Implementation too verbose, from MIT course
* @param s is starting node.
* @param t is destination node
* @return distance between s and t
*/
int BFS(int s, map<int, vector<int>> adj, int t)
{
    queue<int> frontier; //Queue to add children of the visited node
    frontier.push(s);
    unordered_map<int, int> level;//Optional to keep track of level
    level.emplace(s, 0);
    unordered_map<int, int> parent; // aka visited, map is used to extract shorted path
    parent.emplace(s, -1);

    while (!frontier.empty())
    {
        int u = frontier.front();
        frontier.pop(); //remove from queue

        //Print u, do something with u
        /*do something with u*/

        int i = level[u];//level of node
        //Loop adjacent vertices
        for (int v : adj[u])
        {
            //If adjacent nodes are not visited (not in parent)
            if (parent.count(v))
            {
                if (v == t)
                    return i+1;

                parent.insert({ v, u }); //assign u as parent
                level.insert({ v, i + 1 }); //set their level
                frontier.push(v); //add to queue 
            }
        }
    }
    return -1; //destination not found
}
/*Bidirectional Breadth first search graph traversal 
* Start the search on both the start and destination
* Stop when they merge
* if s has k adjacent vertices, then the distance to t is k^d
* if we start the search in both direction, the search will only 
* take 2 * k^(d/2), which is a factor of k^d better
* @param s is starting node.
* @param t is destination node
*/
int bidirectional_BFS(int s, vector<vector<int>> adj, int t)
{
    unordered_map<int, int> visited1;
    unordered_map<int, int> visited2;

    queue<int> q1;
    queue<int> q2;
    q1.push(s);
    q2.push(t);
    visited1[s] = 0;
    visited2[t] = 0;
    while (!q1.empty() && !q2.empty())
    {
        //visit next nodes in 1
        s = q1.front(); q1.pop();
        for (int n : adj[s])
        {
            if (visited1.find(n) != visited1.end()) //already visited
            {
                continue;
            }
            if (visited2.find(n) != visited2.end()) //bidirectionally visited, return ans
            {
                return visited2[n] + visited1[s] + 1;
            }
            else
            {
                visited1[n] = visited1[s] + 1; // set distance
                q1.push(n);
            }
        }
        t = q2.front(); q2.pop();
        for (int n : adj[t])
        {
            if (visited2.find(n) != visited2.end()) //already visited
            {
                continue;
            }
            if (visited1.find(n) != visited1.end()) //bidirectionally visited, return ans
            {
                return visited2[t] + visited1[n] + 1;
            }
            else
            {
                visited2[n] = visited2[t] + 1; // set distance
                q2.push(n);
            }
        }
    }
    return -1; //destination not found
}

/* Bellman-Ford
* @param edgesList: List of all edges in the graph [u,v,w] edge between u,v with weight w
* @param N: number of vertices
* @return: shortest distance from 0 to N-1
* Runtime: O(V*E)
*/
int bellmanFordShortestPath(vector<vector<int>> edgesList, int N)
{
    //Steps: 
    // Create distance array d & initialize with inf value 
    // Create predecessor array p & initialize with -1
    // Set start distance to 0
    // For every edge, relax distance and set predecessor
    // Then, for every edge find negative cycle to discard result
    const int INF = 1e9;
    vector<int> d(N, INF);
    d[0] = 0; //d[Start]=0
    vector<int> p(N, -1);
    for(int i=0; i<N; i++) {
        for (vector<int> e : edgesList) {
            if (d[e[1]] > d[e[0]] + e[2]) {
                d[e[1]] = d[e[0]] + e[2];
                p[e[1]] = e[0];
            }
        }
    }
    for(int i=0; i<N; i++) {
        for (vector<int> e : edgesList) {
            if (d[e[1]] > d[e[0]] + e[2])
                d[e[1]] = -INF; //negative cycle
        }
    }
    return d[N - 1];
}
/* Dijkstra 
 * Lazy implementation using a regular priority queue (eager implementation uses indexed priority queue)
 * @param adjList : weighted adjacency list of the graph (only positive weights)
 * @param N: vertices, 0 indexed
 * @return: shortest distance from node 0 to N-1
 */

int dijkstraShortestPath(vector<vector<vector<int>>> adjList, int N)
{
    assert(adjList.size() == N+1 && "Adjacency List must equal vertices N");
    vector<int> dist(N+1, INT_MAX);
    //using priority queue in lazy implementation
    priority_queue<pair<int, int>, vector<pair<int,int>>, greater<pair<int,int>>> pq; //pair: distance, vertex
    //initial state
    dist[0] = 0; //starting at vertex 0
    pq.emplace(0, 0);
    
    while (!pq.empty())
    {
        //poll best distance pair
        pair<int, int> p = pq.top();
        pq.pop();
        if (dist[p.second] < p.first) //node visited already
            continue;
        //else visit node
        for (vector<int> neighbor : adjList[p.second])
        {
            int distToNeighbor = p.first + neighbor[1];
            dist[neighbor[0]] = min(dist[neighbor[0]], distToNeighbor);
            pq.emplace(distToNeighbor, neighbor[0]);
        }
    } 
    return dist[N]; 
}
//Topological sort of a DAG (directed acyclic graph)
vector<int> kahnsort(vector<vector<int>>& graph)
{
    vector<int> indegree(graph.size());
    for (vector<int> n : graph)
    {
        for (int v : n)
        {
            indegree[v]++;
        }
    }
    vector<int> sorted;

    queue<int> q;
    for (int i = 0; i < (int)indegree.size(); i++)
    {
        if (indegree[i] == 0)
            q.push(i);
    }

    while (!q.empty())
    {
        int v = q.front(); q.pop();
        sorted.push_back(v);

        for (int n : graph[v])
        {
            indegree[n]--;
            if (indegree[n] <= 0)
            {
                q.push(n);
            }
        }
    }
    /*for(int i : indegree){
        if(i > 0)
            return vector<int>(); //graph has a cycle,
    }*/
    //alternatively check added nodes in the output,should match the number of vertices
    if(sorted.size() != graph.size()) 
        return vector<int>();
    
    return sorted;
}

void modifiedDfs(vector<vector<int>>& adjList, vector<int>& solution, int v, vector<int> &outdegrees)
{
    //visit all edges in adjList[v];
    //infinite loop
    /*for (int n : adjList[v])
    {
        if (outdegrees[v] == 0)
            break;
        modifiedDfs(adjList, solution, n, outdegrees);
        outdegrees[v]--;
    }*/

    while (outdegrees[v] != 0)
    {
        int next_edge = adjList[v][--outdegrees[v]];
        modifiedDfs(adjList, solution, next_edge, outdegrees);
    }
    solution.push_back(v);
}

/* Hierholzer Algorithm (Finding Euerlian Path in a directed graph)
 * Eulerian path: a path which visits every edge in a graph exactly once
 * Alg. description: modified DFS to visit all edges in the graph by using outdegree edges to keep track of unvisited edges.
 * @param adjList: Graph
 * @param N: number of vertices
 * @return: vector of vertices depicting the path, returns empty vector if path doesn't exist.
 */
vector<int> hierholzerEulerianPath(vector<vector<int>>& adjList, int N)
{
    //Step one, verify that graph has a eulerian path
    //Conditions: indegrees & outdegrees of every vertex is equal, or two vertex where one has additional outdegree and the other an additional indegree

    vector<int>indegrees(N);
    vector<int>outdegrees(N);

    for (int i = 0; i < N; i++)
    {
        outdegrees[i] = adjList[i].size();
        for (int n : adjList[i])
        {
            indegrees[n]++;
        }
    }

    int startNode = 0;
    int diffout = 0;
    int diffin = 0;
    for (int i = 0; i < N; i++)
    {
        int diff = outdegrees[i] - indegrees[i];
        if (diff == 0)
            continue;
        else if (diff == -1)
            diffin++;
        else if (diff == 1)
        {
            startNode = i;
            diffout++;
        }
        else
            return vector<int>();
    }

    auto valid = [=]()
    {
        return (diffin == 0 && diffout == 0) //circuit
            || (diffin == 1 && diffout == 1);
    };

    if (!valid())
        return vector<int>();


    //Step 2, make sure we have a startNode with outgoing edge if not found previously with the unique outdegree +1
    if (startNode == 0)
    {
        for (int i = 0; i < outdegrees.size(); i++)
        {
            if (outdegrees[i] > 0)
            {
                startNode = i;
                break;
            }
        }
    }

    //Step 3, run modified dfs 
    vector<int> solution;
    modifiedDfs(adjList, solution, startNode, outdegrees);

    //the solution must be reversed, because the algorithm should add the vertex in the front of the solution. I used push_back in modifiedDfs for optimization
    reverse(solution.begin(), solution.end());

    return solution;
}
/* Prim's Minimum Spanning Tree algorithm
 * @param edges: vector of vectors containing e edges in the graph. vector e[0] start node, e[1] end node, e[2] weight
 * @param N: Number of vertices
 * @return: total weights/costs of the minimum spanning tree
 */

int primsMinimumSpanningTree(vector<vector<int>> &edges, int N)
{
    vector<vector<pair<int, int>>> graph(N + 1);
    for (auto& edge : edges) {
        graph[edge[0]].push_back(make_pair(edge[2], edge[1]));
        graph[edge[1]].push_back(make_pair(edge[2], edge[0]));
    }

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;//minheap edge, adjacency list
    unordered_set<int> visited;
    q.push(make_pair(0, 1));
    
    int totalCost = 0;
    vector<vector<vector<int>>> minTree(N + 1);

    while (!q.empty())
    {
        auto curr = q.top();
        q.pop();
        if (visited.find(curr.second) != visited.end()) continue;

        visited.insert(curr.second);

        totalCost += curr.first;
        for (auto& adj : graph[curr.second])
        {
            if (visited.find(adj.second) != visited.end()) continue;
            q.push(adj);
        }
    }

    return totalCost;
}

/* Krushkal's Minimum Spanning Tree algorithm
 * @param edges: vector of vectors containing e edges in the graph. vector e[0] start node, e[1] end node, e[2] weight
 * @param N: vertices
 * @return: total weights/costs of the minimum spanning tree
 */
int krushkalsMinimumSpanningTree(vector<vector<int> >& edges, int N)
{
    auto customsort = [](vector<int> a, vector<int> b)
    {
        return a[2] < b[2];
    };
    sort(edges.begin(), edges.end(), customsort);

    DSU dsu;
    dsu.make_set(N);

    int weights = 0;
    vector<vector<vector<int>>> minTree(N+1);
    for (vector<int> e : edges)
    {
        int x = e[0];
        int y = e[1];
        int w = e[2];

        //already a uni
        if (dsu.find_set(x) == dsu.find_set(y))
            continue;
        else
        {
            minTree[x].push_back({ y, w });
            minTree[y].push_back({ x, w });

            weights += w;
            dsu.union_by_size(x, y);
        }
    }
    return weights;
}

////////////////////////////////////////////
////////////   Geometry       //////////////
////////////////////////////////////////////

/* 2D to 1D mapping
 * Using row major order (left to right then down)
 * @param coordinates first row, second col
 * @param width matrix width
 */
int matrix2dTo1dMaping(pair<int, int> coordinates, int width)
{
    return width * coordinates.first + coordinates.second;
}
pair<int, int> linearTo2dMapping(int index, int width)
{
    pair<int, int> coordinates;
    coordinates.first = index / width;
    coordinates.second = index % width;
    return coordinates;
}

/*Rotate image by 90 degrees inplace clockwise or counterclockwise
* @param A 2D input image passed by reference to be modified
* @param CCW flag for counterclockwise
*/
void rotateImage(vector<vector<int>>& A, bool CCW = false)
{
    //counter clockwise : swap first, then reverse
    if (!CCW)
        reverse(A.begin(), A.end()); 
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = i + 1; j < A[0].size(); j++)
        {
            swap(A[i][j], A[j][i]);
            print2dVector(A);
        }
    if (CCW)
        reverse(A.begin(), A.end());
}


vector<vector<int>> matrix_transpose(vector<vector<int>>& matrix)
{
    if (matrix.empty() || matrix[0].empty())
        return vector<vector<int>>();

    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<vector<int>> transpose(cols, vector<int>(rows));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            transpose[j][i] = matrix[i][j];
        }
    }
    return transpose;
}

vector<vector<int>> matrix_reflect_cols(vector<vector<int>>& matrix)
{
    if (matrix.empty() || matrix[0].empty())
        return vector<vector<int>>();

    vector<vector<int>> reflect(matrix.size(), vector<int>(matrix[0].size()));

    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            reflect[i][j] = matrix[i][matrix[0].size() - j - 1];
        }
    }

    return reflect;
}

vector<vector<int>> matrix_reflect_rows(vector<vector<int>>& matrix)
{
    if (matrix.empty() || matrix[0].empty())
        return vector<vector<int>>();

    vector<vector<int>> reflect(matrix.size(), vector<int>(matrix[0].size()));

    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            reflect[i][j] = matrix[matrix.size() - i - 1][j];
        }
    }

    return reflect;
}

vector<vector<int>> rotateImage2(vector<vector<int>>& matrix, bool CCW = false)
{
    vector<vector<int>> rotated = matrix_transpose(matrix);
    if (CCW)
        rotated = matrix_reflect_rows(rotated);
    else
        rotated = matrix_reflect_cols(rotated);
    return rotated;
}
////////////////////////////////////////////
////////////   Mathematics    //////////////
////////////////////////////////////////////

//Factorial n! = 1*2*3*..*n (optional parameter, start*start+1*...*n) 
unsigned long long int factorial(int n, int start = 1)
{
    unsigned long long int fac = 1;
    for (int i = start; i <= n; i++)
        fac *= i;
    return fac;
}

//Binomial coefficients (N choose k) 
long long nCk(int n, int k)
{
    if (k<0 || k>n) return 0;
    long long numerator = factorial(n, n - k);
    long long denominator = factorial(k);
    return numerator / denominator;
}


////////////////////////////////////////////
//////////// Linear time algorithms  ///////
////////////////////////////////////////////


/* Boyer Moore Majority Vote 
 * Find majority element in linear time, constant space
 * @param nums input vector
 * @param candidate return majority element if true
 * @return true if majority element appearing more than 50% or false if no majority is found
 */
bool boyerMooreMajorityVote(const vector<int>& nums, int &candidate)
{
    int count = 0;
    //find candidate
    for (int i : nums)
    {
        if (count == 0)
            candidate = i;

        if (candidate == i)
            count++;
        else
            count--;
    }

    // verify validity of candidate
    count = 0;
    for (int i : nums)
    {
        if (i == candidate)
            count++;
    }
    if (count > nums.size() / 2)
        return true;
    else
        return false; //invalid, no majority element
}

/* Kadane's Algorithm (Maximum Sum Subarray)
 * In computer vision, maximum-subarray algorithms are used on bitmap images to detect the brightest area in an image.
 * @return the maximum sum in subarray
 * @param nums input array which may include negative numbers
 */

int maximumSumSubarray(vector<int>& nums)
{
    int current = 0;
    int ans = INT_MIN;
    for (int i : nums)
    {
        current += i;
        ans = max(ans, current); //this is evaluated first to include negative numbers
        current = max(current, 0); 
    }
    return ans;
}
/* Modified Kadane's Algorithm ( Start and End of Maximum Sum Subarray)
 * @return pair first start index of subarray, pair second last index of subarray;
 * @param nums input array which may include negative numbers

pair<int, int> maximumSubarrayIndices(vector<int>& nums)
{


} 
*/

/*
Probability / Random distribution


std::uniform_real_distribution<double> unifrom(lowerBound, UpperBound);
std::default_random_engine re;

*/

/*
* Nested weighted sum? Not sure what to call it.
* Simplified: multiplying the current value with its depth, find the maximum if you can start at any level
* The idea is to use a prefix sum (iterating backwards) and (re)add it at every iteration to simulate the depth multiplication
* Example problems: LC 339. Nested List Weight Sum, LC 1402. Reducing Dishes
*/ 
int maximumStartingAtAnyDepth(vector<int> arr)
{
    int prefixsum = 0;
    int currentsum = 0;
    int maxsum = 0;

    for (int i = arr.size() - 1; i >= 0; i--)  {
        prefixsum += arr[i];
        currentsum += prefixsum; //with every iteration prefix sum gets added again with the first added element being added i times
        maxsum = max(maxsum, currentsum);
    }
    return maxsum;
}
