#pragma once

#include "Common.h"

///////////////////////////////////////////////
/////// Custom Data Structures ////////////////
///////////////////////////////////////////////
struct ListNode
{
    int val;
    ListNode* next;
    ListNode* previous; //for double linked list;
    ListNode()
    {
        val = 0;
        next = NULL;
        previous = NULL;
    }
    ListNode(int v)
    {
        val = v;
        next = NULL;
        previous = NULL;
    }
};

struct TreeNode
{
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode()
    {
        val = 0;
        left = NULL;
        right = NULL;
    }
    TreeNode(int v)
    {
        val = v;
        left = NULL;
        right = NULL;
    }
};

struct LinkedList
{
    ListNode* head;  //front
    ListNode* tail;	 //back

    ~LinkedList() { /* delete nodes from head to end, NULL ptr*/ }
    LinkedList() { head = NULL; tail = NULL; }
    void clear() {/*clear nodes from head to end, NULL ptr*/ }

    void popBack()
    {
        if (tail != NULL) {
            if (head == tail)
                head = NULL;

            ListNode* temp = tail;
            tail = tail->previous;

            if (tail != NULL)
                tail->next = NULL;

            delete temp;
            temp = NULL;
        }
    }
    void pushFront(ListNode* newHead)
    {
        if (head != NULL) {
            newHead->next = head;
            head->previous = newHead;
            head = newHead;
        }
        else {
            head = newHead;
            tail = newHead;
        }
    }
    //emplace_front
    void moveToHead(ListNode* n)
    {
        if (n != head) {
            n->previous->next = n->next;
            if (n != tail)
                n->next->previous = n->previous;
            else
                tail = n->previous;

            pushFront(n);
        }
    }
};

class Trie {
    unordered_map<char, Trie*> children;
    bool isCompleteWord;
    int frequency;
    Trie()
    {
        frequency = 0;
        isCompleteWord = false;
    }
    void addWord(string word)
    {
        if (word.empty()) return;
        Trie* current = this;
        for (char c : word)
        {
            current->frequency++;
            if (current->children.find(c) == current->children.end())
                current->children[c] = new Trie();

            current = current->children[c];
        }
        current->isCompleteWord = true;
    }
    bool isPrefix(string pref)
    {
        Trie* current = this;
        for (char c : pref)
        {
            if (current->children.find(c) == current->children.end())
                return false;
            current = current->children[c];
        }
        return true;
    }
    bool hasWord(string word)
    {
        Trie* current = this;
        for (char c : word)
        {
            if (current->children.find(c) == current->children.end())
                return false;
            current = current->children[c];
        }
        return current->isCompleteWord;
    }
};

//DSU: Disjoint - Set - Union
//Complexity O(log n) in worst case because of path compression
struct DSU
{
    unordered_map<int, int> parents;
    unordered_map<int, int> size; //union by size path compression
    unordered_map<int, int> rank; //union by rank path compression 

    void make_set(const vector<int>& vertices)
    {
        for (int v : vertices)
        {
            parents[v] = v;
            size[v] = 1;
            rank[v] = 0;
        }
    }
    int find_set(int n)
    {
        if (parents[n] == n)
            return n;
        parents[n] = find_set(parents[n]); //path compression
        return parents[n];
    }
    void dsu_union_by_size(int a, int b)
    {
        a = find_set(a);
        b = find_set(b);

        if (a != b) {
            //union by size
            if (size[b] > size[a])
                swap(a, b);

            parents[b] = a;
            size[a] += size[b];
        }
    }
    void dsu_union_by_rank(int a, int b)
    {
        a = find_set(a);
        b = find_set(b);

        if (a != b) {
            //union by rank
            if (rank[b] > rank[a])
                swap(a, b);

            parents[b] = a;
            if (rank[a] == rank[b])
                rank[a]++;
        }
    }
};

///////////////////////////////////////////////
/////// Data types Constructions //////////////
///////////////////////////////////////////////

ListNode* createLinkedList(const vector<int>& v)
{
    if (v.size() == 0) return NULL;

    ListNode* head = new ListNode(v[0]);
    ListNode* ptr = head;

    for (int i = 1; i < v.size(); i++)
    {
        ptr->next = new ListNode(v[i]);
        ptr = ptr->next;
    }
    return head;
}

vector<int> linkedListToVector(ListNode* head)
{
    vector<int> v;
    while (head)
    {
        v.push_back(head->val);
        head = head->next;
    }
    return v;
}
TreeNode* createTree(const vector<int>& v)
{
    if (v.size() == 0) return NULL;

    TreeNode* root = new TreeNode(v[0]);

    queue<TreeNode*> q;
    q.push(root);

    for (int i = 1; i < v.size(); i += 2)
    {
        TreeNode* p = q.front();
        q.pop();
        if (v[i] != -1)
        {
            p->left = new TreeNode(v[i]);
            q.push(p->left);
        }
        if (v[i + 1] != -1)
        {
            p->right = new TreeNode(v[i]);
            q.push(p->right);
        }
    }

    return root;
}


///////////////////////////////////////////////
/////// Pretty Print //////////////////////////
///////////////////////////////////////////////

void print2dVector(vector<vector<int>>& input)
{
    for (vector<int> v : input)
    {
        for (int i : v)
            cout << i << ", ";
        cout << endl;
    }
}
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

////////////////////////////////////////////
////////////   Strings        //////////////
////////////////////////////////////////////


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
*Space Complexity : AdjList O(V + E) AdjMat O(V ^ 2) -
* Time Complexity: O(V)
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
* Space Complexity : AdjList O(V + E) AdjMat O(V ^ 2)
* Time Complexity Time : O(V)
* BFS is used for Shortest Path.
* @param s is starting node.
* @param t is destination node(optional)
*/
void BFS(int s, map<int, vector<int>> adj, int t)
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
                //if (v == t)
                /*destination reached*/

                parent.insert({ v, u }); //assign u as parent
                level.insert({ v, i + 1 }); //set their level
                frontier.push(v); //add to queue 
            }
        }
    }
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
    }
    alternatively check added nodes in the output,should match the number of vertices
    if(sorted.size() != graph.size()) return vector<int>();
    */

    return sorted;
}


////////////////////////////////////////////
////////////   Backtracking   //////////////
////////////////////////////////////////////


/*3 Variant Examples for permutation function :
 * Given a collection of numbers, return all possible permutations.
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
*/
void permute(vector<int>& nums, vector<int>& curr, vector<vector<int>>& result) {
    if (curr.size() == nums.size()) {
        result.push_back(curr);
        return;
    }
    for (size_t i = 0; i < nums.size(); i++)
    {
        curr.push_back(nums[i]);
        int temp = nums[i];
        nums[i] = INT_MAX;
        permute(nums, curr, result);
        nums[i] = temp;
        curr.pop_back();
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
            swap(A[i][j], A[j][i]);
    if (CCW)
        reverse(A.begin(), A.end());
}

//Euclidean distance between two points in 2d
// sqrt( (x2-x1)^2 + (y2-y1)^2)
int distance(pair<int, int> p1, pair<int, int> p2)
{
    double deltax_squared = pow(p2.first - p1.first, 2);
    double deltay_squared = pow(p2.second - p1.second, 2);
    return (int)(sqrt(deltax_squared + deltay_squared));
}

//Dot product 
//Intuition: length of the projection of v onto w or vice versa. Dot product of perpendicular vectors is 0 
int dotProduct(vector<int> v, vector<int> w)
{
    assert(v.size() == w.size());
    size_t size = v.size();
    int dot = 0;
    for (size_t i = 0; i < size; i++)
    {
        dot += (v[i] * w[i]);
    }
    return dot;
}
//Cross product
//Intuition: The area of the parallelogram formed by v and w represented as a vector perpendicular to v x w
//alternative calculation   : v x w = det([v_x w_x]) 
//                                        [v_y w_y] 
vector<int> crossProduct3(vector<int> v, vector<int> w)
{
    vector<int> cross(3);
    cross[0] = v[1] * w[2] - v[2] * w[1];
    cross[1] = v[2] * w[0] - v[0] * w[2];
    cross[2] = v[0] * w[1] - v[1] * w[0];
    return cross;
}

//there is no cross product in 2d, however setting the z component of the inputs to zero will result
//to a vector with 0 in the x,y components and return the scalar value of z 
//also called the perp dot product & equivalent to the determinant of a square matrix
int crossProduct2(vector<int> v, vector<int> w)
{
    return v[0] * w[1] - v[1] * w[0];
}
//Determinant of a matrix
//Intuition: how much the area/volume of 1 (unit vectors) is stretched by this matrix.
//Det = 0 -> the transformation causes a reduction to a smaller dimension (the columns of the matrix are lineraly dependent)
//Negative det : inverts the orientation of space
int determinant2(vector<vector<int>> A)
{
    assert(A.size() == 2);
    assert(A[0].size() == 2);
    return A[0][0] * A[1][1] - A[1][0] * A[0][1]; //<-- cross product z value
}
int determinant3(vector<vector<int>> A)
{
    assert(A.size() == 3);
    assert(A[0].size() == 3);

    //Side note : a b c are the values of the cross product vector if first column of A is 1 1 1 
    int a = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
    int b = A[1][0] * (A[2][1] * A[0][2] - A[0][1] * A[2][2]);
    int c = A[2][0] * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);

    return a + b + c;
}
//Addition of two vectors
//Intuition: each vector represent moving towards that vector, up towards positive y, or down towards negative y
//left towareds negative x, and right towards positive x
//The sum of two vectors is the consequence of moving towards the first then the second vector or vice versa.
vector<int> vectorAddition(vector<int> v, vector<int>w)
{
    assert(v.size() == w.size());

    vector<int> ans(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        ans[i] = v[i] + w[i];
    }
    return ans;
}
//Multiplication of vectors (scaling) 
//Intuition: scaling the length of the vector
vector<int> vectorMultiplication(int s, vector<int> v)
{
    for (int i = 0; i < v.size(); i++)
        v[i] *= s;
    return v;
}
//2D Transformation Multiplication of a square matrix and a 2d vector 
//Intuition: imagining the columns of the matrix as transformed basis vectors i^ j^, we want to find where x lands in this transformation
//Note: transformation that keep preserve dot product=0 are called orthonormal (example: rigid transformation like rotation, translation & scaling)
// [a c] [x] = x[a] + y[c] = [xa + yc] 
// [b d] [y]    [b]    [d]   [xb + yd]
vector<int> transform2d(vector<vector<int>> A, vector<int> x)
{
    assert(x.size() == 2);
    assert(A.size() == 2);
    assert(A[0].size() == 2);

    vector<int> result(2);
    result[0] = x[0] * A[0][0] + x[1] * A[1][0];
    result[1] = x[0] * A[1][0] + x[1] * A[1][1];

    return result;
}
/* TODO
//Inverse of a matrix
//Intuition: it is the opposite transformation of the input matrix
vector<vector<int>> inverseOfMatrix(vector<vector<int>> A)
{
    return vector<vector<int>>();
}

//Solving linear of equations Ax = b or matlab A\b
//Intuition: find the original vector after a transformation happened.
vector<int> solveAslashB2(vector<vector<int>> A, vector<int> b)
{
    //Find the inverse of A if the determinant is not zero
    int d = determinant2(A);
    vector<vector<int>> A_;
    if (d != 0)
    {
        A_ = inverseOfMatrix(A);
    }
    else ???
    vector<int> x = transform2d(A_, b);
    return x;
}


//Eigenvector
//Intuition: a nonzero vector that after the transformation A doesn't lose its span but only gets scaled by eigenvalue lambda. For example, in a rotation transformation it's the rotation axis.
//Av = yv --> (A-yI)v = 0
void eigenvectors(vector<vector<int>> A, int &lambda1, int &lambda2, vector<int> &e1, vector<int>& e2)
{
}

//Convolution
//
void convolution2d(vector<vector<int>> A, vector<vector<int>> kernal, bool zeropadding)
{

}

*/
////////////////////////////////////////////
////////////   Mathematics    //////////////
////////////////////////////////////////////

//Factorial n! = 1*2*3*..*n (optional parameter, start*start+1*...*n) 
unsigned long long int factorial(int n, int start = 1)
{
    unsigned long long int fac = 1;
    for (int i = start + 1; i <= n; i++)
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
