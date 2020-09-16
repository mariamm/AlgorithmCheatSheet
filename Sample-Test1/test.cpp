#include "gtest/gtest.h"
#include "AlgorithmCheatSheet.h"

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

TEST(String, KMP_SUBSTRING1)
{
	string A = "Hello World";
	string B = "World";
	int index = kmpSubstring(A, B);
	EXPECT_EQ(index, 6);
}
TEST(String, KMP_SUBSTRING2)
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

TEST(Strings, KMP)
{
	string s = "Hello World";
	string b = "World";
	int i = kmpSubstring(s, b);
	EXPECT_EQ(i, 6);
	int j = kmpSubstring(s, "f");
	EXPECT_EQ(j, -1); 
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