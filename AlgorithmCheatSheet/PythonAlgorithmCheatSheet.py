######################
# Basics
######################
# arithmetic operations
import inline as inline
import matplotlib


def arithmetic_test():
    x = 3
    y = 5
    print(x + y)  # Note, there is no x++
    print(x - y)  # or x--
    print(x / y)  # floating division
    print(x // y)  # integer division
    print(x * y)  # regular multiplication
    print(x ** y)  # exponential


# boolean operators
def boolean_test():
    t = True  # notice capital letter
    f = False
    print(t and f)  # and operator
    print(t or f)  # or operator
    print(f != f)  # xor operator
    print(not t)  # not operator


# Strings and formatting
def string_formatting():
    h = 'Hello'
    w = 'World'
    print(h + ' ' + w)
    hw = '{} {}{}'.format(h, w, '!')
    print(hw)


# lists
def list_test():
    x = list(range(5))  # create a list with numbers from 0 to 4
    x[4] = 'foo'  # access element with [], allowed to assign different type
    f = x.pop()  # pop and return last element
    x.append(f)  # append element
    print(x[1:])  # slicing: print from index 1 to end
    print(x[0:2])  # slicing: print from index 0 to 2-1
    print(x[:-2])  # slicing: backward print from 0 until end-2 index
    print(x[:])  # print entire list


# object oriented: classes
class Greeter:
    # constructor
    def __init__(self, name):
        self.name = name

    def greet(self, loud):
        if not loud:
            print('Hello, ', self.name)
        else:
            print('HELLO, ', self.name.upper(), '!!!!!')


################################
# Algorithms
#################################
# Quicksort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# print(quicksort([3,6,8,10,1,2,1]))

###################
# numpy syntax
###################
import numpy as np


# Add vector to each row of matrix example
def add_vector_matrix():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3 matrix
    v = np.array([1, 2, 3])
    y = np.empty_like(x)  # empty matrix with x dimensions
    # looping each row 0 to 2
    for i in range(3):
        y[i, :] = x[i, :] + v

    # using a tiled matrix
    vv = np.tile(v, (3, 1))  # tile vector v 3 times on top of each other
    y2 = x + vv  # should be the same as y

    # using broadcasting
    y3 = x + v


# numpy a range, create an array in range of arange(start, stop, step)
# start & step are optional parameters
def arange_test():
    x = np.arange(0, 11, 2)
    print(x)


##############################
# Mat plot
###############################
import matplotlib.pyplot as plt


def sinus_plot_test():
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Sine', 'Cosine'])
    plt.show()


sinus_plot_test()
