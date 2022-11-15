def IterativeFibo(n):
    f1 = 0
    f2 = 1
    for i in range(n):
        if i<2:
            print(i, end = ' ')
        else:
            f3 = f1 + f2   # 1, 1, 2, 3, 5
            f1 = f2
            f2 = f3
            print(f3, end = ' ')

def RecursiveFibo(n):
    if (n == 0 or n == 1):
        return n
    else:
        return(RecursiveFibo(n-1) + RecursiveFibo(n-2))


def main():
    n = int(input())

    print("Iterative Fibonacci: ")
    IterativeFibo(n)

    print("\nRecursive Fibonacci: ")
    for i in range(n):
        print(RecursiveFibo(i), end = ' ')

if __name__ == '__main__':
    main()