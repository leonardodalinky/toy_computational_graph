from value import Scalar


def example1():
    print("example1:")
    x = Scalar(10)
    y = Scalar(2)
    r = x + 2 * y
    r.backward()
    print(f"x={x}, y={y}, r=x+2*y={r}")
    print(f"=> x.grad={x.grad}, y.grad={y.grad}")
    print()


def example2():
    print("example2:")
    x = Scalar(10)
    (x * x).backward()
    print(f"x={x}, r=x*x={x * x}")
    print(f"=> x.grad={x.grad}")
    print()


def example3():
    print("example3:")
    x = Scalar(10)
    r = x * (x + 1)
    r.backward()
    print(f"x={x}, r=x*(x+1)={r}")
    print(f"=> x.grad={x.grad}")
    print()


def example4():
    print("example4:")
    x = Scalar(8)
    y = Scalar(4)
    r = x / y
    r.backward()
    print(f"x={x}, y={y}, r=x/y={r}")
    print(f"=> x.grad={x.grad}, y.grad={y.grad}")
    print()


def example5():
    print("example5:")
    x = Scalar(3)
    r = 1 / (x * x + 1)
    r.backward()
    print(f"x={x}, r=1/(x*x+1)={r}")
    print(f"=> x.grad={x.grad}")
    print()


def example6():
    print("example6:")
    x = Scalar(8)
    y = Scalar(3)
    r = (x * x + 1) / (y * y - 1)
    r.backward()
    print(f"x={x}, y={y}, r=(x*x+1)/(y*y-1)={r}")
    print(f"=> x.grad={x.grad}, y.grad={y.grad}")
    print()


if __name__ == "__main__":
    example1()
    example2()
    example3()
    example4()
    example5()
    example6()
