import math
import cmath
from smolagents import tool

@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers.
    Args:
        a (float): The first number.
        b (float): The second number.
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """
    Subtract one number from another.
    Args:
        a (float): The number to subtract from.
        b (float): The number to subtract.
    """
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    Args:
        a (float): The first number.
        b (float): The second number.
    """
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """
    Divide one number by another.
    Args:
        a (float): The numerator.
        b (float): The denominator.
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """
    Get the modulus of two numbers.
    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return a % b

@tool
def power(a: float, b: float) -> float:
    """
    Raise a number to a power.
    Args:
        a (float): The base number.
        b (float): The exponent.
    """
    return a ** b

@tool
def square_root(a: float) -> float | complex:
    """
    Get the square root of a number.
    Args:
        a (float): The number to get the square root of.
    Returns:
        float or complex: The square root. Returns complex if a is negative.
    """
    if a >= 0:
        return math.sqrt(a)
    return cmath.sqrt(a)

@tool
def factorial(n: int) -> int:
    """
    Get the factorial of a number.
    Args:
        n (int): The number to get the factorial of.
    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(n)

@tool
def combination(n: int, k: int) -> int:
    """
    Get the number of combinations (n choose k).
    Args:
        n (int): The total number of items.
        k (int): The number of items to choose.
    """
    return math.comb(n, k)

@tool
def permutation(n: int, k: int) -> int:
    """
    Get the number of permutations (n permute k).
    Args:
        n (int): The total number of items.
        k (int): The number of items to arrange.
    """
    return math.perm(n, k)

@tool
def log(a: float, base: float = math.e) -> float:
    """
    Get the logarithm of a number with a given base.
    Args:
        a (float): The number to take the logarithm of.
        base (float): The logarithm base (default is natural log).
    Raises:
        ValueError: If a is not positive.
    """
    if a <= 0:
        raise ValueError("Logarithm is not defined for non-positive numbers.")
    return math.log(a, base)

@tool
def sin(x: float) -> float:
    """
    Get the sine of an angle in radians.
    Args:
        x (float): The angle in radians.
    """
    return math.sin(x)

@tool
def cos(x: float) -> float:
    """
    Get the cosine of an angle in radians.
    Args:
        x (float): The angle in radians.
    """
    return math.cos(x)

@tool
def tan(x: float) -> float:
    """
    Get the tangent of an angle in radians.
    Args:
        x (float): The angle in radians.
    """
    return math.tan(x)

@tool
def mean(numbers: list) -> float:
    """
    Get the mean (average) of a list of numbers.
    Args:
        numbers (list): The list of numbers.
    """
    return sum(numbers) / len(numbers)

@tool
def median(numbers: list) -> float:
    """
    Get the median (middle value) of a list of numbers.
    Args:
        numbers (list): The list of numbers.
    """
    nums = sorted(numbers)
    n = len(nums)
    if n % 2 == 1:
        return nums[n // 2]
    else:
        return (nums[n // 2 - 1] + nums[n // 2]) / 2

@tool
def variance(numbers: list) -> float:
    """
    Get the variance of a list of numbers.
    Args:
        numbers (list): The list of numbers.
    """
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)

@tool
def standard_deviation(numbers: list) -> float:
    """
    Get the standard deviation of a list of numbers.
    Args:
        numbers (list): The list of numbers.
    """
    return variance(numbers) ** 0.5

@tool
def total(numbers: list) -> float:
    """
    Get the sum of a list of numbers.
    Args:
        numbers (list): The list of numbers.
    """
    return sum(numbers)

@tool
def linear_regression(x: list, y: list) -> dict:
    """
    Perform linear regression on two lists of numbers.
    Args:
        x (list): The list of independent variable values.
        y (list): The list of dependent variable values.
    Returns:
        dict: Regression results including slope, intercept, r_value, p_value, and std_err.
    """
    import numpy as np
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err
    }