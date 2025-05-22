from smolagents import Tool
import numpy as np
from scipy.stats import linregress

class DataAnalysisTool(Tool):
    """
    A class-based implementation of the Data Analysis Tool.
    Provides various numerical and statistical analysis functions.
    """

    name = "data_analysis_tool"
    description = "A tool for performing various numerical and statistical analysis tasks, such as mean, median, variance, standard deviation, sum, and linear regression."
    inputs = {
        "prompt": {
            "type": "string",
            "description": "A string containing the operation and data in a structured format. For example: 'operation: mean, data: {'numbers': [1, 2, 3]}'",
            "required": True,
        }
    }
    output_type = "object"

    def calculate_mean(self, numbers: list) -> float:
        """
        Calculates the mean of a list of numbers.
        """
        return np.mean(numbers)

    def calculate_median(self, numbers: list) -> float:
        """
        Calculates the median of a list of numbers.
        """
        return np.median(numbers)

    def calculate_variance(self, numbers: list) -> float:
        """
        Calculates the variance of a list of numbers.
        """
        return np.var(numbers)

    def calculate_standard_deviation(self, numbers: list) -> float:
        """
        Calculates the standard deviation of a list of numbers.
        """
        return np.std(numbers)

    def calculate_sum(self, numbers: list) -> float:
        """
        Calculates the sum of a list of numbers.
        """
        return np.sum(numbers)

    def perform_linear_regression(self, x: list, y: list) -> dict:
        """
        Performs linear regression on two lists of numbers.
        """
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        }

    def forward(self, prompt: str) -> dict:
        """
        Processes the prompt and returns the result of the requested operation.
        Args:
            prompt: A string containing the operation and data in a structured format.
                    For example: "operation: mean, data: {'numbers': [1, 2, 3]}"
        Returns:
            The result of the operation.
        """
        if not prompt:
            raise ValueError("The 'prompt' field is required.")

        # Parse the prompt to extract the operation and data
        try:
            # Example parsing logic: "operation: mean, data: {'numbers': [1, 2, 3]}"
            parts = prompt.split(", data: ")
            operation = parts[0].split("operation: ")[1].strip()
            data = eval(parts[1].strip())  # Convert the string representation of the dictionary to an actual dictionary
        except (IndexError, ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid prompt format: {prompt}. Expected format: 'operation: <operation>, data: <data>'") from e

        # Validate the operation and data
        if not operation or not data:
            raise ValueError("The prompt must contain both 'operation' and 'data' fields.")

        operations = {
            "mean": lambda d: self.calculate_mean(d["numbers"]),
            "median": lambda d: self.calculate_median(d["numbers"]),
            "variance": lambda d: self.calculate_variance(d["numbers"]),
            "std_dev": lambda d: self.calculate_standard_deviation(d["numbers"]),
            "sum": lambda d: self.calculate_sum(d["numbers"]),
            "regression": lambda d: self.perform_linear_regression(d["x"], d["y"]),
        }

        if operation not in operations:
            raise ValueError(f"Unsupported operation: {operation}")

        return operations[operation](data)