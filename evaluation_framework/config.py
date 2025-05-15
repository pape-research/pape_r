import os


class Configurable:
    """
    The `Configurable` class helps bring a standardized hierarchical strategy for retrieving configuration values.
    A `Configurable` variable can be set by using a function parameter, an environment variable or a default value.

    When trying to retrieve a value - given a parameter value - the following steps are taken:
    1. Return the parameter value if it is not None. Else proceed to next step.
    2. Return the value of the environment variable specified at creation time. Else proceed to next step.
    3. Return the default value specified at creation time.
    """

    def __init__(self, environment_variable_name, default_value):
        self.environment_variable_name = environment_variable_name
        self.default_value = default_value

    def value(self, param):
        if param:
            return param
        elif self.environment_variable_name in os.environ:
            return os.getenv(self.environment_variable_name)
        else:
            return self.default_value