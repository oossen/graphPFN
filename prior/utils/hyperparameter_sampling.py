from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.distributions as dist


class DistributionSampler(ABC):
    """Abstract base class for distribution samplers."""
    
    @abstractmethod
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        """Sample a value from this distribution."""
        pass


class FixedSampler(DistributionSampler):
    """Sampler that always returns a fixed value."""
    
    def __init__(self, value: Any):
        self.value = value
    
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        return self.value
    

class TorchDistributionSampler(DistributionSampler):
    """Wrapper for torch.distributions samplers."""
    
    def __init__(self, distribution: dist.Distribution):
        self.distribution = distribution
    
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        if generator is not None:
            # Use the generator for sampling
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = self.distribution.sample()
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            value = self.distribution.sample()
        
        # Convert to appropriate Python type
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            else:
                return value.tolist()
        return value
    

class CategoricalSampler(DistributionSampler):
    """Categorical (choice) sampler using torch.distributions."""
    
    def __init__(self, choices: List[Any], probabilities: Optional[List[float]] = None):
        self.choices = choices
        if probabilities is not None:
            if len(probabilities) != len(choices):
                raise ValueError("Length of probabilities must match length of choices")
            self.categorical = dist.Categorical(torch.tensor(probabilities))
        else:
            # Uniform probabilities
            uniform_probs = torch.ones(len(choices)) / len(choices)
            self.categorical = dist.Categorical(uniform_probs)
    
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        if generator is not None:
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                idx = self.categorical.sample()
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            idx = self.categorical.sample()
        
        return self.choices[int(idx.item())]


class DiscreteUniformSampler(DistributionSampler):
    """Discrete uniform distribution sampler (integers) using torch."""
    
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
    
    def sample(self, generator: Optional[torch.Generator] = None) -> int:
        if generator is not None:
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = torch.randint(self.low, self.high + 1, (1,))
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            value = torch.randint(self.low, self.high + 1, (1,))
        
        return int(value.item())
    

DISTRIBUTION_FACTORIES = {
    "fixed": lambda params: FixedSampler(params["value"]),
    "uniform": lambda params: TorchDistributionSampler(
        dist.Uniform(low=params["low"], high=params["high"])
    ),
    "normal": lambda params: TorchDistributionSampler(
        dist.Normal(loc=params["mean"], scale=params["std"])
    ),
    "lognormal": lambda params: TorchDistributionSampler(
        dist.LogNormal(loc=params["mean"], scale=params["std"])
    ),
    "exponential": lambda params: TorchDistributionSampler(
        dist.Exponential(rate=params["lambd"])
    ),
    "gamma": lambda params: TorchDistributionSampler(
        dist.Gamma(concentration=params["alpha"], rate=params["beta"])
    ),
    "beta": lambda params: TorchDistributionSampler(
        dist.Beta(concentration1=params["alpha"], concentration0=params["beta"])
    ),
    "categorical": lambda params: CategoricalSampler(
            params["choices"], params.get("probabilities")
    ),
    "discrete_uniform": lambda params: DiscreteUniformSampler(params["low"], params["high"]),
}


def build_samplers(config: Dict[str, Any],
                   config_name: str,
                   expected_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build sampler objects from the configuration."""
    samplers = {}

    for param_name, param_config in config.items():
        # Check if parameter is known
        if expected_params is not None and param_name not in expected_params:
            raise ValueError(f"Unknown {config_name} hyperparameter: {param_name}")

        # Handle shorthand fixed value notation
        if "value" in param_config and "distribution" not in param_config:
            sampler = FixedSampler(param_config["value"])
                    
        elif "distribution" in param_config:
            dist_type = param_config["distribution"]
            # Regular parameter: create sampler
            if dist_type not in DISTRIBUTION_FACTORIES:
                raise ValueError(f"Unknown distribution type: {dist_type}")

            # Get distribution parameters
            dist_params = param_config.get("distribution_parameters", {})
            if dist_type == "fixed":
                if "value" not in param_config:
                    raise ValueError(f"Fixed distribution for {param_name} requires 'value' key")
                dist_params = {"value": param_config["value"]}

                # Create sampler
            try:
                sampler = DISTRIBUTION_FACTORIES[dist_type](dist_params)
            except Exception as e:
                raise ValueError(f"Error creating sampler for {config_name}.{param_name}: {e}")
        else:
            raise ValueError(f"Configuration for {config_name}.{param_name} must specify 'distribution' or 'value'")

        samplers[param_name] = sampler

    # Check that all required parameters are specified
    if expected_params is not None:
        required_params = set(expected_params.keys())
        provided_params = set(config.keys())
        missing_params = required_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required {config_name} parameters: {missing_params}")

    return samplers
    

def sample_parameters(samplers: Dict[str, Any],
                      config_name: str,
                      generator: torch.Generator,
                      expected_types: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sample parameters from samplers with type validation."""
    sampled_params = {}

    for param_name, sampler in samplers.items():
        value = sampler.sample(generator)

        # Type validation
        if expected_types is not None:
            expected_type = expected_types[param_name]
            if not isinstance(value, expected_type):
                # Try to convert if possible
                if isinstance(expected_type, type) and expected_type is int and isinstance(value, float):
                    value = int(value)
                elif isinstance(expected_type, type) and expected_type is float and isinstance(value, int):
                    value = float(value)
                elif isinstance(expected_type, type) and expected_type is tuple and isinstance(value, list):
                    value = tuple(value)
                elif isinstance(expected_type, tuple) and type(None) in expected_type:
                    # Optional parameter - check if value is one of the allowed types
                    allowed_types = [t for t in expected_type if t is not type(None)]
                    if value is not None and not isinstance(value, tuple(allowed_types)):
                        raise ValueError(f"Parameter {config_name}.{param_name} has invalid type. Expected one of {expected_type}, got {type(value)}")
                else:
                    raise ValueError(f"Parameter {config_name}.{param_name} has invalid type. Expected {expected_type}, got {type(value)}")

        sampled_params[param_name] = value

    return sampled_params