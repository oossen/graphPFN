"""Python wrapper to provide easy config access via dot notation."""

from ConfigSpace import ConfigurationSpace


class DotDict(dict):
    """dot notation access to dictionary attributes."""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.get_dict_recursive()

    def __setstate__(self, state):
        self.update(make_dotdict(state))

    def get_dict_recursive(self) -> dict:
        return {
            k: (v.get_dict_recursive() if isinstance(v, DotDict) else v)
            for k, v in self.items()
        }

    def merge(self, other, *, with_overwrite: bool = True):
        result = self.get_dict_recursive()

        if other is None:
            return make_dotdict(result)

        for key, value in other.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = make_dotdict(result[key]).merge(
                    value, with_overwrite=with_overwrite
                )
            elif with_overwrite or key not in result:
                result[key] = value

        return make_dotdict(result)

    def contains(self, subset: dict) -> bool:
        for key, sub_value in subset.items():
            if key not in self:
                return False
            super_value = self[key]
            if isinstance(sub_value, dict) and isinstance(super_value, dict):
                if not super_value.contains(sub_value):
                    return False
            elif isinstance(sub_value, list):
                for i_ in range(len(sub_value)):
                    sub_value_ele = sub_value[i_]
                    super_value_ele = super_value[i_]
                    if isinstance(super_value_ele, dict):
                        super_value_ele.contains(sub_value_ele)
                    elif super_value_ele != sub_value_ele:
                        return False
            elif isinstance(sub_value, ConfigurationSpace):
                continue
            elif super_value != sub_value:
                return False
        return True


def make_dotdict(data=None) -> DotDict:
    """Transforms a dict into a DotDict, including nested dicts."""
    if data is None:
        data = {}
    data = DotDict(data)
    for key in data:
        if type(data[key]) is dict:
            data[key] = make_dotdict(data[key])

    return data
