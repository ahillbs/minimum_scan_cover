class xContainer():
    def __init__(self, x=None):
        self.x = x
    def __repr__(self):
        return str(self.x)
    def __str__(self):
        return self.__repr__()


class Multidict:
    def __init__(self, param=None):
        self._dict = {}
        if not param:
            return
        for k in param:
            if not isinstance(param[k], list):
                obj = [param[k]]
            else:
                obj = param[k]
            self._dict[k] = obj

    def update(self, dictionary):
        for k in dictionary:
            self.__setitem__(k, dictionary[k])

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if not isinstance(value, list):
            obj = [value]
        else:
            obj = value
        if key in self._dict:
            for o in obj:
                self._dict[key].append(o)
        else:
            self._dict[key] = obj

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)
    
    def keys(self):
        return self._dict.keys()

    def get_ordered_keys(self):
        return sorted(self._dict.__iter__())
