class FeatureDict(dict):
    # 重写dict方法，让tensor作为key时，进判断值相同则为同一个value，不再判断地址

    def __getitem__(self, key):
        for k in self.keys():
            if key == k:
                return super().__getitem__(k)
        raise KeyError(key)

    def __setitem__(self, key, value):
        for k in self.keys():
            if key == k:
                super().__setitem__(k, value)
                return
        super().__setitem__(key, value)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default
