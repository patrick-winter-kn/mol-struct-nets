class Rasterizer:

    def __init__(self, factor, padding, min_x, max_x, min_y, max_y, square=False):
        self.factor = factor
        self.padding = padding * factor
        self.offset_x = -min_x
        self.offset_y = -min_y
        self.size_x = max_x - min_x
        self.size_y = max_y - min_y
        if square:
            if self.size_x < self.size_y:
                self.offset_x += abs(self.size_x - self.size_y) / 2
            else:
                self.offset_y += abs(self.size_x - self.size_y) / 2
            self.size_x = max(self.size_x, self.size_y)
            self.size_y = self.size_x
        self.size_x = int(round(self.size_x * self.factor) + 1 + 2 * self.padding)
        self.size_y = int(round(self.size_y * self.factor) + 1 + 2 * self.padding)

    def apply(self, x, y):
        x = int(round((x + self.offset_x) * self.factor) + self.padding)
        y = int(round((y + self.offset_y) * self.factor) + self.padding)
        return x, y
