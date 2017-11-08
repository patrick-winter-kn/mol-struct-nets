import math


class Transformer:

    def __init__(self, min_x, max_x, min_y, max_y):
        self.center_x = min_x + (max_x - min_x) / 2
        self.center_y = min_y + (max_y - min_y) / 2

    def apply(self, x, y, flip=False, rotation=0):
        if flip:
            x = 2 * self.center_x - x
        if rotation != 0:
            x, y = Transformer.rotate((self.center_x, self.center_y), (x, y), rotation)
        return x, y

    @staticmethod
    def rotate(origin, point, angle):
        angle = math.radians(angle)
        origin_x, origin_y = origin
        point_x, point_y = point
        rotated_x = origin_x + math.cos(angle) * (point_x - origin_x) - math.sin(angle) * (point_y - origin_y)
        rotated_y = origin_y + math.sin(angle) * (point_x - origin_x) + math.cos(angle) * (point_y - origin_y)
        return rotated_x, rotated_y
