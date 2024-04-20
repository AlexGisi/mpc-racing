from math import sqrt

def euclidean(p1, p2):
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def midpoint(p1, p2, alpha=0.5):
        dx = (p2[0] - p1[0]) * alpha
        dy = (p2[1] - p1[1]) * alpha

        return dx+p1[0], dy+p1[1]

def interpolate(p1, p2, eps):
        if euclidean(p1, p2) <= eps:
            return p1, p2
        else:
            mid = midpoint(p1, p2)
            left_points = interpolate(p1, mid, eps)
            right_points = interpolate(mid, p2, eps)
            return left_points[:-1] + right_points
