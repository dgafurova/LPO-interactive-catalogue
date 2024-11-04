def NextDxNewton(dx, y0, y1, dxmax, tol):
    NewDx = 0
    if abs(y1)>=tol and y0 != y1:
        NewDx = dx
        if y1 != 0:  
            NewDx = max(min(dx * y1/(y0-y1),dxmax),-dxmax)

    return NewDx

class optimizer:
    dx = 0.001
    dxmax = 0.1
    xvals = []
    yvals = []
    step = 0
    tolerance = 0.000001
    output = False

    def __init__(self):
        self.xvals = []
        self.yvals = []
        self.step = 0

    def inCycle(self):
        if self.step <= 4:
            return False
        if abs(self.xvals[2] - self.xvals[4]) < 1e-16:
            return True
        return False
        
    def needNextStep(self):
        if len(self.yvals) <= 2: return True
        #if self.inCycle():
        #    if self.output:
        #        print('Cycled')
        #    return False
        if abs(self.xvals[0]-self.xvals[1]) <= self.tolerance:
            return False
        else:
            return True

    def nextX(self, x, y):
        self.step += 1
        
        i = 0
        #находим место для добавления точки, чтобы массив остался отсортированным
        while i < min(2, len(self.yvals)) and y > self.yvals[i]: i += 1
        self.xvals.insert(i, x)
        self.yvals.insert(i, y)

        xvals = self.xvals
        yvals = self.yvals
        
        if self.step == 1:
            return x+self.dx

        x1 = xvals[0]+NextDxNewton(xvals[1]-xvals[0], yvals[0], yvals[1], self.dx, 0)
        if self.step > 2:

            #print(xvals, yvals)
            a = yvals[0]/(xvals[0]-xvals[1])/(xvals[0]-xvals[2]) - yvals[1]/(xvals[0]-xvals[1])/(xvals[1]-xvals[2]) + yvals[2]/(xvals[1]-xvals[2])/(xvals[0]-xvals[2])
            #print(a)
            if a > 0:
                b = (yvals[0]-yvals[1])/(xvals[0]-xvals[1]) - a*(xvals[0]+xvals[1])
                xm = -b/2/a
                x1 = x + max(min(xm-x, self.dxmax), -self.dxmax)
        #if self.inCycle():
        #    x1 = (x1 + 2 * x) / 3

        return x1

    def getXY(self):
        return self.xvals[0], self.yvals[0]
    
    def getdx(self):
        return abs(self.xvals[0]-self.xvals[1])

