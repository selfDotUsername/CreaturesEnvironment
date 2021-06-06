import pygame
import numpy as np
import copy
import pickle
import time

#"cm4v12" da iawn

loadFrom =0
saveTo ="aaa"#"cm4v13"
keepStats = 1

pRateTo0 = 0.95

fps = 30000

level = 0
hiddelLayerShape = [5]
training = 1

maxCreatures = 30
foods = 60
poisons = 0
foodstoBreed = 1

creaturesLidarSize = 50
creaturesStartingEnergy = 100
creaturesnLidar = 2
creaturesRecurrence = 3
creaturesInitialRate = 1
creaturesv, creaturesdv, creaturesSize = 10, 0.1, 10

foodSize = 5
foodColor = (0, 255, 0)
poisonColor = (255, 0, 0)

lidarSizeRate = 1
creaturesRateRate = 1

foodEnergy = 100
window_width = 1000
window_height = 1000


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Network:
    def __init__(self, shape, activation):
        self.layers = []
        self.nlayers = len(shape) - 1
        self.activation = activation()
        for i in range(self.nlayers):
            self.layers.append(Layer_Dense(shape[i], shape[i + 1]))

    def forward(self, inputs):
        self.layers[0].forward(inputs)
        self.activation.forward(self.layers[0].output)
        for i in range(self.nlayers - 1):
            self.layers[i + 1].forward(self.activation.output)
            self.activation.forward(self.layers[i + 1].output)
        self.output = self.activation.output
        # for i in range(len(self.output)):
        #    for j in range(len(self.output[i])):
        #        if self.output[i][j] > 0:
        #            self.output[i][j] = 1

    def child(self, rate):
        for i in range(self.nlayers):
            for j in range(len(self.layers[i].weights)):
                for k in range(len(self.layers[i].weights[j])):
                    if np.random.rand() > sigmoid(rate):
                        self.layers[i].weights[j][k] *= random(-2, 2)
                        if self.layers[i].weights[j][k] > 1000:
                            self.layers[i].weights[j][k] = 1000
                        if self.layers[i].weights[j][k] < -1000:
                            self.layers[i].weights[j][k] = -1000
            for j in range(len(self.layers[i].biases)):
                for k in range(len(self.layers[i].biases[j])):
                    if np.random.rand() > sigmoid(rate):
                        self.layers[i].biases[j][k] *= random(-2, 2)
                        if self.layers[i].biases[j][k] > 1000:
                            self.layers[i].biases[j][k] = 1000
                        if self.layers[i].biases[j][k] < -1000:
                            self.layers[i].biases[j][k] = -1000


pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("food creatures")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)


def sigmoid(_x):
    return 1 / (1 * np.e ** (-_x))


def random(r0, r1):
    return np.random.rand() * (r1 - r0) + r0


def zeros(n):
    answer = []
    for i in range(n):
        answer.append(0)
    return answer


def isL1Collisoin(X0, X1, d):
    return np.abs(X0[0] - X1[0]) < d and np.abs(X0[1] - X1[1]) < d


def isPointAhead(x0, y0, x1, y1, c, s):
    # print("arctan=",np.arctan2([y1 - y0], [x1 - x0]))
    # print("arctanabs=", np.abs((np.arctan2([y1 - y0], [x1 - x0]) - t) % (2 * np.pi)),np.abs((np.arctan2([y1 - y0], [x1 - x0]) - t))% (2 * np.pi))
    # print("arctanresult = ", np.abs((np.arctan2([y1 - y0], [x1 - x0]) - t) % (2 * np.pi)) < 1)
    #return np.abs((np.arctan2([y1 - y0], [x1 - x0]) - t)) % (2 * np.pi) < 1
    return c*(x1-x0)+s*(y1-y0) > 0

def squaredistance(x0, y0, x1, y1):
    return (x0 - x1) ** 2 + (y0 - y1) ** 2


def isLidarCollision(X0, X1, t, dmax, sin, cos):
    x0, y0, x1, y1 = X0[0], X0[1], X1[0], X1[1]
    c = cos
    s = sin
    termsInCommon = (x1 - x0) * c + (y1 - y0) * s
    xbar = x0 + c * termsInCommon
    ybar = y0 + s * termsInCommon
    d2 = squaredistance(xbar, ybar, x1, y1)
    if d2 <= dmax and isPointAhead(x0, y0, xbar, ybar, c, s):
        l2 = squaredistance(xbar, ybar, x0, y0)
        return l2
    return None


def update_fps():
    fps = str(int(clock.get_fps()))
    fps_text = font.render("fps:" + fps, 1, pygame.Color("coral"))
    return fps_text


class creature:
    highScore = 0
    bestModel = 0
    creatures = []
    nCreatures = 0
    highScore = 0
    foodsCollected = 0
    poisonsCollected = 0
    totalCreatures = 0

    def __init__(self, position, d, v, dv, size, color, nlidar, lidarsize, startingEnergy, model, recurrence, rate):
        self.n = len(creature.creatures)
        creature.creatures.append(self)
        creature.nCreatures = len(creature.creatures)
        self.score = 0

        self.X = position
        self.d = d
        self.v = v
        self.dv = dv
        self.size = size
        self.color = color
        self.nlidar = nlidar
        self.lidarSize = lidarsize
        self.lidarSize2 = self.lidarSize ** 2

        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)

        self.mUp = 0
        self.mDown = 0
        self.mLeft = 0
        self.mRight = 0

        self.energy = startingEnergy

        self.model = model
        self.recurrence = zeros(recurrence)
        self.rate = rate
        self.networkInput = []
        self.networkOutput = []

    def draw(self):
        if creature.creatures[0] == self:
            pygame.draw.circle(screen, (0, 0, 255), (self.X[0], self.X[1]), self.size)
        else:
            if self.recurrence[0] == 0:
                pygame.draw.circle(screen,(255,0,0) , (self.X[0], self.X[1]), self.size)
            else:
                pygame.draw.circle(screen, (0, 255, 0), (self.X[0], self.X[1]), self.size)
        pygame.draw.circle(screen, self.color, (self.X[0] + self.vcosd, self.X[1] + self.vsind), self.size / 2)


    def up(self):
        self.X[0] += self.vcosd
        self.X[1] += self.vsind

    def down(self):
        self.X[0] -= self.vcosd
        self.X[1] -= self.vsind

    def left(self):
        self.d -= self.dv
        self.d = self.d % (2 * np.pi)
        # print(self.d)
        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)

    def right(self):
        self.d += self.dv
        self.d = self.d % (2 * np.pi)
        self.vsind = self.v * np.sin(self.d)
        self.vcosd = self.v * np.cos(self.d)

    def move(self):
        if self.mUp == 1:
            self.up()
        if self.mDown == 1:
            self.down()
        if self.mLeft == 1:
            self.left()
        if self.mRight == 1:
            self.right()
        if self.X[0] > window_width - self.size:
            self.X[0] = window_width - self.size
        if self.X[0] < self.size:
            self.X[0] = self.size
        if self.X[1] > window_height - self.size:
            self.X[1] = window_height - self.size
        if self.X[1] < self.size:
            self.X[1] = self.size

    def lidarSingle(self, d):
        _lidarOutput = []
        _detections = []
        _sin = np.sin(d)
        _cos = np.cos(d)
        for i in range(len(creature.creatures)):
            if creature.creatures[i] != self:
                _detections.append(isLidarCollision(self.X, creature.creatures[i].X, d, self.lidarSize2, _sin, _cos))
        _foodDetections = []
        for i in range(len(food.foods)):
            _foodDetections.append(isLidarCollision(self.X, food.foods[i].X, d, self.lidarSize2, _sin, _cos))
        _min = None
        _mini = None
        for i in range(len(_detections)):
            if _detections[i] is not None:
                if _min == None:
                    _min = _detections[i]
                    _mini = i

                if _detections[i] < _min:
                    _min = _detections[i]
                    _mini = i
        _minFoods = None
        _miniFoods = None
        for i in range(len(_foodDetections)):
            if _foodDetections[i] is not None:
                if _minFoods == None:
                    _minFoods = _foodDetections[i]
                    _miniFoods = i

                if _foodDetections[i] < _minFoods:
                    _minFoods = _foodDetections[i]
                    _miniFoods = i
        if _min is None and _minFoods is None:
            _x1, _y1 = self.X[0], self.X[1]
            if d == 0:
                return [window_width - _x1, 0, 0, 0, 0]

            if d == np.pi:
                return [_x1, 0, 0, 0, 0]

            if d == np.pi / 2:
                return [window_height - _y1, 0, 0, 0, 0]

            if d == 3 * np.pi / 2:
                return [_y1, 0, 0, 0, 0]

            _m = np.tan(d)
            _detections = []

            if d < np.pi / 2:
                if _m * (window_width - _x1) + _y1 < window_height:
                    # right
                    return [squaredistance(_x1, _y1, window_width, _m * (window_width - _x1) + _y1), 0, 0, 0, 0]
                else:
                    # top
                    return [squaredistance(_x1, _y1, _x1 - _y1 / _m, 0), 0, 0, 0, 0]
            if d < np.pi:
                if _y1 - _m * _x1 <= window_height:
                    # left
                    return [squaredistance(_x1, _y1, 0, _y1 - _m * _x1), 0, 0, 0, 0]
                else:
                    # top
                    return [squaredistance(_x1, _y1, _x1 - _y1 / _m, 0), 0, 0, 0, 0]
            if d < 3 * np.pi / 2:
                if _y1 - _m * _x1 >= 0:
                    # left
                    return [squaredistance(_x1, _y1, 0, _y1 - _m * _x1), 0, 0, 0, 0]
                else:
                    # bottom
                    return [squaredistance(_x1, _y1, (window_height - _y1) / _m + _x1, window_height), 0, 0, 0, 0]
            if _x1 - _y1 / _m < window_width:
                #
                return [squaredistance(_x1, _y1, (window_height - _y1) / _m + _x1, window_height), 0, 0, 0, 0]
            else:
                return [squaredistance(_x1, _y1, window_width, _m * (window_width - _x1) + _y1), 0, 0, 0, 0]
            # print(d)
            # print(_detections)
        if _minFoods is None:
            return [_min, creature.creatures[_mini].color[0], creature.creatures[_mini].color[1],
                    creature.creatures[_mini].color[2], creature.creatures[_mini].size]
        elif _min is None:
            return [_minFoods, food.foods[_miniFoods].color[0], food.foods[_miniFoods].color[1],
                    food.foods[_miniFoods].color[2], food.foods[_miniFoods].size]
        if min(_min, _minFoods) is _min:
            return [_min, creature.creatures[_mini].color[0], creature.creatures[_mini].color[1],
                    creature.creatures[_mini].color[2], creature.creatures[_mini].size]
        else:
            return [_minFoods, food.foods[_miniFoods].color[0], food.foods[_miniFoods].color[1],
                    food.foods[_miniFoods].color[2], food.foods[_miniFoods].size]

    def delete(self):
        if len(creature.creatures) != 1:
            for i in range(self.n + 1, len(creature.creatures)):
                creature.creatures[i].n -= 1
            if self.n < len(creature.creatures):
                creature.creatures.pop(self.n)

    def updateEnergy(self):
        self.energy -= 1+ 2**(self.energy/100)
        if self.energy < 1:
            self.delete()

    def clone(self):
        if len(creature.creatures) < maxCreatures:
            _newModel = copy.deepcopy(self.model)
            _newModel.child(self.rate)
            creature.totalCreatures += 1
        # print(self.rate)
            _randomx = np.random.rand() * window_width
            _randomy = np.random.rand() * window_height
            _randomd = np.random.randn() * 2 * np.pi
            print(self.rate, self.lidarSize)
            _newRate = self.rate + random(-0.1, 0.1) * creaturesRateRate
            if np.random.rand() > pRateTo0:
                _newRate = self.score/10
            creature([_randomx, _randomy], _randomd, 10, 0.1, 10, (100, 100, 100), creaturesnLidar,
                     self.lidarSize + random(-1, 1) * lidarSizeRate, creaturesStartingEnergy,
                     _newModel, creaturesRecurrence, _newRate)

    def brainCombo(self,p1,p2):
        for i in range(self.model.nlayers):
            for j in range(len(self.model.layers[i].weights)):
                for k in range(len(self.model.layers[i].weights[j])):
                    if np.random.rand() > 0.5:
                        self.model.layers[i].weights[j][k] = copy.deepcopy(p1.model.layers[i].weights[j][k])
                    else:
                        self.model.layers[i].weights[j][k] = copy.deepcopy(p2.model.layers[i].weights[j][k])
            for j in range(len(self.model.layers[i].biases)):
                for k in range(len(self.model.layers[i].biases[j])):
                    if np.random.rand() > 0.5:
                        self.model.layers[i].biases[j][k] = copy.deepcopy(p1.model.layers[i].biases[j][k])
                    else:
                        self.model.layers[i].biases[j][k] = copy.deepcopy(p2.model.layers[i].biases[j][k])


    def clone2(self, otherCreature):
        if len(creature.creatures) < maxCreatures:
            _newModel = copy.deepcopy(self.model)
            _newModel.child(self.rate)
            creature.totalCreatures += 1
        # print(self.rate)
            _randomx = np.random.rand() * window_width
            _randomy = np.random.rand() * window_height
            _randomd = np.random.randn() * 2 * np.pi
            print(self.rate, self.lidarSize)
            _newRate = self.rate + random(-0.1, 0.1) * creaturesRateRate
            if np.random.rand() > pRateTo0:
                _newRate = self.score/10
            _newCreature = creature([_randomx, _randomy], _randomd, 10, 0.1, 10, (100, 100, 100), creaturesnLidar,
                 self.lidarSize + random(-1, 1) * lidarSizeRate, creaturesStartingEnergy,
                 _newModel, creaturesRecurrence, _newRate)
            _newCreature.brainCombo(self,otherCreature)
            _newCreature.model.child(_newCreature.rate)


    def colisionDetection(self):
        for i in range(len(food.foods)):
            if i >= len(food.foods):
                break
            if isL1Collisoin(self.X, food.foods[i].X, self.size + food.foods[i].size):
                # print("aaaa")
                if food.foods[i].kind == 1:
                    self.score += 1
                    if self.score % foodstoBreed == 0:
                        self.clone2(creature.creatures[0])
                    self.energy += foodEnergy
                    creature.foodsCollected += 1
                if food.foods[i].kind == 0:
                    self.delete()
                    creature.poisonsCollected += 1
                food.foods.pop(i)
        for i in range(len(creature.creatures)):
            if i >= len(creature.creatures):
                break
            if creature.creatures[i] is not self:
                if isL1Collisoin(self.X, creature.creatures[i].X, self.size + creature.creatures[i].size):
                    creature.creatures[i].delete()
                    self.delete()
                    #pass
                    #if self.recurrence[0] == 0 and creature.creatures[i].recurrence[0] == 0:
                    #    self.clone2(creature.creatures[i])
                    #    self.energy -= 50
                    #    creature.creatures[i].energy -= 50
                    #elif self.recurrence[0] != 0 and creature.creatures[i].recurrence[0] == 0 and self.energy > creature.creatures[i].energy:
                        #self.energy += 50
                        #creature.creatures[i].energy -= 50

                     #   self.energy += creature.creatures[i].energy
                      #  creature.creatures[i].delete()
                    #elif self.recurrence[0] != 0 and creature.creatures[i].recurrence[0] != 0:
                        #creature.creatures[i].delete()
                        #self.delete()

    def lidarAll(self):
        _lidarList = []
        _directionStep = (2 * np.pi) / (self.nlidar)
        for i in range(self.nlidar):
            _singleLidar = self.lidarSingle(self.d + i * _directionStep)
            _lidarList += _singleLidar
        return _lidarList

    def collectAiData(self):
        return self.lidarAll() + self.recurrence + [self.energy]

    def forwardModel(self):
        self.networkInput = self.collectAiData()
        # print(self.networkInput)
        self.model.forward([self.networkInput])
        self.networkOutput = self.model.output[0]

        if self.networkOutput[0] > 0:
            self.mUp = 1
        else:
            self.mUp = 0

        if self.networkOutput[1] > 0:
            self.mDown = 1
        else:
            self.mDown = 0

        if self.networkOutput[2] > 0:
            self.mLeft = 1
        else:
            self.mLeft = 0

        if self.networkOutput[3] > 0:
            self.mRight = 1
        else:
            self.mRight = 0

        self.color = np.floor(
            ((self.networkOutput[4]) % 255, self.networkOutput[5] % 255, self.networkOutput[6] % 255))
        # print(self.networkOutput[4],self.networkOutput[5],self.networkOutput[6])
        # print("c=",self.color)
        for i in range(len(self.recurrence)):
            self.recurrence[i] = self.networkOutput[i + 7] % 100

    def updateHighScore(self):
        if self.score > creature.highScore:
            creature.highScore = self.score
            creature.bestModel = self

    def frame(self):
        self.updateEnergy()
        self.forwardModel()
        self.move()
        self.draw()
        self.colisionDetection()
        self.updateHighScore()


class food:
    foods = []

    def __init__(self, X, size, color, kind):
        self.n = len(food.foods)
        food.foods.append(self)
        self.X = X
        self.size = size
        self.color = color
        self.kind = kind

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.X[0], self.X[1]), self.size)

    def frame(self):
        self.draw()


def playerInput(player):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
            player.mUp = 1
        if event.key == pygame.K_a:
            player.mLeft = 1
        if event.key == pygame.K_s:
            player.mDown = 1
        if event.key == pygame.K_d:
            player.mRight = 1

    if event.type == pygame.KEYUP:
        if event.key == pygame.K_w:
            player.mUp = 0
        if event.key == pygame.K_a:
            player.mLeft = 0
        if event.key == pygame.K_s:
            player.mDown = 0
        if event.key == pygame.K_d:
            player.mRight = 0

def createCreature(model):
    _randomx = np.random.rand() * window_width
    _randomy = np.random.rand() * window_height
    _randomd = np.random.randn() * 2 * np.pi
    creature([_randomx, _randomy], _randomd, 10, 0.1, 10, (100, 100, 100), creaturesnLidar,
             creaturesLidarSize, creaturesStartingEnergy,
             model, creaturesRecurrence, creaturesInitialRate)



def nFoods():
    _answer = 0
    for i in range(len(food.foods)):
        if food.foods[i].kind == 1:
            _answer += 1
    return _answer


def nPoisons():
    return len(food.foods) - nFoods()


def createRandomFood():
    _randomX = random(0, window_width)
    _randomY = random(0, window_height)
    food([_randomX, _randomY], foodSize, foodColor, 1)


def createRandomPoison():
    _randomX = random(0, window_width)
    _randomY = random(0, window_height)
    food([_randomX, _randomY], foodSize, poisonColor, 0)


def frameCreatures():
    for i in range(len(creature.creatures)):
        if i >= len(creature.creatures):
            break
        creature.creatures[i].frame()


def frameFoods():
    for i in range(len(food.foods)):
        food.foods[i].frame()

def randomCreature():
    _random = int(np.floor(random(0,len(creature.creatures))))
    return creature.creatures[_random]
# p1 = creature([100, 100], 0, 10, 0.1, 10, (100, 100, 100), 2, creaturesLidarSize, creaturesStartingEnergy)
# nCreaturesExcludingPlayer = 9
# for i in range(nCreaturesExcludingPlayer):
#    _randomx = np.random.rand() * window_width
#    _randomy = np.random.rand() * window_height
#    creature([_randomx, _randomy], 0, 10, 0.1, 10, (100, 100, 100), 10, creaturesLidarSize, creaturesStartingEnergy)
# nFoods = 10
# for i in range(nFoods):
#    _randomx = np.random.rand() * window_width
#    _randomy = np.random.rand() * window_height
#    food([_randomx, _randomy], 5, (255,0,0),1)

# print(creaturesnLidar * 5 + creaturesRecurrence)


def savemodels(direct):
    modelau = []
    for i in range(len(creature.creatures)):
        modelau.append([creature.creatures[i].model, creature.creatures[i].rate, creature.creatures[i].lidarSize])
    data = [modelau, creature.totalCreatures, creature.poisonsCollected, creature.foodsCollected]
    pickle.dump(data, open(direct, "wb"))


def loadmodels(direct):
    modelau = pickle.load(open(direct, "rb"))[0]
    if keepStats == 1:
        creature.totalCreatures = pickle.load(open(direct, "rb"))[1]
        creature.poisonsCollected = pickle.load(open(direct, "rb"))[2]
        creature.foodsCollected = pickle.load(open(direct, "rb"))[3]
    else:
        creature.totalCreatures = 0
        creature.poisonsCollected = 0
        creature.foodsCollected = 0
    for i in range(len(modelau)):
        creature.totalCreatures += 0
        _randomx = np.random.rand() * window_width
        _randomy = np.random.rand() * window_height
        _randomd = (np.random.randn() + 0) * np.pi
        creature([_randomx, _randomy], _randomd, creaturesv, creaturesdv, creaturesSize, (0, 0, 0), creaturesnLidar,
                 modelau[i][2], creaturesStartingEnergy, modelau[i][0], creaturesRecurrence, modelau[i][1])


if loadFrom == 0:
    creature0model = Network([creaturesnLidar * 5 + creaturesRecurrence + 1] + hiddelLayerShape + [7 + creaturesRecurrence],
                             Activation_ReLU)
    creature0 = creature([0, 0], creaturesv, 10, 0.1, 10, (100, 100, 100), creaturesnLidar, creaturesLidarSize,
                         creaturesStartingEnergy,
                         creature0model, creaturesRecurrence, creaturesInitialRate)
else:
    loadmodels(loadFrom)

for i in range(0):
    creature.creatures[0].clone()

timebetweenlevels = 0

running = True
while running:
    if timebetweenlevels > 0:
        timebetweenlevels -= 1
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        # playerInput(p1)
        if event.type == pygame.QUIT:
            savemodels(saveTo)
            running = False
    # print(p1.lidarAll())
    while nFoods() < foods - len(creature.creatures):
        createRandomFood()
    while nPoisons() < (poisons)*len(creature.creatures):
        createRandomPoison()
    while len(creature.creatures) < 2:
        randomCreature().clone()
    screen.blit(font.render("C:" + str(len(creature.creatures)), 1, pygame.Color("coral")), (10, 20))
    screen.blit(font.render("f:" + str(nFoods()), 1, pygame.Color("coral")), (10, 40))
    screen.blit(font.render("p:" + str(nPoisons()), 1, pygame.Color("coral")), (10, 60))
    screen.blit(font.render("TC:" + str(creature.totalCreatures), 1, pygame.Color("coral")), (10, 80))
    screen.blit(font.render("TF:" + str(creature.foodsCollected), 1, pygame.Color("coral")), (10, 100))
    screen.blit(font.render("TP:" + str(creature.poisonsCollected), 1, pygame.Color("coral")), (10, 120))
    screen.blit(font.render("Level:" + str(level), 1, pygame.Color("coral")), (10, 140))
    screen.blit(font.render("highScore:" + str(creature.highScore), 1, pygame.Color("coral")), (10, 160))
    frameCreatures()
    frameFoods()
    screen.blit(update_fps(), (10, 0))
    pygame.display.update()
    clock.tick(fps)
    #print(timebetweenlevels)
    #if creature.totalCreatures % 100 == 0 and creature.bestModel != 0:
    #    for i in range(5):
    #        creature.bestModel.clone()
    #if creature.totalCreatures % 1000 <10 :
    #    creature.bestModel = 0
    #    creature.highScore = 0
    if (len(creature.creatures) > 30 or creature.creatures[0].rate > 3) and timebetweenlevels < 1 and len(creature.creatures) > 15:
        level += 1
        if level % 2 == 0 :
            foodstoBreed += 1
        #if level % 2 == 1:
        #    poisons += 1
        timebetweenlevels = 1000
    #if creature.totalCreatures % 5000 == 0 and level > 0 and timebetweenlevels < 1:
    #    level -= 1
    #    if level % 2 == 1 :
    #        foodstoBreed -= 1
        #if level % 2 == 0:
        #    poisons -= 1



