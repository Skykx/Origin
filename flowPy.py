#https://towardsdatascience.com/computational-fluid-dynamics-using-python-modeling-laminar-flow-272dad1ebec

import numpy as np
import pandas as pd
import os

from parameters_htm_cfd import rowPoints, colPoints


class Boundary:
    def __init__(self, type, value):
        self.__defineBoundary(type, value)

    def __defineBoundary(self, type, value):
        self.type = type
        self.value = value


class Space:
    def __init__(self):
        self.dt = None
        self.S_y = None
        self.S_x = None
        self.p_c = None
        self.p = None
        self.u_next = None
        self.v_c = None
        self.v_next = None
        self.v_star = None
        self.u_star = None
        self.v = None
        self.u = None
        self.colPoints = None
        self.rowPoints = None

    def createMesh(self, rowpts, colpts):
        # domain gridpoints
        self.rowPoints = rowpts
        self.colPoints = colpts

        # velocity matrices
        self.u = np.zeros((self.rowPoints + 2, self.colPoints + 2))
        self.v = np.zeros((self.rowPoints + 2, self.colPoints + 2))
        self.u_star = np.zeros((self.rowPoints + 2, self.colPoints + 2))
        self.v_star = np.zeros((self.rowPoints + 2, self.colPoints + 2))
        self.u_next = np.zeros((self.rowPoints + 2, self.colPoints + 2))
        self.v_next = np.zeros((self.rowPoints + 2, self.colPoints + 2))
        self.u_c = np.zeros((self.rowPoints, self.colPoints))
        self.v_c = np.zeros((self.rowPoints, self.colPoints))

        # pressure matrices
        self.p = np.zeros((self.rowPoints + 2, self.colPoints + 2))
        self.p_c = np.zeros((self.rowPoints, self.colPoints))

        # set default source term
        self.setSourceTerm()

    def setDeltas(self, breadth, length):
        self.dx = length / (self.colPoints - 1)
        self.dy = breadth / (self.rowPoints - 1)

    def setInitalU(self, U):
        self.u = U * self.u

    def setInitialV(self, V):
        self.v = V * self.v

    def setInitialP(self, P):
        self.p = P * self.p

    def setSourceTerm(self, S_x=0, S_y=0):
        self.S_x = S_x
        self.S_y = S_y


class Fluid:
    def __init__(self, rho, mu):
        self.mu = None
        self.rho = None
        self.setFluidProperties(rho, mu)

    def setFluidProperties(self, rho, mu):
        self.rho = rho
        self.mu = mu


# Note: The arguments to the function are all objects of our defined classes
# Set boundary conditions for horizontal velocity
def setUBoundary(space, left, right, top, bottom):
    if left.type == "D":
        space.u[:, 0] = left.value
    elif left.type == "N":
        space.u[:, 0] = -left.value * space.dx + space.u[:, 1]

    if right.type == "D":
        space.u[:, -1] = right.value
    elif right.type == "N":
        space.u[:, -1] = right.value * space.dx + space.u[:, -2]

    if top.type == "D":
        space.u[-1, :] = 2 * top.value - space.u[-2, :]
    elif top.type == "N":
        space.u[-1, :] = -top.value * space.dy + space.u[-2, :]

    if bottom.type == "D":
        space.u[0, :] = 2 * bottom.value - space.u[1, :]
    elif bottom.type == "N":
        space.u[0, :] = bottom.value * space.dy + space.u[1, :]


# Set boundary conditions for vertical velocity
def setVBoundary(space, left, right, top, bottom):
    if left.type == "D":
        space.v[:, 0] = 2 * left.value - space.v[:, 1]
    elif left.type == "N":
        space.v[:, 0] = -left.value * space.dx + space.v[:, 1]

    if right.type == "D":
        space.v[:, -1] = 2 * right.value - space.v[:, -2]
    elif right.type == "N":
        space.v[:, -1] = right.value * space.dx + space.v[:, -2]

    if top.type == "D":
        space.v[-1, :] = top.value
    elif top.type == "N":
        space.v[-1, :] = -top.value * space.dy + space.v[-2, :]

    if bottom.type == "D":
        space.v[0, :] = bottom.value
    elif bottom.type == "N":
        space.v[0, :] = bottom.value * space.dy + space.v[1, :]


# Set boundary conditions for pressure
def setPBoundary(space, left, right, top, bottom):
    if left.type == "D":
        space.p[:, 0] = left.value
    elif left.type == "N":
        space.p[:, 0] = -left.value * space.dx + space.p[:, 1]

    if right.type == "D":
        space.p[1, -1] = right.value
    elif right.type == "N":
        space.p[:, -1] = right.value * space.dx + space.p[:, -2]

    if top.type == "D":
        space.p[-1, :] = top.value
    elif top.type == "N":
        space.p[-1, :] = -top.value * space.dy + space.p[-2, :]

    if bottom.type == "D":
        space.p[0, :] = bottom.value
    elif bottom.type == "N":
        space.p[0, :] = bottom.value * space.dy + space.p[1, :]


def setTimeStep(CFL, space, fluid):
    with np.errstate(divide='ignore'):
        dt = CFL / np.sum([np.amax(space.u) / space.dx,
                           np.amax(space.v) / space.dy])
    # Escape condition if dt is infinity due to zero velocity initially
    if np.isinf(dt):
        dt = CFL * (space.dx + space.dy)
    space.dt = dt


# The first function is used to get starred velocities from u and v at timestep t
def getStarredVelocities(space, fluid):
    # Save object attributes as local variable with explicity typing for improved readability
    rows = int(space.rowPoints)
    cols = int(space.colPoints)
    u = space.u.astype(float, copy=False)
    v = space.v.astype(float, copy=False)
    dx = float(space.dx)
    dy = float(space.dy)
    dt = float(space.dt)
    S_x = float(space.S_x)
    S_y = float(space.S_y)
    rho = float(fluid.rho)
    mu = float(fluid.mu)

    # Copy u and v to new variables u_star and v_star
    u_star = u.copy()
    v_star = v.copy()

    # Calculate derivatives of u and v using the finite difference scheme.
    # Numpy vectorization saves us from using slower for loops to go over each element in the u and v matrices
    u1_y = (u[2:, 1:cols + 1] - u[0:rows, 1:cols + 1]) / (2 * dy)
    u1_x = (u[1:rows + 1, 2:] - u[1:rows + 1, 0:cols]) / (2 * dx)
    u2_y = (u[2:, 1:cols + 1] - 2 * u[1:rows + 1, 1:cols + 1] + u[0:rows, 1:cols + 1]) / (dy ** 2)
    u2_x = (u[1:rows + 1, 2:] - 2 * u[1:rows + 1, 1:cols + 1] + u[1:rows + 1, 0:cols]) / (dx ** 2)
    v_face = (v[1:rows + 1, 1:cols + 1] + v[1:rows + 1, 0:cols] + v[2:, 1:cols + 1] + v[2:, 0:cols]) / 4
    u_star[1:rows + 1, 1:cols + 1] = u[1:rows + 1, 1:cols + 1] - dt * (
            u[1:rows + 1, 1:cols + 1] * u1_x + v_face * u1_y) + (dt * (mu / rho) * (u2_x + u2_y)) + (dt * S_x)

    v1_y = (v[2:, 1:cols + 1] - v[0:rows, 1:cols + 1]) / (2 * dy)
    v1_x = (v[1:rows + 1, 2:] - v[1:rows + 1, 0:cols]) / (2 * dx)
    v2_y = (v[2:, 1:cols + 1] - 2 * v[1:rows + 1, 1:cols + 1] + v[0:rows, 1:cols + 1]) / (dy ** 2)
    v2_x = (v[1:rows + 1, 2:] - 2 * v[1:rows + 1, 1:cols + 1] + v[1:rows + 1, 0:cols]) / (dx ** 2)
    u_face = (u[1:rows + 1, 1:cols + 1] + u[1:rows + 1, 2:] + u[0:rows, 1:cols + 1] + u[0:rows, 2:]) / 4
    v_star[1:rows + 1, 1:cols + 1] = v[1:rows + 1, 1:cols + 1] - dt * (
            u_face * v1_x + v[1:rows + 1, 1:cols + 1] * v1_y) + (dt * (mu / rho) * (v2_x + v2_y)) + (dt * S_y)

    # Save the calculated starred velocities to the space object
    space.u_star = u_star.copy()
    space.v_star = v_star.copy()


# The second function is used to iteratively solve the pressure Possion equation from the starred velocities
# to calculate pressure at t+delta_t
def solvePressurePoisson(space, fluid, left, right, top, bottom):
    # Save object attributes as local variable with explicity typing for improved readability
    rows = int(space.rowPoints)
    cols = int(space.colPoints)
    u_star = space.u_star.astype(float, copy=False)
    v_star = space.v_star.astype(float, copy=False)
    p = space.p.astype(float, copy=False)
    dx = float(space.dx)
    dy = float(space.dy)
    dt = float(space.dt)
    rho = float(fluid.rho)
    factor = 1 / (2 / dx ** 2 + 2 / dy ** 2)

    # Define initial error and tolerance for convergence (error > tol necessary initially)
    error = 1
    tol = 1e-3

    # Evaluate derivative of starred velocities
    ustar1_x = (u_star[1:rows + 1, 2:] - u_star[1:rows + 1, 0:cols]) / (2 * dx)
    vstar1_y = (v_star[2:, 1:cols + 1] - v_star[0:rows, 1:cols + 1]) / (2 * dy)

    # Continue iterative solution until error becomes smaller than tolerance
    i = 0
    while error > tol:
        i += 1

        # Save current pressure as p_old
        p_old = p.astype(float, copy=True)

        # Evaluate second derivative of pressure from p_old
        p2_xy = (p_old[2:, 1:cols + 1] + p_old[0:rows, 1:cols + 1]) / dy ** 2 + (
                p_old[1:rows + 1, 2:] + p_old[1:rows + 1, 0:cols]) / dx ** 2

        # Calculate new pressure
        p[1:rows + 1, 1:cols + 1] = p2_xy * factor - (rho * factor / dt) * (ustar1_x + vstar1_y)

        # Find maximum error between old and new pressure matrices
        error = np.amax(abs(p - p_old))

        # Apply pressure boundary conditions
        setPBoundary(space, left, right, top, bottom)

        # Escape condition in case solution does not converge after 500 iterations
        if i > 500:
            tol *= 10


# The third function is used to calculate the velocities at timestep t+delta_t using the pressure at t+delta_t and
# starred velocities
def solveMomentumEquation(space, fluid):
    # Save object attributes as local variable with explicity typing for improved readability
    rows = int(space.rowPoints)
    cols = int(space.colPoints)
    u_star = space.u_star.astype(float, copy=False)
    v_star = space.v_star.astype(float, copy=False)
    p = space.p.astype(float, copy=False)
    dx = float(space.dx)
    dy = float(space.dy)
    dt = float(space.dt)
    rho = float(fluid.rho)
    u = space.u.astype(float, copy=False)
    v = space.v.astype(float, copy=False)

    # Evaluate first derivative of pressure in x direction
    p1_x = (p[1:rows + 1, 2:] - p[1:rows + 1, 0:cols]) / (2 * dx)
    # Calculate u at next timestep
    u[1:rows + 1, 1:cols + 1] = u_star[1:rows + 1, 1:cols + 1] - (dt / rho) * p1_x

    # Evaluate first derivative of pressure in y direction
    p1_y = (p[2:, 1:cols + 1] - p[0:rows, 1:cols + 1]) / (2 * dy)
    # Calculate v at next timestep
    v[1:rows + 1, 1:cols + 1] = v_star[1:rows + 1, 1:cols + 1] - (dt / rho) * p1_y


def setCentrePUV(space):
    space.p_c = space.p[1:-1, 1:-1]
    space.u_c = space.u[1:-1, 1:-1]
    space.v_c = space.v[1:-1, 1:-1]


def makeResultDirectory(u_in, wipe=False):
    # Get path to the result directory
    cwdir = os.getcwd()
    dir_path = os.path.join(cwdir, "result/u_in_{}".format(u_in))
    # If directory does not exist, make it
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    else:
        # If wipe is True, remove files present in the directory
        if wipe:
            os.chdir(dir_path)
            filelist = os.listdir()
            for file in filelist:
                os.remove(file)

    os.chdir(cwdir)


def writeToFile(space, iteration, interval, dir_path):
    if iteration % interval == 0:
        filename = "PUV{0}.csv".format(iteration) #der Funktion writeToFile wird der Dateiname durch die Zeile "filename = "PUV{0}.csv".format(iteration)" festgelegt. Dabei wird der Iterationswert in den Dateinamen eingefügt.
        path = os.path.join(dir_path, filename) #Dann wird der Pfad der Datei erstellt, indem der Dateiname mit dem Pfad des Ordners zusammengefügt wird.
        data = {"p": space.p_c.ravel(), #Die Druck- (space.p_c), die x-Geschwindigkeits- (space.u_c) und die y-
                "u": space.u_c.ravel(), #Geschwindigkeitswerte (space.v_c) werden aus dem space-Objekt extrahiert und in ein
                "v": space.v_c.ravel()} # Python-Dictionary gespeichert.
        df = pd.DataFrame(data, columns=["p", "u", "v"]) #Dieses Dictionary wird dann in einen Pandas DataFrame umgewandelt, wobei die Spaltennamen "p", "u" und "v" verwendet werden.
        df.to_csv(path)


def read_datafile(iteration, dir_path):
    # set filename and path according to given iteration
    filename = "PUV{0}.csv".format(iteration)   #Diese Zeile legt den Dateinamen für die CSV-Datei fest, die geladen werden soll. Das Format-Kommando {0} wird durch die variable iteration ersetzt. Beispielsweise, wenn iteration den Wert 5 hat, wird filename den Wert "PUV5.csv" haben. Es wird ein Dateiname erstellt, der aus einem festen Teil "PUV" und dem Wert der iteration variable besteht.
    filepath = os.path.join(dir_path, filename) #Genau, der erste Befehl, filename = "PUV{0}.csv".format(iteration), erstellt den Dateinamen auf Basis des übergebenen Iterationswerts. Der zweite Befehl, filepath = os.path.join(dir_path, filename), fügt den Pfad und den Dateinamen zusammen, um den vollständigen Pfad der Datei zu erhalten. Anschließend kann die Datei mit diesem Pfad gelesen werden.

    # load csv
    df = pd.read_csv(filepath)  #Der Befehl "pd.read_csv(filepath)" liest die CSV-Datei, die sich an der angegebenen Pfad "filepath" befindet, mit Hilfe der Pandas-Bibliothek ein. Es wird ein DataFrame erstellt, der die Daten der CSV-Datei enthält und in der Variablen "df" gespeichert wird.
                                #Pandas ist eine Bibliothek in Python, die es ermöglicht, Daten in einem sogenannten DataFrame (ähnlich wie in einer Excel-Tabelle) zu verwalten und zu analysieren. Der Befehl "pd.read_csv(filepath)" liest die CSV-Datei, die sich an dem Pfad "filepath" befindet, ein und speichert sie in einem DataFrame, der in der Variablen "df" gespeichert wird.
    p_arr = df["p"].to_numpy()  #Dieser Befehl nimmt die Spalte mit dem Namen "p" aus dem geladenen DataFrame "df" und konvertiert sie in ein numpy Array "p_arr". Mit der Methode ".to_numpy()" wird die Spalte in ein numpy Array umgewandelt, damit es leichter ist, mit den Daten in dieser Spalte zu arbeiten.
    u_arr = df["u"].to_numpy()  #Das heißt mit diesen Befehlen: p_arr = df["p"].to_numpy() erkennt panda die Formatierung in der Datei:,p,u,v -> 0,0.0,0.0,0.0 -> 1,0.0,0.0,0.0 und weiß dann automatisch das die erste Zahl nur der Index ist, die 2. Zahl für p, 3. Zahl für u und 4. Zahl für v und alle mit einem ","getrennt sind ?
    v_arr = df["v"].to_numpy()  #Ja, das ist richtig. Pandas verwendet die erste Zeile als Kopfzeile zur Identifizierung der Spaltennamen und die restlichen Zeilen werden als Datenzeilen interpretiert. Da die erste Spalte keinen Namen hat, verwendet Pandas standardmäßig den Index 0, 1, 2, usw. für die Spalte. Es ist jedoch möglich, einen eigenen Namen für die erste Spalte festzulegen, wenn die CSV-Datei gelesen wird.

    # Reshape 1D data into 2D -> Man erstellt eine 128x128 Matrix wegen Parameter rowPoints&colPoints
    p_p = p_arr.reshape((rowPoints, colPoints)) #Die Methode .reshape() von numpy dient dazu, ein bestehendes Array in eine neue Form umzuwandeln. In diesem Fall wird das Array p_arr in eine 2D-Form mit den Dimensionen rowPoints und colPoints umgeformt. Dadurch kann das Array beispielsweise in einer Matrix dargestellt werden und es können einfacher bestimmte Elemente ausgewählt werden. rowPoints und colPoints sind dabei Variablen, die vorher definiert wurden und die die Größe der Matrix bestimmen.
    u_p = u_arr.reshape((rowPoints, colPoints)) #Um eine Matrix mit den Dimensionen rowPoints x colPoints zu erhalten, benötigt man insgesamt rowPoints * colPoints Daten. In dem Beispiel 129 * 129 = 16641 Daten. Daher müssen in p_arr mindestens 16641 Daten vorliegen, damit die reshape Methode erfolgreich durchgeführt werden kann.
    v_p = v_arr.reshape((rowPoints, colPoints)) #Wenn weniger als 16641 Daten vorliegen, wird die reshape() Funktion einen Fehler werfen, da nicht genug Daten vorhanden sind, um die Matrix mit der angegebenen Größe zu füllen. Wenn mehr Daten vorliegen, werden die überschüssigen Daten ignoriert. In beiden Fällen wird das Ergebnis der reshape() Funktion nicht korrekt sein und es kann zu unerwartetem Verhalten im restlichen Skript kommen. Es ist wichtig sicherzustellen, dass die Anzahl der Daten, die gelesen werden, mit der Größe der erwarteten Matrix übereinstimmt.

    return p_p, u_p, v_p
