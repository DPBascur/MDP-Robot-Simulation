import pygame
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# --- Parametros generales de visualizacion ---
CELL_SIZE = 80  # Tamano de cada celda en la ventana
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_COLOR = (200, 200, 200)
ROBOT_COLOR = (255, 0, 0)

# --- Definicion del mapa del entorno ---
# 0: espacio libre, 1: obstaculo, 2: penalizacion, 3: meta
lstMapa = [  # mapa
    [0, 0, 2, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 2, 0, 1, 0, 2, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 2, 0],
    [0, 1, 0, 0, 1, 0, 0, 0]
]

# --- Definicion de acciones permitidas: Norte, Sur, Este, Oeste ---
dictAcciones = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "O": (-1, 0)}  # acciones
lstAcciones = ["N", "S", "E", "O"]  # acciones_lista

# --- Mapeo de posiciones a estados y viceversa ---
def fnGenerarMapeo(lstMapa):  # generar_mapeo
    # generar_mapeo(mapa)
    # Recorre el mapa y asigna un ID de estado a cada posicion libre del mapa
    # Devuelve dos diccionarios para conversion entre posiciones y estados
    """Genera los mapeos de posicion a estado y de estado a posicion."""
    iEstadoId = 0  # estado
    dictPosEstado, dictEstadoPos = {}, {}  # pos_a_estado, estado_a_pos
    for y, lstFila in enumerate(lstMapa):  # y, fila
        for x, iCelda in enumerate(lstFila):  # x, celda
            if iCelda != 1:
                dictPosEstado[(x, y)] = iEstadoId
                dictEstadoPos[iEstadoId] = (x, y)
                iEstadoId += 1
    return dictPosEstado, dictEstadoPos

dictPosEstado, dictEstadoPos = fnGenerarMapeo(lstMapa)  # pos_a_estado, estado_a_pos
iNumEstados = len(dictEstadoPos)  # nS

# --- Validacion de movimientos dentro de los limites del mapa ---
def fnEsValido(lstMapa, x, y):  # es_valido
    # es_valido(mapa, x, y)
    # Comprueba si las coordenadas estan dentro del mapa y no corresponden a un obstaculo
    """Verifica si una posicion es valida y no es un obstaculo."""
    iFilas, iColumnas = len(lstMapa), len(lstMapa[0])  # filas, columnas
    return 0 <= x < iColumnas and 0 <= y < iFilas and lstMapa[y][x] != 1

# --- Generacion de las matrices de transicion para el MDP ---
def fnGenerarMatricesTransicion(lstMapa, dictPosEstado, dictEstadoPos, fProbExito, fProbLateral):  # generar_matrices_transicion
    # generar_matrices_transicion(mapa, pos_a_estado, estado_a_pos, prob_exito, prob_lateral)
    # Calcula las probabilidades de transicion para cada estado y accion en el mapa
    # Usa fProbExito para la accion deseada y fProbLateral para las acciones laterales (giro)
    # Los estados meta (celda con valor 3) se configuran como absorbentes (probabilidad 1 de permanecer)
    """Genera las matrices de transicion para cada accion considerando incertidumbre."""
    iNumEstados = len(dictEstadoPos)  # nS
    lstP = [np.zeros((iNumEstados, iNumEstados)) for _ in range(4)]  # P
    for iAccionIdx, strAccion in enumerate(lstAcciones):  # a_idx, accion
        for s in range(iNumEstados):
            x, y = dictEstadoPos[s]
            iDeltaX, iDeltaY = dictAcciones[strAccion]  # dx, dy
            x1, y1 = x + iDeltaX, y + iDeltaY
            iEstadoDestino = dictPosEstado.get((x1, y1), s) if fnEsValido(lstMapa, x1, y1) else s  # s1
            lstP[iAccionIdx][s][iEstadoDestino] += fProbExito
            lstLaterales = ["O", "E"] if strAccion in ["N", "S"] else ["N", "S"]  # laterales
            for strLateral in lstLaterales:  # lateral
                iDeltaXLateral, iDeltaYLateral = dictAcciones[strLateral]  # dx_l, dy_l
                x2, y2 = x + iDeltaXLateral, y + iDeltaYLateral
                iEstadoDestinoLateral = dictPosEstado.get((x2, y2), s) if fnEsValido(lstMapa, x2, y2) else s  # s2
                lstP[iAccionIdx][s][iEstadoDestinoLateral] += fProbLateral
    # Estados absorbentes en la meta
    for s in range(iNumEstados):
        x, y = dictEstadoPos[s]
        if lstMapa[y][x] == 3:
            for iAccionIdx in range(4):  # a_idx
                lstP[iAccionIdx][s][:] = 0
                lstP[iAccionIdx][s][s] = 1.0
    return lstP

# --- Definicion de la funcion de recompensa ---
def fnObtenerRecompensa(lstMapa, dictEstadoPos, s, dictPosEstado):  # obtener_recompensa
    # obtener_recompensa(mapa, estado_a_pos, s, pos_a_estado)
    # Retorna la recompensa inmediata segun el tipo de celda del estado dado
    # 10 para estado meta, -0.5 para estado con penalizacion, -0.1 para un estado normal
    """Devuelve la recompensa asociada al estado s."""
    x, y = dictEstadoPos[s]
    iCelda = lstMapa[y][x]  # celda
    return 10 if iCelda == 3 else -0.5 if iCelda == 2 else -0.1

# --- Algoritmo de Value Iteration para obtener la politica optima ---
def fnValueIteration(lstMapa, dictPosEstado, dictEstadoPos, lstP, fGamma=0.9, fEpsilon=0.001):  # value_iteration
    # value_iteration(mapa, pos_a_estado, estado_a_pos, P, gamma=0.9, epsilon=0.001)
    # Algoritmo de iteracion de valores para obtener la politica optima
    # Inicializa los valores de todos los estados en 0 y luego itera hasta converger (cambio menor a fEpsilon)
    # En cada paso de iteracion calcula los valores Q(s,a) para cada accion a desde cada estado s
    # Q(s,a) se calcula como la recompensa inmediata mas el valor descontado de estados futuros
    # Actualiza arrV[s] al maximo de Q(s,a) y arrPolitica[s] con la accion que da dicho maximo
    # Calcula delta como el mayor cambio absoluto en V; cuando este es menor a fEpsilon, se alcanza la convergencia
    """Implementacion de Value Iteration."""
    iNumEstados = len(dictEstadoPos)  # nS
    arrV, arrPolitica = np.zeros(iNumEstados), np.zeros(iNumEstados, dtype=int)  # V, politica
    while True:
        delta = 0
        for s in range(iNumEstados):
            v_actual = arrV[s]
            Q_sa = [fnObtenerRecompensa(lstMapa, dictEstadoPos, s, dictPosEstado) + fGamma * sum(lstP[a][s][iEstadoDestino] * arrV[iEstadoDestino] for iEstadoDestino in range(iNumEstados)) for a in range(4)]
            arrV[s] = max(Q_sa)
            arrPolitica[s] = np.argmax(Q_sa)
            delta = max(delta, abs(v_actual - arrV[s]))
        if delta < fEpsilon:
            break
    return arrV, arrPolitica

# --- Visualizacion grafica de la politica y simulacion del robot ---
def fnSimulacionVisual(lstMapa, arrPolitica, dictEstadoPos, dictPosEstado, arrV, lstPActual):  # simulacion_visual
    # simulacion_visual(mapa, politica, estado_a_pos, pos_a_estado, V, P_actual)
    # Inicializa la visualizacion grafica con pygame y configura la ventana
    # Carga las imagenes del robot orientado en las cuatro direcciones (sprites)
    # Selecciona una posicion inicial aleatoria para el robot en una celda valida (no obstaculo ni meta)
    # El robot seguira la politica optima: en cada paso tomara la accion indicada por arrPolitica para su estado actual
    # Si el robot alcanza la meta (celda con 3), se reinicia en una nueva posicion aleatoria y continua
    # La tecla 'E' alterna la visualizacion de flechas de direccion y valores en cada celda
    """Simulacion visual interactiva del robot siguiendo la politica optima."""
    pygame.init()
    iFilas, iColumnas = len(lstMapa), len(lstMapa[0])  # filas, columnas
    objPantalla = pygame.display.set_mode((iColumnas * CELL_SIZE, iFilas * CELL_SIZE))  # screen
    pygame.display.set_caption("MDP Robot Simulation By Daniel")
    objReloj = pygame.time.Clock()  # clock

    # Carga de sprites para cada direccion
    dictSprites = {  # sprite
        "N": pygame.image.load("rn.png").convert_alpha(),
        "S": pygame.image.load("rs.png").convert_alpha(),
        "E": pygame.image.load("re.png").convert_alpha(),
        "O": pygame.image.load("ro.png").convert_alpha()
    }
    for strKey in dictSprites:  # key
        dictSprites[strKey] = pygame.transform.scale(dictSprites[strKey], (CELL_SIZE, CELL_SIZE))

    lstPosicionesValidas = [(x, y) for y, lstFila in enumerate(lstMapa) for x, iCelda in enumerate(lstFila) if iCelda != 1 and iCelda != 3]  # posiciones_validas
    tplRobotPos = random.choice(lstPosicionesValidas)  # robot_pos
    iEstadoActual = dictPosEstado[tplRobotPos]  # estado_actual
    bMostrarElementos = True  # mostrar_elementos

    # Funcion interna para dibujar el mapa
    def fnDibujar():  # dibujar
        # Dibuja el mapa completo en la pantalla de juego
        # Colorea cada celda segun su tipo: obstaculos en negro, penalizaciones en rojo claro, meta en verde, libres en blanco
        # Si la celda es penalizacion (2) dibuja 'P'; si es meta (3) dibuja 'M'
        # Si bMostrarElementos es True, dibuja en cada celda libre una flecha indicando la accion optima y el valor V
        fntFuente = pygame.font.SysFont(None, 24)  # font
        for y, lstFila in enumerate(lstMapa):  # y, fila
            for x, iCelda in enumerate(lstFila):  # x, celda
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = BLACK if iCelda == 1 else (255, 200, 200) if iCelda == 2 else (180, 255, 180) if iCelda == 3 else WHITE
                pygame.draw.rect(objPantalla, color, rect)
                pygame.draw.rect(objPantalla, GRID_COLOR, rect, 1)
                if iCelda == 2:
                    objPantalla.blit(fntFuente.render("P", True, (0, 0, 0)), (rect.right - 20, rect.bottom - 20))
                elif iCelda == 3:
                    objPantalla.blit(fntFuente.render("M", True, (0, 0, 0)), (rect.right - 20, rect.bottom - 20))
                if iCelda != 1 and bMostrarElementos:
                    iEstado = dictPosEstado[(x, y)]  # estado
                    accion_idx = arrPolitica[iEstado]
                    iDeltaX, iDeltaY = dictAcciones[lstAcciones[accion_idx]]  # dx, dy
                    start_pos = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                    end_pos = (start_pos[0] + iDeltaX * CELL_SIZE // 3, start_pos[1] + iDeltaY * CELL_SIZE // 3)
                    pygame.draw.line(objPantalla, (0, 0, 255), start_pos, end_pos, 3)
                    pygame.draw.circle(objPantalla, (0, 0, 255), end_pos, 5)
                    fValor = arrV[iEstado]  # estado (valor)
                    objPantalla.blit(pygame.font.SysFont(None, 18).render(f"{fValor:.2f}", True, (50, 50, 50)), (x * CELL_SIZE + 5, y * CELL_SIZE + CELL_SIZE - 20))



    # Bucle principal de simulacion
    while True:
        for objEvento in pygame.event.get():
            if objEvento.type == pygame.QUIT:
                pygame.quit()
                return
            if objEvento.type == pygame.KEYDOWN:
                if objEvento.key == pygame.K_e:
                    bMostrarElementos = not bMostrarElementos

        x, y = tplRobotPos
        print(f"Robot en estado {iEstadoActual} en posicion ({x},{y})")

        if lstMapa[y][x] == 3:
            print("Meta alcanzada. Reiniciando posicion aleatoria.")
            tplRobotPos = random.choice(lstPosicionesValidas)
            iEstadoActual = dictPosEstado[tplRobotPos]
        else:
            iAccionIdx = arrPolitica[iEstadoActual]
            strAccion = lstAcciones[iAccionIdx]
            lstDestinos = list(range(iNumEstados))
            arrProbabilidades = lstPActual[iAccionIdx][iEstadoActual]
            iEstadoSiguiente = random.choices(lstDestinos, weights=arrProbabilidades)[0]
            tplRobotPos = dictEstadoPos[iEstadoSiguiente]
            print(f"Accion tomada: {strAccion} -> Nuevo estado: {iEstadoSiguiente} en posicion {tplRobotPos}")
            iEstadoActual = iEstadoSiguiente

        print("-" * 40)

        objPantalla.fill((0, 0, 0))        # Limpia pantalla
        fnDibujar()                        # Dibuja mapa y flechas

        # --- Dibuja el robot en su posicion actual ---
        x, y = tplRobotPos
        iAccionIdx = arrPolitica[iEstadoActual]
        strDireccion = lstAcciones[iAccionIdx]
        imgRobot = dictSprites[strDireccion]
        objPantalla.blit(imgRobot, (x * CELL_SIZE, y * CELL_SIZE))

        pygame.display.flip()             # Refresca pantalla
        time.sleep(0.5)


# --- Simulacion de episodios para robustez ---
def fnSimularEpisodio(lstMapa, arrPolitica, dictEstadoPos, dictPosEstado, lstPActual, iPasosTotales=1000):  # simular_episodio
    # simular_episodio(mapa, politica, estado_a_pos, pos_a_estado, P_actual, pasos_totales)
    # Simula un recorrido del robot en el mapa siguiendo una politica fija
    # El robot parte de una posicion aleatoria valida (no obstaculo ni meta) y ejecuta iPasosTotales pasos
    # Si el robot llega a la meta en algun paso, se reposiciona aleatoriamente para continuar simulando
    # En cada paso se aplica la accion dada por la politica y se determina el nuevo estado segun las probabilidades P_actual
    # Se acumula la recompensa de cada paso: -0.1 en celdas normales, -0.5 en penalizaciones, +10 al alcanzar la meta
    # Retorna la recompensa total obtenida al finalizar el episodio simulado
    """Simula un episodio completo para evaluacion de robustez."""
    lstPosicionesValidas = [(x, y) for y, lstFila in enumerate(lstMapa) for x, iCelda in enumerate(lstFila) if iCelda != 1 and iCelda != 3]
    tplRobotPos = random.choice(lstPosicionesValidas)
    iEstadoActual = dictPosEstado[tplRobotPos]
    fRecompensaTotal = 0

    for iPaso in range(iPasosTotales):  # paso
        x, y = tplRobotPos
        if lstMapa[y][x] == 3:
            tplRobotPos = random.choice(lstPosicionesValidas)
            iEstadoActual = dictPosEstado[tplRobotPos]
            fRecompensaTotal += 10
            continue

        iAccionIdx = arrPolitica[iEstadoActual]
        lstDestinos = list(range(len(dictPosEstado)))  # destinos
        arrProbabilidades = lstPActual[iAccionIdx][iEstadoActual]
        iEstadoSiguiente = random.choices(lstDestinos, weights=arrProbabilidades)[0]
        tplRobotPos = dictEstadoPos[iEstadoSiguiente]
        iEstadoActual = iEstadoSiguiente

        iCeldaActual = lstMapa[tplRobotPos[1]][tplRobotPos[0]]  # celda_actual
        if iCeldaActual == 2:
            fRecompensaTotal += -0.5
        else:
            fRecompensaTotal += -0.1

    return fRecompensaTotal

# --- Bloque principal de ejecucion ---
# --- Ejecucion principal del programa ---
# Calcula la politica optima inicial y lanza una simulacion visual del robot
if __name__ == "__main__":
    # Simulacion visual inicial
    lstPActual = fnGenerarMatricesTransicion(lstMapa, dictPosEstado, dictEstadoPos, 0.8, 0.1)
    arrVFinal, arrPoliticaFinal = fnValueIteration(lstMapa, dictPosEstado, dictEstadoPos, lstPActual, fGamma=0.90)
    fnSimulacionVisual(lstMapa, arrPoliticaFinal, dictEstadoPos, dictPosEstado, arrVFinal, lstPActual)

    # Evaluacion de la robustez de las politicas optimas bajo distintos escenarios
    lstLambdas = [0.86, 0.90, 0.94, 0.98]
    lstProbabilidades = [(0.80, 0.10), (0.90, 0.05), (0.70, 0.15), (0.50, 0.25)]

    plt.figure(figsize=(12, 8))

    for iIndiceLambda, fLambda in enumerate(lstLambdas):  # Recorre cada factor de descuento (lambda) indicado
        lstRecompensasProm = []  # Lista para almacenar recompensas promedio con este lambda
        for fProbExito, fProbLateral in lstProbabilidades:  # Itera sobre cada escenario de probabilidad de exito y lateral
            lstPActual = fnGenerarMatricesTransicion(lstMapa, dictPosEstado, dictEstadoPos, fProbExito, fProbLateral)  # Genera las matrices de transicion para este escenario
            arrVFinal, arrPoliticaFinal = fnValueIteration(lstMapa, dictPosEstado, dictEstadoPos, lstPActual, fGamma=fLambda)  # Obtiene la politica optima para este escenario
            lstRecompensas = [fnSimularEpisodio(lstMapa, arrPoliticaFinal, dictEstadoPos, dictPosEstado, lstPActual) for _ in range(10)]  # Simula 10 episodios y recopila recompensas obtenidas
            fPromedioRecompensa = np.mean(lstRecompensas)  # Calcula la recompensa promedio para este escenario
            lstRecompensasProm.append(fPromedioRecompensa)

        plt.subplot(2, 2, iIndiceLambda+1)  # Crea una subplot para graficar resultados de este lambda
        plt.bar([fProbExito for fProbExito, _ in lstProbabilidades], lstRecompensasProm, color='green', width=0.08)  # Grafica las recompensas promedio para cada prob_exito
        plt.title(f"Recompensa vs Prob. Exito (lambda = {fLambda})")  # Titulo del grafico indicando el valor de lambda
        plt.xlabel("Probabilidad de exito")  # Etiqueta del eje X
        plt.ylabel("Recompensa promedio")  # Etiqueta del eje Y
        plt.ylim(900, 1900)  # Define limites del eje Y
        for j, v in enumerate(lstRecompensasProm):
            plt.text(lstProbabilidades[j][0], v + 5, f"{v:.1f}", ha='center', color='red')  # Muestra los valores de recompensa sobre cada barra

    plt.tight_layout()  # Ajusta el layout para que los subplots no se sobrepongan
    plt.show()  # Muestra en pantalla todas las graficas
    input("Simulacion completa. Presiona Enter para salir.")  # Pausa final hasta que el usuario presione Enter
