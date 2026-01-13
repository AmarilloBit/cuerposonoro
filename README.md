# cuerposonoro
Proyecto software que transforma el movimiento del cuerpo humano en sonido


### MPE Features

El usuario quiere un resumen claro y en texto de todas las features que hemos diseñado para el sistema MPE antes de empezar a implementarlas. Voy a organizar esto de forma clara, separando acordes, melodía y controles globales.Resumen de Features para MPE - Cuerpo Sonoro
ACORDES (Parte inferior del cuerpo)
1. Selección de acorde - Posición de pies en eje X
Los pies determinan qué acorde suena. El espacio capturado por la cámara se divide en 4 zonas horizontales. Caminar hacia la izquierda o derecha cambia el acorde. Usamos el punto medio entre ambos tobillos para calcular la posición. Escala Do Mayor con grados I, IV, V y VI en triadas.
2. Expresividad del acorde - Inclinación de cadera
La inclinación lateral de la cadera controla dos cosas: el pitch bend (desafinación sutil que crea tensión) y la adición de notas de extensión. Cuando la inclinación es moderada, se aplica pitch bend. Cuando es extrema, se añaden notas de 6ª o 7ª al acorde para enriquecerlo armónicamente.
3. Volumen del acorde - Ángulo de rodillas
El ángulo de flexión de las rodillas controla el volumen. Piernas estiradas significa volumen máximo. Flexionar las rodillas reduce el volumen progresivamente. Esto permite hacer crescendos y diminuendos con el cuerpo de forma natural.

MELODÍA (Parte superior del cuerpo)
4. Nota melódica mano derecha - Altura Y de la mano derecha
La altura vertical de la mano derecha en el espacio determina qué nota suena en la octava grave (C3 a B3). Mano abajo toca notas graves, mano arriba toca notas agudas dentro de esa octava.
5. Nota melódica mano izquierda - Altura Y de la mano izquierda
Igual que la mano derecha pero en la octava aguda (C5 a B5). Esto permite tocar melodías a dos voces separadas por dos octavas.
6. Disparo de nota - Movimiento brusco de mano/muñeca
Las notas no suenan continuamente. Se disparan únicamente cuando se detecta un movimiento brusco (jerk alto) en la mano o muñeca. Esto da un control percusivo y expresivo sobre cuándo suenan las notas.
7. Intensidad y duración - Velocidad del brazo
La velocidad con la que se mueve el brazo al hacer el gesto determina dos cosas: la velocity MIDI (qué tan fuerte suena la nota) y la duración de la nota. Movimiento rápido produce notas fuertes y cortas (staccato). Movimiento lento produce notas suaves y más largas.
8. Glissando y vibrato - Ángulo codo-cadera
El ángulo que forma el brazo respecto al torso (medido entre codo y cadera) controla el pitch bend de las notas melódicas. Brazo pegado al cuerpo significa nota estable. Brazo extendido aplica glissando. Movimiento oscilante del codo genera vibrato.

CONTROL GLOBAL
9. Filtro global - Inclinación de cabeza
La inclinación lateral de la cabeza (tilt) controla un filtro de frecuencia global que afecta a todo el sonido. Cabeza recta es sonido neutro. Inclinación hacia un lado oscurece el sonido, hacia el otro lado lo hace más brillante.
10. Texturas y drones - Energy global
La energía general del movimiento del cuerpo (ya implementada) se sigue enviando a SuperCollider para controlar las texturas y drones de fondo que complementan la parte melódica y armónica del MPE.

Nuevas features

## Nuevas features a implementar en features.py

| Feature | Landmarks utilizados | Rango de salida |
|---------|----------------------|-----------------|
| `feetCenterX` | Tobillos (27, 28) | 0.0 - 1.0 |
| `hipTilt` | Caderas (23, 24) | -1.0 - 1.0 |
| `kneeAngle` | Cadera, rodilla, tobillo (23/24, 25/26, 27/28) | 0.0 - 1.0 |
| `rightHandY` | Muñeca derecha (16) | 0.0 - 1.0 |
| `leftHandY` | Muñeca izquierda (15) | 0.0 - 1.0 |
| `rightHandJerk` | Muñeca derecha (16) velocidad | 0.0 - 1.0 |
| `leftHandJerk` | Muñeca izquierda (15) velocidad | 0.0 - 1.0 |
| `rightArmVelocity` | Muñeca derecha (16) | 0.0 - 1.0 |
| `leftArmVelocity` | Muñeca izquierda (15) | 0.0 - 1.0 |
| `rightElbowHipAngle` | Hombro, codo, cadera (12, 14, 24) | 0.0 - 1.0 |
| `leftElbowHipAngle` | Hombro, codo, cadera (11, 13, 23) | 0.0 - 1.0 |
| `headTilt` | Orejas (7, 8) | -1.0 - 1.0 |
