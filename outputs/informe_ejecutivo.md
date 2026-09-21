# Informe ejecutivo: mercados de café (High Garden Coffee)


*Modelo: claude-sonnet-5 · generado 2026-09-20 14:05 · versión de prompt v3*

## Proyección de +12,4% en el consumo de 51 mercados, concentrada en 7 países prioritarios

La proyección agregada para los 51 mercados modelados apunta a un consumo de 3370.6 M en 2024/25, frente a 2998.9 M en 2019/20, un crecimiento proyectado de 12.4% y un CAGR proyectado de 2.4%, por encima del 1.5% observado en las últimas 5 temporadas. La concentración es alta: los 7 países prioritarios reúnen el 82.0% del consumo y explican 361.9 M del crecimiento neto proyectado total de 371.7 M. El escenario se apoya en los 8 grandes mercados en crecimiento sostenido, que representan el 70.9% del consumo de 2019/20. El backtest sitúa el WAPE en 4.7%, con 5.8% de sobreestimación agregada a 5 temporadas.

**Hallazgos**

- La proyección agregada pasa de 2998.9 M en 2019/20 a 3370.6 M en 2024/25: 12.4% de crecimiento total proyectado y 2.4% de CAGR proyectado frente al 1.5% observado en las últimas 5 temporadas.
- Siete mercados prioritarios (Brazil, Ethiopia, Vietnam, Indonesia, Philippines, Colombia, Mexico) concentran el 82.0% del consumo y 361.9 M del crecimiento neto proyectado total de 371.7 M.
- Los 8 grandes en crecimiento sostenido aportan 70.9% del consumo de 2019/20 y 267.8 M de crecimiento proyectado; los 3 en retroceso restan 29.1 M.
- Los segmentos Maduros estancados (19 países), Nicho plano (10) y Pequeños volátiles (6) suman 10.5 M, 0.1 M y 0.1 M de crecimiento proyectado, respectivamente.
- El modelo registra WAPE de 4.7% frente a 7.7% del baseline ingenuo, una mejora de 39.0%; la sobreestimación agregada a 5 temporadas es de 5.8%.

**Riesgos y límites**

- El dataset no contiene precios ni demanda de importación: solo mide consumo doméstico en países productores, de modo que la proyección no incorpora valor ni comercio exterior.
- Hay alerta de optimismo en Vietnam, Thailand, Tanzania, Nicaragua y Guyana: su proyección supera en más de 3 puntos porcentuales por año su ritmo reciente.
- El backtest muestra 5.8% de sobreestimación agregada a 5 temporadas, por lo que el escenario agregado tiende a quedar por encima de lo observado.
- La unidad del dataset no está declarada (probablemente kg), muchas series son planas y los datos terminan en 2019/20, sin capturar shocks posteriores.

**Próximos pasos**

- Revisar con especial cautela las proyecciones de los países con alerta de optimismo antes de asignar recursos.
- Evaluar los nichos prometedores (Bolivia, Uganda) y los candidatos Thailand y El Salvador dentro del escenario planteado.
- Planificar sobre el escenario del modelo (WAPE 4.7%) asumiendo el 5.8% de sobreestimación agregada del backtest.

---

## Mercados

### Brazil — Priorizar

**Brazil: primer puesto del ranking y consumo doméstico en expansión sostenida**

Brazil es el mercado Prioritario número 1 de 51, con un score de 84,7 sobre 100 y presencia en el top 10 en el 99,0 % de los escenarios de pesos simulados. Su consumo doméstico pasó de 1320,0 M en 2019/20 a una proyección de 1466,3 M para 2024/25, con una serie dinámica y sin temporadas planas. La proyección es coherente con el ritmo reciente, sin alerta de optimismo.

*A favor:* Primer puesto entre 51 países y score de 84,7 sobre 100, con 99,0 % de los escenarios simulados dentro del top 10. Serie dinámica, sin temporadas planas, con crecimiento proyectado de 146,3 M hasta 2024/25. Error típico del método de 2,7 %, el más contenido del análisis, y brecha frente al ritmo reciente de solo 0,3 puntos porcentuales por año.

*Riesgos:* Brecha de proyección frente al ritmo reciente de 0,3 puntos porcentuales por año: la proyección es algo más optimista que la tendencia observada, no una fortaleza. Error típico del método de 2,7 %: no indica dirección, la proyección podría desviarse por encima o por debajo. El rango del 80 % abarca de 1196,6 M a 1647,3 M para 2024/25, una amplitud que exige no tratar la proyección como certeza.

*Por qué:* Nivel Prioritario y primer puesto del ranking, con 99,0 % de escenarios en el top 10 y brecha de solo 0,3 puntos porcentuales frente al ritmo reciente.

### Ethiopia — Priorizar

**Etiopía se consolida como mercado prioritario de café Arábica con crecimiento sostenido y ranking robusto.**

Etiopía ocupa la posición 2 de 51 en el ranking, con un score de 78,7 sobre 100 y una cuota de consumo total de 7,6%. El consumo pasó de 226,9 M en 2019/20 a una proyección de 256,3 M para 2024/25, dentro de un rango entre 209,1 M y 287,9 M.

*A favor:* Ranking robusto: en el 100,0% de los escenarios simulados de pesos, Etiopía se mantiene en el top 10. Crecimiento proyectado de 29,4 M hacia 2024/25, con una tasa proyectada de 2,5% anual (CAGR proyectado) Serie de consumo sin años planos (0,0%) y bajo error típico de proyección de 1,3%, lo que da mayor confianza al pronóstico.

*Riesgos:* La proyección supera el ritmo reciente en 0,6 puntos porcentuales por año, lo que introduce un componente de optimismo a validar frente al crecimiento reciente observado de 1,9%. El rango de proyección para 2024/25 es amplio, entre 209,1 M y 287,9 M, lo que refleja incertidumbre relevante en el horizonte de 5 temporadas.

*Por qué:* Etiopía es un mercado prioritario con ranking muy robusto (100,0% en top 10) y sin alerta de optimismo, lo que respalda una proyección de crecimiento sostenido en el segmento de grandes mercados.

### Vietnam — Validar antes de invertir

**Vietnam, prioritario: tercera posición entre 51 mercados con proyección de 220,2 M**

Vietnam ocupa la posición 3 de 51 mercados con un score de 78,6 sobre 100 y una cuota del 5,3% del consumo total de la muestra. El escenario proyecta pasar de 159,0 M en 2019/20 a 220,2 M en 2024/25, con un crecimiento proyectado de 61,2 M y una serie dinámica.

*A favor:* Posición 3 de 51 en el ranking, con score de 78,6 sobre 100 y presencia en el top 10 en el 85,0% de los escenarios de pesos simulados. Proyección de 220,2 M frente a 159,0 M en 2019/20: un crecimiento proyectado de 61,2 M en el horizonte. Cuota del 5,3% del consumo total y trayectoria de largo recorrido, desde 9,0 M en 1990/91 y 72,5 M en 2009/10.

*Riesgos:* Alerta de optimismo activa: la proyección supera en 3,0 puntos porcentuales por año el ritmo reciente, que fue del 3,7% frente al 6,7% proyectado. Intervalo del 80% amplio en 2024/25: entre 179,7 M y 247,4 M, por debajo y por encima de la proyección central de 220,2 M. Error típico del método del 11,0% en este mercado, y 3,0% de años planos en una serie clasificada como dinámica.

*Por qué:* El mercado es prioritario y la proyección supera en 3,0 puntos porcentuales por año el ritmo reciente, por lo que conviene validar el escenario antes de comprometer inversión.

### Indonesia — Priorizar

**Indonesia se consolida como mercado prioritario con crecimiento sostenido y ranking robusto.**

Indonesia ocupa la posición 4 de 51 países con un puntaje de 78,2 sobre 100, y su consumo pasaría de 288,4 M en 2019/20 a una proyección de 348,0 M para 2024/25. La serie es dinámica y el país aparece en el top 10 en el 96,0% de los escenarios simulados, lo que da solidez a su ubicación en el ranking.

*A favor:* Segundo lugar consolidado en el segmento "Grandes en crecimiento sostenido", con cuota de consumo total de 9,6% sobre el conjunto de países. Robustez de ranking alta: en el 96,0% de las combinaciones de pesos simuladas, Indonesia se mantiene en el top 10. Crecimiento proyectado de 59,6 M hacia 2024/25, con una tasa de crecimiento anual compuesta proyectada de 3,8%.

*Riesgos:* La proyección supera el ritmo reciente de consumo en 2,4 puntos porcentuales por año (brecha proyección vs. reciente), lo que introduce un margen de optimismo a vigilar, aunque no se activó la alerta formal. El error típico del método de proyección en este país es de 4,5%, por lo que el resultado final podría desviarse de la cifra puntual proyectada. El rango de escenario para 2024/25 va de 284,0 M a 390,9 M, una amplitud considerable que debe tenerse en cuenta al planificar.

*Por qué:* El país es Prioritario, con score alto, robustez de ranking del 96,0% y crecimiento proyectado sostenido, sin alerta de optimismo activada, lo que respalda una recomendación de priorización.

### Philippines — Priorizar

**Filipinas, puesto 5 de 51 y prioridad alta en café Robusta/Arabica**

Filipinas ocupa la posición 5 entre 51 mercados, con un score de 77,9 sobre 100, y queda en el top 10 en el 98,0% de los escenarios de pesos simulados. El consumo doméstico pasó de 195,0 M en 2019/20 a una proyección de 231,2 M en 2024/25, un crecimiento proyectado de 36,2 M (3,5% anual). El nivel asignado es Prioritario.

*A favor:* Posición 5 de 51 con score de 77,9 sobre 100, dentro del segmento de emergentes de alto crecimiento. Presencia en el top 10 en el 98,0% de los escenarios de pesos simulados: ranking robusto. Proyección de 231,2 M en 2024/25, con un crecimiento proyectado de 36,2 M frente a 195,0 M en 2019/20.

*Riesgos:* La proyección supera en 1,0 puntos porcentuales por año el ritmo reciente (3,5% frente a 2,5%): es un riesgo de optimismo, aunque no activa la alerta. Rango del intervalo del 80% amplio: de 188,7 M a 259,7 M en 2024/25, lo que deja margen a escenarios por debajo del consumo de partida. El método presenta un error típico de 3,0% en este mercado; no indica dirección, pero exige validación adicional.

*Por qué:* Nivel Prioritario, posición 5 de 51, score de 77,9 sobre 100 y 98,0% de escenarios en el top 10 sostienen la prioridad, sin alerta de optimismo activada.

### Colombia — Priorizar

**Colombia se consolida como mercado prioritario con proyección de crecimiento sostenido en café arábica.**

Colombia ocupa la posición 6 de 51 en el ranking, con un puntaje de 71,3 sobre 100 y una cuota sobre el consumo total de 4,1%. El consumo pasó de 121,5 M en 2019/20 a una proyección de 142,3 M para 2024/25, dentro de un escenario con rango entre 116,1 M y 159,9 M.

*A favor:* Nivel prioritario con posición 6 de 51 en el ranking y puntaje de 71,3 sobre 100. El país aparece en el top 10 en el 87,0% de los escenarios simulados, lo que indica un ranking robusto frente a distintos pesos. Serie de tipo dinámica con 0,0% de temporadas planas, sin señal de estancamiento en el historial.

*Riesgos:* El error típico de backtest del método para este país es de 6,4%, sin indicar si tiende a sobreestimar o subestimar. La brecha entre la proyección y el ritmo reciente es de -0,4 puntos porcentuales por año, por lo que la proyección es levemente menos optimista que la tendencia reciente observada (crecimiento reciente de 3,6% frente a un CAGR proyectado de 3,2%).

*Por qué:* El nivel prioritario, la alta robustez del ranking (87,0% en top 10) y la ausencia de alerta de optimismo respaldan priorizar este mercado en el escenario proyectado.

### Thailand — Validar antes de invertir

**Thailand ocupa la posición 7 en el ranking con proyección de 5,0% anual, pero el escenario supera el ritmo reciente**

Thailand registró 84,0 M en 2019/20 (cuota del 2,8%) y el escenario apunta a 107,4 M para 2024/25, con rango de incertidumbre entre 87,6 M y 120,6 M. El país mantiene ranking en el top 10 en el 63,0% de las simulaciones de pesos. La proyección implica 5,0% anual.

*A favor:* Crecimiento histórico sostenido: de 11,3 M en la primera temporada a 84,0 M en 2019/20, serie dinámica con solo 28,0% de años planos. Posición 7 de 51 países y cuota material del 2,8% del consumo total del grupo analizado. Proyección de 23,4 M adicionales hacia 2024/25 con tasa anual del 5,0%.

*Riesgos:* La proyección supera el ritmo reciente en 3,2 puntos porcentuales por año (alerta de optimismo activa). Error típico del método en este país de 13,2%, con rango de incertidumbre amplio (87,6 M a 120,6 M). El 63,0% de escenarios de pesos mantiene al país en top 10, señalando sensibilidad moderada del ranking.

*Por qué:* Score de 65,6 en nivel Candidato y brecha de 3,2 pp entre proyección y ritmo reciente justifican validación sobre terreno antes de comprometer recursos.

### Mexico — Priorizar

**México: mercado prioritario con base sólida y crecimiento proyectado moderado**

México ocupa la posición 8 entre 51 países productores, con un consumo doméstico de 145,5 M en 2019/20 que representa el 4,9% del total. La proyección para 2024/25 alcanza 153,9 M, con un crecimiento de 8,4 M y una tasa anual de 1,1%. El escenario presenta robustez: en el 73,0% de simulaciones de pesos el país permanece en el top 10.

*A favor:* Base de consumo material de 145,5 M con cuota del 4,9%, entre los ocho mercados más grandes del universo analizado. Ranking robusto: en el 73,0% de escenarios simulados el país se mantiene en el top 10, señalando estabilidad estructural. Serie histórica desde 1990/91 con comportamiento dinámico: solo el 24,0% de años planos, reflejando capacidad de respuesta al entorno.

*Riesgos:* El rango de proyección es amplio: de 125,6 M a 172,9 M, generando incertidumbre operativa para dimensionar volúmenes de entrada. El error histórico del método en este país es de 2,6%, introduciendo margen de variabilidad en el escenario central de 153,9 M. Ritmo proyectado de 1,1% anual ligeramente por debajo del reciente (1,2%), sugiriendo moderación en la dinámica de consumo doméstico.

*Por qué:* Score de 62,0 sobre 100, posición 8 de 51 y presencia en el 73,0% de escenarios top 10 configuran un mercado prioritario con volumen material, trayectoria sólida y proyección de crecimiento moderado.

### Bolivia — Explorar como nicho

**Bolivia: nicho prometedor con crecimiento sostenido y proyección alineada a la tendencia histórica**

Bolivia ocupa la posición 9 entre 51 países analizados, con un score de 61,1 sobre 100, dentro del segmento de grandes mercados en crecimiento sostenido. El consumo doméstico de arábica proyectado para la temporada 2024/25 es de 4,2 M, frente a 3,7 M registrados en 2019/20, con una tasa de crecimiento anual proyectada de 2,6%. La proyección coincide exactamente con el ritmo reciente observado, lo que le otorga coherencia técnica.

*A favor:* Bolivia muestra una trayectoria ascendente de largo plazo: desde 1,5 M en la temporada 1990/91 pasó a 2,8 M en 2009/10 y a 3,7 M en 2019/20, consolidando un historial de crecimiento continuo. El error de backtesting del método es de apenas 0,9%, lo que refleja un ajuste técnico muy preciso para este mercado. En 84,0% de los escenarios simulados con distintas combinaciones de pesos, Bolivia mantiene su posición entre los diez primeros, evidenciando una alta robustez del ranking frente a variaciones metodológicas.

*Riesgos:* Bolivia está clasificada como 'Nicho prometedor', nivel que refleja una cuota sobre el consumo total de apenas 0,1%, lo que limita el volumen absoluto alcanzable en el horizonte analizado. El intervalo de confianza al 80% para la temporada 2024/25 se extiende de 3,4 M a 4,7 M, una banda amplia alrededor del escenario central de 4,2 M que debe tenerse en cuenta al dimensionar cualquier acción comercial. El crecimiento proyectado en términos absolutos es de 0,5 M a lo largo del horizonte de cinco temporadas, un volumen modesto que refuerza el carácter de nicho del mercado.

*Por qué:* El nivel asignado es "Nicho prometedor" con una cuota de consumo total de 0,1%, lo que orienta hacia una exploración selectiva antes de comprometer recursos significativos.

### Uganda — Explorar como nicho

**Uganda: nicho prometedor con crecimiento sostenido que merece seguimiento activo**

Uganda ocupa la posición 10 entre 51 países analizados, con un score de 59,1 sobre 100 y un consumo doméstico proyectado de 17,9 M para la temporada 2024/25, frente a los 15,2 M registrados en 2019/20. La proyección apunta a un crecimiento de 2,7 M bajo un escenario de tasa anual compuesta del 3,3%, con un intervalo del 80% que va de 14,6 M a 20,1 M, lo que refleja una dispersión moderada en torno al escenario central.

*A favor:* Uganda muestra una trayectoria de expansión consistente desde los 4,2 M de la primera temporada con dato hasta los 15,2 M de 2019/20, con crecimiento adicional recogido en la proyección de 17,9 M para 2024/25. La posición en el ranking resulta robusta: en el 78,0 % de los escenarios de pesos simulados el país queda dentro del top 10, lo que indica estabilidad de la clasificación ante distintas hipótesis analíticas. La proyección supera el ritmo reciente en apenas 1,2 puntos porcentuales por año y la alerta de optimismo no se activa, lo que otorga al escenario de crecimiento del 3,3 % una base relativamente conservadora.

*Riesgos:* El nivel asignado es 'Nicho prometedor', lo que implica que el mercado aún no alcanza el umbral de madurez o escala que caracteriza a los mercados prioritarios; la cuota sobre el consumo total es de apenas el 0,5 %. El error típico del método de pronóstico en este país es del 4,5 %, por lo que el escenario central de 17,9 M puede desviarse en esa magnitud sin que ello indique sesgo en ninguna dirección. La proyección supera el ritmo reciente en 1,2 puntos porcentuales por año; aunque este diferencial no activa la alerta de optimismo, sigue siendo una cautela a considerar al interpretar el escenario proyectado.

*Por qué:* El nivel 'Nicho prometedor' y la cuota del 0,5 % limitan la escala actual; el crecimiento proyectado y la robustez del ranking justifican seguimiento cercano antes de comprometer recursos significativos.

### El Salvador — Monitorear

**El Salvador: candidato a seguimiento con proyección moderada y volatilidad contenida**

El Salvador ocupa la posición 11 entre 51 países con un score de 55,3 sobre 100. La proyección sitúa el consumo en 18,8 M para 2024/25, con un crecimiento de 1,3 M y una tasa anual de 1,4 %. Su serie histórica es dinámica y el error de backtest alcanza 1,8 %, señalando estabilidad metodológica.

*A favor:* Ocupa en 40,0 % de escenarios simulados el top 10, reflejando robustez frente a cambios en los pesos del modelo. La serie temporal es dinámica con solo 24,0 % de años planos, indicando evolución sostenida del consumo. El error de backtest de 1,8 % es bajo, otorgando confianza en la precisión del método de pronóstico.

*Riesgos:* La proyección acelera 0,6 puntos porcentuales por año sobre el ritmo reciente de 0,8 %, introduciendo un sesgo optimista moderado. El rango de incertidumbre al 80 % oscila entre 15,3 M y 21,1 M, exigiendo flexibilidad operativa ante escenarios adversos. Cuota de consumo de 0,6 % mantiene al país fuera de los mercados de mayor materialidad global.

*Por qué:* Score de 55,3 y posición 11 lo sitúan como candidato, nivel intermedio que aún no justifica inversión; conviene seguimiento periódico de su evolución.
