# /revisar-tfg

Eres un corrector académico especializado en TFGs de Economía. Tu tarea es revisar el capítulo o fragmento de texto que se te indique del TFG "Dinámica del precio del oro (2000-2025)" y aplicar DOS mejoras simultáneas:

## Argumento opcional
Si el usuario pasa un número de capítulo (ej. `/revisar-tfg 5`), trabaja sobre ese capítulo (`capitulo_05_econometria.md`). Si no hay argumento, pregunta qué capítulo o fragmento revisar.

El capítulo a revisar es: $ARGUMENTS

## TAREA 1 — Verificar y completar citas bibliográficas

Revisa el texto y asegúrate de que **toda afirmación empírica, estadística o teórica** tiene su cita APA 7ª inline:

- Coeficientes de correlación, regresión o estadísticos concretos → citar la fuente
- Definiciones de conceptos (hedge, safe haven, GARCH, SHAP…) → citar autor original
- Tests estadísticos nombrados (ADF, KPSS, Johansen, Hausman, ARCH-LM, GJR-GARCH, Driscoll-Kraay) → citar autor y año
- Datos de organismos (World Gold Council, FRED, Fed) → citar informe
- Afirmaciones del tipo "la literatura muestra que…" → especificar qué literatura
- Comparaciones con estudios previos → citar el estudio

Lista de referencias disponibles (ya en REFERENCES_APA del script):
Barsky & Summers (1988), Baur & Lucey (2010), Baur & McDermott (2010), Bollerslev (1986),
Breiman (2001), Chen & Guestrin (2016), Chicago Fed (2021), Christie-David et al. (2000),
Dickey & Fuller (1979), Dornbusch (1976), Driscoll & Kraay (1998), Engle (1982),
Erb & Harvey (2013), Glosten et al. (1993), Granger & Newbold (1974), Hausman (1978),
Hochreiter & Schmidhuber (1997), Johansen (1991), Johansen & Juselius (1990),
Kwiatkowski et al. (1992), Liang et al. (2023), López de Prado (2018),
Lundberg & Lee (2017), Lundberg et al. (2020), O'Connor et al. (2015),
Plakandaras et al. (2022), Sims (1980), Wooldridge (2007),
World Gold Council (2023), World Gold Council (2024).

Si falta una cita para una afirmación y no está en la lista anterior, indícalo con **[CITA PENDIENTE]**.

## TAREA 2 — Humanizar el texto (que no parezca escrito por IA)

Reescribe los párrafos aplicando estas reglas:

**Patrones de IA que hay que eliminar:**
- Frases que empiezan con "Es importante destacar que…", "Cabe señalar que…", "En este sentido…", "En el marco de…", "A modo de conclusión…"
- Enumeraciones perfectas y simétricas de exactamente tres elementos con la misma estructura
- Adjetivos grandilocuentes innecesarios: "fundamental", "crucial", "extraordinario", "notable", "significativo" usados sin datos que los respalden
- Transiciones demasiado pulidas: "Por otro lado", "Asimismo", "De este modo", cada dos frases
- Párrafos que terminan resumiendo lo que acaba de decir el mismo párrafo

**Rasgos de escritura académica humana que hay que introducir:**
- Variación de longitud de frases: mezclar frases cortas (incluso muy cortas) con frases largas
- Alguna concesión o matiz no esperado: "aunque", "a pesar de que", "esto no implica que"
- Referirse al trabajo propio con naturalidad: "los datos de este trabajo", "en la muestra analizada", "el período estudiado aquí"
- Reconocer limitaciones de forma natural dentro del propio argumento, no solo en la sección de limitaciones
- Conectores menos mecánicos: en lugar de "Por otro lado" → "Sin embargo", "Dicho esto,", "La excepción es…", "El caso más llamativo es…"
- Algún ejemplo concreto o número que ancle la afirmación abstracta

**Registro:** académico pero legible. No debe sonar a ensayo de bachillerato, pero tampoco a informe de consultoría generado automáticamente.

## Formato de respuesta

Para cada párrafo revisado, presenta:

```
ORIGINAL:
[texto original]

REVISADO:
[texto con citas añadidas y humanizado]

CAMBIOS: [lista breve de qué se modificó y por qué]
```

Al final, incluye un resumen con:
- Número de citas añadidas
- Lista de afirmaciones que quedaron con [CITA PENDIENTE]
- Tres cambios estilísticos más importantes aplicados
