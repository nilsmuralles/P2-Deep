# P2-Deep — Detección de lavado de dinero en remesas Guatemala–EE. UU.

## Orden de ejecución

1. `01_ingenieria_datos.ipynb`: genera `C1_output_*` (secuencias, meta, features).
2. `02_aprendizaje_normalidad.ipynb`: entrena el autoencoder de Etapa A y genera `stage_a_output/`.
3. `03_clasificador_etapa_b.ipynb`: clasificador de Etapa B, ablación y sistema final. Genera `stage_b_output/`

## Enlaces

- **Reporte ejecutivo:** ver PDF entregado en la plataforma del curso.
- **MVP (interfaz funcional):** https://claude.ai/artifact/5x4mUvWFC7admu8ND7sZqb