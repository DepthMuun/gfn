# Guía de Git y Ramas

Estrategia de branching para mantener la estabilidad del framework Geodesic Flow Networks.

## Ramas Principales

- **`main`**: El estado "production-ready". Solo versiones estables (v2.7.x).
- **`dev`**: La rama de desarrollo principal. Todos los features se integran aquí primero.

## Ramas de Trabajo (Features/Fixes)

Usa prefijos para identificar el propósito de la rama:

- `feat/nombre-feature`: Nuevas capacidades.
- `fix/error-especifico`: Corrección de fallos.
- `docs/aspecto-documentado`: Mejoras puras de documentación.
- `refactor/area-optimizada`: Cambios que no alteran el comportamiento.

### Ejemplo de flujo de comandos:

```bash
# 1. Empezar en dev
git checkout dev
git pull origin dev

# 2. Crear rama de trabajo
git checkout -b feat/dynamic-integrator

# 3. (Desarrollo y commits...)

# 4. Integrar (si no usas PRs en GitHub)
git checkout dev
git merge feat/dynamic-integrator
git branch -d feat/dynamic-integrator
```

## Mensajes de Commit
Usa mensajes claros y descriptivos. Recomendamos el formato:
`tipo: descripción breve` (ej: `fix: corrige desbordamiento en proyector bilateral`).
