# Guía de Desarrollo Humano (GFN Framework)

Este documento define las prácticas de desarrollo para colaboradores humanos en el proyecto GFN. El objetivo es mantener una base de código profesional, escalable y libre de errores técnicos evitables.

## 1. El Ciclo de Desarrollo

Para garantizar un desarrollo "lento y controlado", seguimos estos pasos:

1. **Sincronización**: Antes de empezar, asegúrate de estar en `dev` y tener los últimos cambios.
2. **Aislamiento**: Crea una rama para tu tarea (`feat/` o `fix/`). No trabajes nunca sobre `main` o `dev` directamente.
3. **Desarrollo Atómico**: Realiza cambios pequeños y enfocados. Si una tarea es muy grande, divídela en sub-tareas.
4. **Validación de Rigor**: 
   - **Test Específico**: Ejecuta al menos un test que verifique directamente tu cambio.
   - **Suite Completa**: Ejecuta toda la suite de pruebas (excluyendo benchmarks) para asegurar que no hay regresiones.
     - Comando: `python -m pytest tests/ --ignore=tests/gssm/benchmarks --ignore=tests/isn/benchmarks`
5. **Revisión**: En un entorno de equipo, abre un Pull Request (PR) hacia `dev`.

## 2. Desarrollo "Lento y Controlado"
- **Calidad > Velocidad**: Es preferible tardar un día más y entregar código testeado que arreglar bugs en producción.
- **Refactorización Continua**: Si tocas un archivo y ves algo que puede mejorar sin romper nada, hazlo (pero en un commit separado si es posible).
- **Sin Hotfixes ciegos**: Todo fix debe ser probado en una rama antes de mergear.

## 3. Integración con el Agente AI
El proyecto cuenta con un sistema de memoria interna en la carpeta `/memory/`.
- **Nota**: Como desarrollador humano, **no necesitas editar los archivos en `/memory/`**.
- El Agente AI se encarga de mantener esa bitácora actualizada basándose en tus cambios y nuestras sesiones de par-programming. Tu foco debe ser el código en `gfn/` y la documentación en `docs/`.
