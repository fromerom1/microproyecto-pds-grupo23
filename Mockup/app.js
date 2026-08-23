// Configuración global de Chart.js para diseño premium
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#5F6368';
Chart.defaults.scale.grid.color = '#DADCE0';
Chart.defaults.elements.bar.borderRadius = 4;
Chart.defaults.plugins.tooltip.backgroundColor = '#202124';
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;

// Colores de la paleta
const COLORS = {
    high: '#D32F2F',      // Rojo intenso
    medium: '#E57373',    // Rojo más claro (reemplaza amarillo)
    low: '#1A73E8',       // Azul (Primary) para Riesgo Bajo
    primary: '#1A73E8',
    primaryLight: '#E8F0FE',
    bg: '#F8F9FA'
};

// --- NAVEGACIÓN SPA ---
function navigateTo(viewId, navElement) {
    // Actualizar menú
    document.querySelectorAll('.nav-links li').forEach(el => el.classList.remove('active'));
    if(navElement) navElement.classList.add('active');

    // Ocultar todas las vistas
    document.querySelectorAll('.view').forEach(el => {
        el.classList.remove('active');
    });

    // Mostrar la vista solicitada
    document.getElementById(`view-${viewId}`).classList.add('active');

    // Si navegamos a detalle, asegurar que el botón de menú sea visible (aunque no esté activo por defecto)
    if(viewId === 'detalle') {
        document.getElementById('nav-detalle').style.display = 'flex';
        document.getElementById('nav-detalle').classList.add('active');
        document.querySelectorAll('.nav-links li')[0].classList.remove('active');
    } else {
        document.getElementById('nav-detalle').style.display = 'none';
    }
}

// --- DATOS Y RENDERIZADO DE TABLA (VISTA 1) ---
const patientsData = [
    { id: '#10245', edad: '6 meses', peso: '1850g', sga: 'Sí', riesgo: 85 },
    { id: '#10248', edad: '4 meses', peso: '2100g', sga: 'Sí', riesgo: 72 },
    { id: '#10230', edad: '12 meses', peso: '2400g', sga: 'No', riesgo: 45 },
    { id: '#10255', edad: '2 meses', peso: '2800g', sga: 'No', riesgo: 12 },
    { id: '#10212', edad: '18 meses', peso: '3100g', sga: 'No', riesgo: 8 }
];

function renderTable() {
    const tbody = document.getElementById('patientsTableBody');
    tbody.innerHTML = '';
    
    patientsData.forEach(p => {
        let badgeClass = 'badge-blue';
        let riesgoTexto = 'Bajo';
        
        if (p.riesgo >= 70) { badgeClass = 'badge-red'; riesgoTexto = 'Alto'; }
        else if (p.riesgo >= 40) { badgeClass = 'badge-light-red'; riesgoTexto = 'Medio'; }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${p.id}</strong></td>
            <td>${p.edad}</td>
            <td>${p.peso}</td>
            <td>${p.sga}</td>
            <td>
                <span class="badge ${badgeClass}">${p.riesgo}% (${riesgoTexto})</span>
            </td>
            <td>
                <button class="btn-icon" onclick="navigateTo('detalle')" title="Ver Detalle">
                    <span class="material-symbols-rounded">visibility</span>
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}


// --- INICIALIZACIÓN DE GRÁFICOS ---
let charts = {};

function initCharts() {
    // 1. Gráfico de Dona (Cohorte)
    const ctxDonut = document.getElementById('riskDonutChart').getContext('2d');
    charts.donut = new Chart(ctxDonut, {
        type: 'doughnut',
        data: {
            labels: ['Riesgo Alto', 'Riesgo Medio', 'Riesgo Bajo'],
            datasets: [{
                data: [67, 100, 166],   // suma 333, coherente con los indicadores
                backgroundColor: [COLORS.high, COLORS.medium, COLORS.low],
                borderWidth: 4,
                borderColor: COLORS.bg,
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20 } }
            }
        }
    });

    // 2. Gráfico de Barras Apiladas (Distribución por Edad)
    const ctxBar = document.getElementById('admissionsBarChart').getContext('2d');
    charts.bar = new Chart(ctxBar, {
        type: 'bar',
        data: {
            // Tramos de seguimiento de la cohorte (333 bebes en total).
            // La cohorte no tiene registros de etapa gestacional: la primera visita es el nacimiento.
            labels: ['Nacimiento', '0-6 meses', '6-12 meses', '12-18 meses', '18-24 meses'],
            datasets: [
                {
                    label: 'Riesgo Alto',
                    data: [6, 9, 13, 20, 22],
                    backgroundColor: COLORS.high
                },
                {
                    label: 'Riesgo Medio',
                    data: [12, 17, 21, 24, 26],
                    backgroundColor: COLORS.medium
                },
                {
                    label: 'Riesgo Bajo',
                    data: [49, 45, 39, 30, 27],
                    backgroundColor: COLORS.low
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    beginAtZero: true, 
                    stacked: true,
                    border: {display: false} 
                },
                x: { 
                    stacked: true,
                    grid: {display: false}, 
                    border: {display: false} 
                }
            },
            plugins: { 
                legend: { 
                    display: true, 
                    position: 'bottom',
                    labels: { usePointStyle: true, boxWidth: 6, font: {size: 11} }
                } 
            }
        }
    });

    // 3. Medidor de Riesgo (Half-Circle)
    const ctxGauge = document.getElementById('gaugeChart').getContext('2d');
    charts.gauge = new Chart(ctxGauge, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [78, 22],
                backgroundColor: [COLORS.high, COLORS.primaryLight],
                borderWidth: 0,
                circumference: 180,
                rotation: 270
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '80%',
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });

    // 4. Gráfico SHAP (Barras Horizontales)
    const ctxShap = document.getElementById('shapChart').getContext('2d');
    charts.shap = new Chart(ctxShap, {
        type: 'bar',
        data: {
            labels: ['SGA', 'Bajo Peso', 'Riqueza Q1', 'Edad Gest.', 'Sexo femenino', 'Sin depresión'],
            datasets: [{
                label: 'Contribución al riesgo (puntos porcentuales)',
                // Positivo = aumenta el riesgo · Negativo = lo reduce
                data: [25, 15, 8, 5, -6, -9],
                backgroundColor: (ctx) => ctx.raw >= 0 ? COLORS.high : COLORS.low,
                borderRadius: 8
            }]
        },
        options: {
            indexAxis: 'y', // Horizontal
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { min: -15, max: 30, border: {display: false},
                     title: { display: true, text: 'Reduce el riesgo  <-  0  ->  Aumenta el riesgo' } },
                y: { grid: {display: false}, border: {display: false} }
            },
            plugins: { legend: { display: false } }
        }
    });

    // 5. Gráfico de Líneas (LAZ Tracking)
    const ctxLaz = document.getElementById('lazChart').getContext('2d');
    
    // Generar labels 0 a 24 meses
    const months = Array.from({length: 25}, (_, i) => i);
    
    // Función para calcular el Z-score (cae 0.13 por mes)
    const getScore = (m) => m === 0 ? -0.5 : -0.5 - (m * 0.13);

    // Definir el mes actual (hasta donde tenemos datos reales)
    const currentMonth = 6;

    const observadoData = months.map(m => m <= currentMonth ? getScore(m) : null);
    const proyeccionData = months.map(m => m >= currentMonth ? getScore(m) : null);

    // Array lleno de -2 para la línea de umbral
    const thresholdLine = Array(25).fill(-2.0);

    charts.laz = new Chart(ctxLaz, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Realidad (Observado)',
                    data: observadoData,
                    borderColor: COLORS.primary,
                    backgroundColor: 'rgba(26, 115, 232, 0.1)',
                    borderWidth: 4,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: COLORS.primary,
                    pointRadius: 4,
                    pointHoverRadius: 8
                },
                {
                    label: 'Proyección del Modelo',
                    data: proyeccionData,
                    borderColor: COLORS.medium,
                    backgroundColor: 'rgba(229, 115, 115, 0.1)',
                    borderWidth: 4,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: COLORS.medium,
                    pointRadius: 4,
                    pointHoverRadius: 8
                },
                {
                    label: 'Umbral Crítico (-2.0)',
                    data: thresholdLine,
                    borderColor: COLORS.high,
                    borderWidth: 2,
                    borderDash: [10, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { 
                    title: { display: true, text: 'Meses de seguimiento' },
                    grid: { display: false }
                },
                y: { 
                    title: { display: true, text: 'Puntaje Z (LAZ)' },
                    min: -4,
                    max: 1
                }
            },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, padding: 20 } }
            }
        }
    });
}

// Los iconos son ligaduras tipograficas: si la fuente no carga, el navegador imprime
// el nombre literal ("bar_chart", "person_add") en grande. Para evitarlo se ocultan por
// CSS y solo se revelan si la fuente esta realmente disponible.
//
// document.fonts.check() no sirve aqui: devuelve true aunque la familia no exista,
// porque cuenta la fuente de reemplazo. Se mide el ancho: con la fuente cargada la
// ligadura ocupa un glifo (~1em); sin ella ocupa las 9 letras de "bar_chart".
function iconosDisponibles() {
    const probe = document.createElement('span');
    probe.className = 'material-symbols-rounded';
    probe.textContent = 'bar_chart';
    probe.style.cssText = 'position:absolute;visibility:hidden;font-size:24px;white-space:nowrap';
    document.body.appendChild(probe);
    const ancho = probe.offsetWidth;
    probe.remove();
    return ancho > 0 && ancho < 60;   // un glifo mide ~24px; el texto crudo pasa de 100px
}

function resolverIconos() {
    if (iconosDisponibles()) document.documentElement.classList.add('iconos-listos');
}

if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(resolverIconos).catch(resolverIconos);
    setTimeout(resolverIconos, 3000); // red lenta: un reintento
} else {
    resolverIconos();
}

// Iniciar
window.onload = () => {
    renderTable();
    initCharts();
};
