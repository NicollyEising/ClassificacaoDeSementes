$(document).ready(function () {
  // Inicializa os dropdowns do Semantic UI
  $('.ui.dropdown').dropdown();

  // Controle das seções do menu lateral
  $('#menu .item .menu .item').on('click', function () {
    const section = $(this).data('section');
    $('.content-section').hide();
    $('#' + section).show();
  });



  // Configuração do gráfico ApexCharts
  const options = {
    chart: {
      type: "area",
      height: 250,
      toolbar: { show: false }
    },
    series: [{
      name: 'Crescimento',
      data: [8000, 15000, 35000, 55000, 7000, 18000, 50000, 70000]
    }],
    dataLabels: {
      enabled: false // Remove os números fixos sobre o gráfico
    },
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.3,
        opacityTo: 0.05,
        stops: [0, 100],
        colorStops: [
          {
            offset: 0,
            color: "#4ade80", // verde claro
            opacity: 0.3
          },
          {
            offset: 100,
            color: "#4ade80",
            opacity: 0.05
          }
        ]
      }
    },
    stroke: {
      curve: 'straight',
      width: 2,
      dashArray: 4,
      colors: ['#4ade80']
    },
    xaxis: {
      categories: ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
      labels: {
        style: { fontSize: '12px' }
      }
    },
    yaxis: {
      show: false // Oculta o eixo Y (números azuis à esquerda)
    },
    tooltip: {
      enabled: true,
      y: {
        formatter: function (val) {
          return val.toLocaleString();
        }
      }
    },
    grid: {
      show: false
    }
  };

  // Renderiza o gráfico
  const chartElement = document.getElementById("area-chart");
  if (chartElement && typeof ApexCharts !== 'undefined') {
    const chart = new ApexCharts(chartElement, options);
    chart.render();
  }

  const sidebars = document.getElementsByClassName('sidebar-column');
  const toggleButton = document.getElementById('toggleMenu');
  
  if (toggleButton) {
    toggleButton.addEventListener('click', function () {
      for (let i = 0; i < sidebars.length; i++) {
        sidebars[i].style.display = (sidebars[i].style.display === 'none' || sidebars[i].style.display === '') 
          ? 'block' 
          : 'none';
      }
    });
  }
});
