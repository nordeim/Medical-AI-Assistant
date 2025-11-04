"""
Frontend Performance Optimization for Medical AI Applications
Implements lazy loading, code splitting, and medical UI optimizations
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class MedicalComponentMetrics:
    """Performance metrics for medical components"""
    component_name: str
    load_time: float
    render_time: float
    memory_usage: float
    bundle_size: float
    first_paint: float
    interactive_time: float

class MedicalAppOptimizer:
    """
    Optimizes frontend performance for medical AI applications
    Focuses on fast loading and smooth UI interactions
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.component_configs = {}
        self.performance_thresholds = {
            'first_paint': 1.5,  # seconds
            'interactive': 3.0,   # seconds
            'bundle_size': 500,   # KB
            'component_load': 0.5  # seconds
        }
    
    def optimize_vite_config(self) -> Dict[str, Any]:
        """
        Generate optimized Vite configuration for medical AI frontend
        """
        vite_config = {
            "build": {
                "target": "es2020",
                "minify": "terser",
                "sourcemap": False,
                "rollupOptions": {
                    "output": {
                        "manualChunks": {
                            # Split vendor libraries
                            "react-vendor": ["react", "react-dom"],
                            "medical-ui": ["./src/components/medical", "./src/components/charts"],
                            "charts": ["chart.js", "react-chartjs-2"],
                            "forms": ["./src/components/forms", "react-hook-form"]
                        },
                        "chunkFileNames": "js/[name]-[hash].js",
                        "entryFileNames": "js/[name]-[hash].js",
                        "assetFileNames": "assets/[name]-[hash].[ext]"
                    }
                },
                "terserOptions": {
                    "compress": {
                        "drop_console": True,
                        "drop_debugger": True,
                        "pure_funcs": ["console.log", "console.info"],
                        "pure_getters": True,
                        "unsafe": True,
                        "unsafe_comps": True,
                        "warnings": False
                    }
                },
                "assetsInlineLimit": 4096,
                "chunkSizeWarningLimit": 1000
            },
            "optimizeDeps": {
                "include": [
                    "react",
                    "react-dom",
                    "react-router-dom",
                    "chart.js",
                    "react-chartjs-2",
                    "d3",
                    "lodash",
                    "date-fns"
                ]
            },
            "server": {
                "port": 3000,
                "host": "0.0.0.0",
                "cors": True,
                "proxy": {
                    "/api": {
                        "target": "http://localhost:8080",
                        "changeOrigin": True,
                        "rewrite": (lambda path: path.replace(/^\/api/, ""))
                    }
                }
            }
        }
        
        return vite_config
    
    def generate_lazy_loading_components(self) -> Dict[str, str]:
        """Generate lazy-loaded components for medical AI app"""
        components = {
            # Core medical components
            "PatientDashboard": """
import React, { Suspense, lazy } from 'react';
import { LoadingSpinner } from '../ui/LoadingSpinner';

const PatientDashboard = lazy(() => import('./PatientDashboard'));
const VitalSignsChart = lazy(() => import('./charts/VitalSignsChart'));
const MedicationList = lazy(() => import('./medications/MedicationList'));
const LabResults = lazy(() => import('./lab/LabResults'));

export default function PatientDashboardWrapper() {
  return (
    <div className="patient-dashboard">
      <Suspense fallback={<LoadingSpinner message="Loading dashboard..." />}>
        <PatientDashboard />
      </Suspense>
      <Suspense fallback={<LoadingSpinner message="Loading vital signs..." />}>
        <VitalSignsChart />
      </Suspense>
      <Suspense fallback={<LoadingSpinner message="Loading medications..." />}>
        <MedicationList />
      </Suspense>
      <Suspense fallback={<LoadingSpinner message="Loading lab results..." />}>
        <LabResults />
      </Suspense>
    </div>
  );
}
""",
            "MedicalCharts": """
import React, { Suspense, lazy } from 'react';

const VitalSignsChart = lazy(() => import('./charts/VitalSignsChart'));
const LabTrendsChart = lazy(() => import('./charts/LabTrendsChart'));
const MedicationEfficacyChart = lazy(() => import('./charts/MedicationEfficacyChart'));

export default function MedicalCharts() {
  return (
    <div className="medical-charts">
      <Suspense fallback={<div className="chart-loading">Loading charts...</div>}>
        <VitalSignsChart />
      </Suspense>
      <Suspense fallback={<div className="chart-loading">Loading trends...</div>}>
        <LabTrendsChart />
      </Suspense>
      <Suspense fallback={<div className="chart-loading">Loading efficacy data...</div>}>
        <MedicationEfficacyChart />
      </Suspense>
    </div>
  );
}
""",
            "AIInsights": """
import React, { Suspense, lazy } from 'react';

const DiagnosisAssistant = lazy(() => import('./ai/DiagnosisAssistant'));
const TreatmentRecommendations = lazy(() => import('./ai/TreatmentRecommendations'));
const RiskAssessment = lazy(() => import('./ai/RiskAssessment'));

export default function AIInsights() {
  return (
    <div className="ai-insights">
      <Suspense fallback={<div className="ai-loading">Loading AI assistant...</div>}>
        <DiagnosisAssistant />
      </Suspense>
      <Suspense fallback={<div className="ai-loading">Loading recommendations...</div>}>
        <TreatmentRecommendations />
      </Suspense>
      <Suspense fallback={<div className="ai-loading">Loading risk assessment...</div>}>
        <RiskAssessment />
      </Suspense>
    </div>
  );
}
"""
        }
        return components
    
    def generate_performance_optimized_hooks(self) -> Dict[str, str]:
        """Generate performance-optimized React hooks for medical applications"""
        hooks = {
            "usePatientData": """
import { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

export function usePatientData(patientId) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['patient', patientId],
    queryFn: () => fetchPatientData(patientId),
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
    refetchOnWindowFocus: false
  });

  const optimizedData = useMemo(() => {
    if (!data) return null;
    
    // Only process essential data for initial render
    return {
      id: data.id,
      name: data.name,
      mrn: data.mrn,
      demographics: data.demographics,
      // Lazy load heavy data
      labResults: null,
      medications: null
    };
  }, [data]);

  return {
    data: optimizedData,
    isLoading,
    error,
    refetch,
    loadFullData: () => refetch()
  };
}
""",
            "useMedicalChart": """
import { useState, useEffect, useRef, useMemo } from 'react';

export function useMedicalChart(data, chartType) {
  const canvasRef = useRef(null);
  const [isChartReady, setIsChartReady] = useState(false);
  
  // Memoize chart configuration
  const chartConfig = useMemo(() => {
    const configs = {
      'vital-signs': {
        type: 'line',
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: {
            duration: 300,
            easing: 'easeOutQuart'
          }
        }
      },
      'lab-results': {
        type: 'bar',
        options: {
          responsive: true,
          animation: {
            duration: 500
          }
        }
      }
    };
    
    return configs[chartType] || configs['vital-signs'];
  }, [chartType]);

  useEffect(() => {
    if (!canvasRef.current || !data) return;

    const initChart = async () => {
      // Dynamic import for better performance
      const { Chart } = await import('chart.js');
      
      const ctx = canvasRef.current.getContext('2d');
      new Chart(ctx, {
        type: chartConfig.type,
        data: data,
        options: chartConfig.options
      });
      
      setIsChartReady(true);
    };

    initChart();
  }, [data, chartConfig]);

  return { canvasRef, isChartReady };
}
""",
            "useVirtualizedList": """
import { useState, useEffect, useRef, useCallback } from 'react';

export function useVirtualizedList(items, itemHeight = 60, containerHeight = 400) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef(null);

  // Calculate visible items
  const visibleItems = useMemo(() => {
    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
      startIndex + Math.ceil(containerHeight / itemHeight) + 1,
      items.length
    );
    
    return items.slice(startIndex, endIndex).map((item, index) => ({
      ...item,
      index: startIndex + index
    }));
  }, [items, scrollTop, itemHeight, containerHeight]);

  const totalHeight = items.length * itemHeight;
  const offsetY = scrollTop - (scrollTop % itemHeight);

  const handleScroll = useCallback((e) => {
    setScrollTop(e.target.scrollTop);
  }, []);

  return {
    containerRef,
    visibleItems,
    totalHeight,
    offsetY,
    handleScroll
  };
}
"""
        }
        return hooks
    
    def generate_performance_monitoring_component(self) -> str:
        """Generate component for monitoring frontend performance"""
        return """
import React, { useEffect, useState } from 'react';

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState({
    fcp: null, // First Contentful Paint
    lcp: null, // Largest Contentful Paint
    fid: null, // First Input Delay
    cls: null, // Cumulative Layout Shift
    ttfb: null // Time to First Byte
  });

  useEffect(() => {
    // Monitor Core Web Vitals
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const metricName = entry.name;
        const value = entry.value || entry.processingStart - entry.startTime;
        
        setMetrics(prev => ({
          ...prev,
          [metricName]: value
        }));

        // Log slow performance
        if (value > 2000) {
          console.warn(`Slow ${metricName}: ${value}ms`);
        }
      }
    });

    // Observe different performance entries
    observer.observe({ entryTypes: ['paint', 'largest-contentful-paint'] });
    observer.observe({ entryTypes: ['first-input'] });
    observer.observe({ entryTypes: ['layout-shift'] });

    return () => observer.disconnect();
  }, []);

  // Show metrics in development
  if (process.env.NODE_ENV === 'development') {
    return (
      <div className="performance-monitor">
        <h3>Performance Metrics</h3>
        <ul>
          <li>FCP: {metrics.fcp?.toFixed(2)}ms</li>
          <li>LCP: {metrics.lcp?.toFixed(2)}ms</li>
          <li>FID: {metrics.fid?.toFixed(2)}ms</li>
          <li>CLS: {metrics.cls?.toFixed(3)}</li>
          <li>TTFB: {metrics.ttfb?.toFixed(2)}ms</li>
        </ul>
      </div>
    );
  }

  return null;
}
"""
    
    def generate_optimized_medical_components(self) -> Dict[str, str]:
        """Generate optimized medical-specific components"""
        components = {
            "PatientCard": """
import React, { memo, useMemo } from 'react';

const PatientCard = memo(({ patient, onSelect }) => {
  const displayInfo = useMemo(() => ({
    name: `${patient.firstName} ${patient.lastName}`,
    age: calculateAge(patient.dateOfBirth),
    mrn: patient.medicalRecordNumber,
    lastVisit: formatDate(patient.lastVisit)
  }), [patient]);

  const handleClick = useMemo(() => () => {
    onSelect(patient.id);
  }, [patient.id, onSelect]);

  return (
    <div className="patient-card" onClick={handleClick}>
      <h3>{displayInfo.name}</h3>
      <p>Age: {displayInfo.age}</p>
      <p>MRN: {displayInfo.mrn}</p>
      <p>Last Visit: {displayInfo.lastVisit}</p>
    </div>
  );
});

PatientCard.displayName = 'PatientCard';
export default PatientCard;
""",
            "OptimizedVitalSignsChart": """
import React, { memo, useEffect, useRef, useState } from 'react';

const OptimizedVitalSignsChart = memo(({ data, height = 300 }) => {
  const canvasRef = useRef(null);
  const [chartInstance, setChartInstance] = useState(null);

  useEffect(() => {
    if (!canvasRef.current || !data) return;

    const initChart = async () => {
      const Chart = (await import('chart.js/auto')).default;
      
      const ctx = canvasRef.current.getContext('2d');
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.timestamps,
          datasets: [
            {
              label: 'Heart Rate',
              data: data.heartRate,
              borderColor: 'rgb(255, 99, 132)',
              tension: 0.1
            },
            {
              label: 'Blood Pressure',
              data: data.bloodPressure,
              borderColor: 'rgb(54, 162, 235)',
              tension: 0.1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: {
            duration: 300
          },
          plugins: {
            legend: {
              display: true,
              position: 'top'
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Time'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Value'
              }
            }
          }
        }
      });
      
      setChartInstance(chart);
    };

    initChart();

    return () => {
      if (chartInstance) {
        chartInstance.destroy();
      }
    };
  }, [data]);

  return (
    <div style={{ height }}>
      <canvas ref={canvasRef} />
    </div>
  );
});

OptimizedVitalSignsChart.displayName = 'OptimizedVitalSignsChart';
export default OptimizedVitalSignsChart;
"""
        }
        return components


class MedicalAppPerformanceConfig:
    """
    Complete performance configuration for medical AI frontend
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.optimizer = MedicalAppOptimizer(project_root)
    
    def generate_performance_configs(self) -> Dict[str, str]:
        """Generate all performance optimization configurations"""
        configs = {}
        
        # 1. Vite configuration
        configs['vite.config.js'] = self.generate_vite_config_js()
        
        # 2. Bundle analyzer config
        configs['bundle-analyzer.config.js'] = self.generate_bundle_analyzer_config()
        
        # 3. Service worker for caching
        configs['sw.js'] = self.generate_service_worker()
        
        # 4. Performance monitoring setup
        configs['performance-setup.js'] = self.generate_performance_setup()
        
        return configs
    
    def generate_vite_config_js(self) -> str:
        """Generate JavaScript Vite configuration"""
        config = self.optimizer.optimize_vite_config()
        return f"""
import {{ defineConfig }} from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({config});
"""
    
    def generate_bundle_analyzer_config(self) -> str:
        """Generate bundle analyzer configuration"""
        return """
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: 'dist/bundle-analysis.html',
      open: true,
      gzipSize: true,
      brotliSize: true
    })
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'chart-vendor': ['chart.js', 'react-chartjs-2'],
          'ui-vendor': ['@mui/material', '@mui/icons-material']
        }
      }
    }
  }
});
"""
    
    def generate_service_worker(self) -> str:
        """Generate service worker for caching"""
        return """
const CACHE_NAME = 'medical-ai-v1';
const urlsToCache = [
  '/',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/manifest.json',
  '/api/patient-data',
  '/api/vital-signs'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      }
    )
  );
});
"""
    
    def generate_performance_setup(self) -> str:
        """Generate performance monitoring setup"""
        return """
// Performance monitoring setup
export function setupPerformanceMonitoring() {
  // Monitor Core Web Vitals
  if ('PerformanceObserver' in window) {
    // First Contentful Paint
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-contentful-paint') {
          console.log('FCP:', entry.startTime);
        }
      }
    }).observe({ entryTypes: ['paint'] });

    // Largest Contentful Paint
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        console.log('LCP:', entry.startTime);
      }
    }).observe({ entryTypes: ['largest-contentful-paint'] });

    // First Input Delay
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        console.log('FID:', entry.processingStart - entry.startTime);
      }
    }).observe({ entryTypes: ['first-input'] });
  }

  // Monitor bundle sizes
  window.addEventListener('load', () => {
    const navigation = performance.getEntriesByType('navigation')[0];
    console.log('Load Time:', navigation.loadEventEnd - navigation.loadEventStart);
    console.log('DOM Content Loaded:', navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart);
  });
}
"""


async def main():
    """Example usage of frontend performance optimization"""
    
    # Create performance optimizer
    optimizer = MedicalAppOptimizer('/path/to/medical-ai-frontend')
    
    # Generate lazy-loaded components
    lazy_components = optimizer.generate_lazy_loading_components()
    
    # Generate performance hooks
    hooks = optimizer.generate_performance_optimized_hooks()
    
    # Generate optimized components
    components = optimizer.generate_optimized_medical_components()
    
    print("Generated lazy-loaded components:", list(lazy_components.keys()))
    print("Generated performance hooks:", list(hooks.keys()))
    print("Generated optimized components:", list(components.keys()))


if __name__ == "__main__":
    asyncio.run(main())