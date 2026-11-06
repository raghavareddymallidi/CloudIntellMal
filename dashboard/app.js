const { useState, useEffect, useRef } = React;

// --- Chart Components ---

const LineChart = ({ data, title, color }) => {
    const chartRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!canvasRef.current) return;
        
        if (chartRef.current) {
            chartRef.current.destroy();
        }

        const ctx = canvasRef.current.getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, `${color}80`);
        gradient.addColorStop(1, `${color}00`);

        chartRef.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: data.length}, (_, i) => i),
                datasets: [{
                    label: title,
                    data: data,
                    borderColor: color,
                    backgroundColor: gradient,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: { display: false },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Time Steps' } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                }
            }
        });

        return () => {
            if (chartRef.current) chartRef.current.destroy();
        };
    }, [data, title, color]);

    return <div className="chart-container"><canvas ref={canvasRef}></canvas></div>;
};

const MultiLineChart = ({ datasets, title, labels }) => {
    const chartRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!canvasRef.current || !datasets || datasets.length === 0) return;
        
        if (chartRef.current) {
            chartRef.current.destroy();
        }

        const ctx = canvasRef.current.getContext('2d');
        const colors = ['#00e5ff', '#ffea00', '#ff3d00']; // Cyan, Yellow, Red for phases

        const chartDatasets = datasets.map((data, index) => ({
            label: labels[index],
            data: data,
            borderColor: colors[index % colors.length],
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.1
        }));

        chartRef.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: datasets[0].length}, (_, i) => i),
                datasets: chartDatasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: { 
                        position: 'top',
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                }
            }
        });

        return () => {
            if (chartRef.current) chartRef.current.destroy();
        };
    }, [datasets, title, labels]);

    return <div className="chart-container"><canvas ref={canvasRef}></canvas></div>;
};


// --- Main Application ---

const App = () => {
    const [currentView, setCurrentView] = useState('overview');
    const [performanceData, setPerformanceData] = useState(null);
    const [comparisonData, setComparisonData] = useState(null);
    const [inferenceData, setInferenceData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [selectedModel, setSelectedModel] = useState('TCN');

    const API_URL = (() => {
        if (typeof window !== 'undefined' && window.API_BASE_URL) {
            return window.API_BASE_URL.replace(/\/+$/, '');
        }
        if (typeof window !== 'undefined' && window.location) {
            const { hostname, port } = window.location;
            const isLocal = hostname === '127.0.0.1' || hostname === 'localhost';
            if (isLocal && port && port !== '8000') {
                return 'http://127.0.0.1:8000/api';
            }
        }
        return '/api';
    })();

    useEffect(() => {
        fetchData();
        runInference();
    }, []);

    const fetchData = async () => {
        try {
            const perfRes = await fetch(`${API_URL}/results/performance`);
            const perfJson = await perfRes.json();
            setPerformanceData(perfJson);

            const compRes = await fetch(`${API_URL}/results/comparison`);
            const compJson = await compRes.json();
            setComparisonData(compJson);
        } catch (error) {
            console.error("Error fetching data:", error);
        }
    };

    const runInference = async (modelType = selectedModel) => {
        setLoading(true);
        try {
            const res = await fetch(`${API_URL}/inference/random?model_type=${modelType}`);
            const data = await res.json();
            setInferenceData(data);
        } catch (error) {
            console.error("Error running inference:", error);
        }
        setLoading(false);
    };

    const renderSidebar = () => (
        <div className="sidebar">
            <div className="sidebar-header">
                <h2><i className="fa-solid fa-bolt"></i> TCN Microgrid</h2>
            </div>
            <div className="nav-menu">
                <div className={`nav-item ${currentView === 'overview' ? 'active' : ''}`} onClick={() => setCurrentView('overview')}>
                    <i className="fa-solid fa-chart-pie"></i> Overview
                </div>
                <div className={`nav-item ${currentView === 'waveforms' ? 'active' : ''}`} onClick={() => setCurrentView('waveforms')}>
                    <i className="fa-solid fa-wave-square"></i> Waveforms
                </div>
                <div className={`nav-item ${currentView === 'performance' ? 'active' : ''}`} onClick={() => setCurrentView('performance')}>
                    <i className="fa-solid fa-ranking-star"></i> Model Performance
                </div>
                <div className={`nav-item ${currentView === 'advanced' ? 'active' : ''}`} onClick={() => setCurrentView('advanced')}>
                    <i className="fa-solid fa-microscope"></i> Advanced Analysis
                </div>
            </div>
        </div>
    );

    const renderOverview = () => {
        if (!inferenceData) return <div className="loader"></div>;

        const isFault = inferenceData.predicted_detection === 'Fault';

        return (
            <div className="fade-in">
                <div className="header">
                    <h1>Dashboard Overview</h1>
                    <div style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
                        <select 
                            value={selectedModel} 
                            onChange={(e) => {
                                setSelectedModel(e.target.value);
                                runInference(e.target.value);
                            }}
                            style={{
                                background: 'rgba(30, 41, 59, 0.9)',
                                color: 'white',
                                border: '2px solid rgba(255, 255, 255, 0.2)',
                                padding: '1rem 1.5rem',
                                borderRadius: '12px',
                                outline: 'none',
                                cursor: 'pointer',
                                fontWeight: '600',
                                fontSize: '1.2rem',
                                minWidth: '250px'
                            }}
                        >
                            <option value="TCN">TCN (Recommended)</option>
                            <option value="CNN">1D CNN</option>
                            <option value="LSTM">LSTM</option>
                        </select>
                        <button className="button-primary" onClick={() => runInference(selectedModel)} disabled={loading} style={{fontSize: '1.2rem', padding: '1rem 1.5rem', borderRadius: '12px'}}>
                            {loading ? <div className="loader" style={{width: '20px', height: '20px', borderWidth: '3px'}}></div> : <><i className="fa-solid fa-rotate-right"></i> Run Random Sample</>}
                        </button>
                    </div>
                </div>

                {inferenceData.needs_review && (
                    <div className="alert-warning">
                        <i className="fa-solid fa-triangle-exclamation" style={{fontSize: '1.25rem'}}></i>
                        Low Confidence Prediction: Human Review Required!
                    </div>
                )}

                <div className="grid-overview">
                    <div className={`glass-panel stat-card ${isFault ? 'status-fault' : 'status-normal'}`}>
                        <span className="stat-label">System Status</span>
                        <span className={`stat-value ${isFault ? 'fault' : 'normal'}`}>
                            {isFault ? <><i className="fa-solid fa-triangle-exclamation"></i> FAULT DETECTED</> : <><i className="fa-solid fa-check-circle"></i> NORMAL</>}
                        </span>
                    </div>
                    
                    {isFault && (
                        <div className="glass-panel stat-card">
                            <span className="stat-label">Fault Type</span>
                            <span className="stat-value">{inferenceData.predicted_classification}</span>
                        </div>
                    )}

                    <div className="glass-panel stat-card">
                        <span className="stat-label">Detection Confidence</span>
                        <span className="stat-value">{(inferenceData.detection_confidence * 100).toFixed(2)}%</span>
                    </div>

                    <div className="glass-panel stat-card">
                        <span className="stat-label">Sample ID</span>
                        <span className="stat-value">#{inferenceData.sample_id}</span>
                    </div>
                </div>

                <div className="glass-panel" style={{marginBottom: '2rem'}}>
                    <h3>Three-Phase Voltages (Va, Vb, Vc)</h3>
                    <MultiLineChart 
                        datasets={[inferenceData.waveforms.Va, inferenceData.waveforms.Vb, inferenceData.waveforms.Vc]}
                        labels={['Va', 'Vb', 'Vc']}
                    />
                </div>
            </div>
        );
    };

    const renderWaveforms = () => {
        if (!inferenceData) return <div className="loader"></div>;
        return (
            <div className="fade-in">
                <div className="header">
                    <h1>Waveform Analysis</h1>
                    <div style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
                        <select 
                            value={selectedModel} 
                            onChange={(e) => {
                                setSelectedModel(e.target.value);
                                runInference(e.target.value);
                            }}
                            style={{
                                background: 'rgba(30, 41, 59, 0.9)',
                                color: 'white',
                                border: '2px solid rgba(255, 255, 255, 0.2)',
                                padding: '0.75rem 1rem',
                                borderRadius: '8px',
                                outline: 'none',
                                cursor: 'pointer',
                                fontWeight: '600',
                                fontSize: '1.1rem'
                            }}
                        >
                            <option value="TCN">TCN (Recommended)</option>
                            <option value="CNN">1D CNN</option>
                            <option value="LSTM">LSTM</option>
                        </select>
                        <button className="button-primary" onClick={() => runInference(selectedModel)} disabled={loading} style={{fontSize: '1.1rem'}}>
                            {loading ? <i className="fa-solid fa-spinner fa-spin"></i> : <i className="fa-solid fa-forward-step"></i>} Next Sample
                        </button>
                    </div>
                </div>
                
                <div className="grid-charts">
                    <div className="glass-panel">
                        <h3>Voltages</h3>
                        <MultiLineChart 
                            datasets={[inferenceData.waveforms.Va, inferenceData.waveforms.Vb, inferenceData.waveforms.Vc]}
                            labels={['Va', 'Vb', 'Vc']}
                        />
                    </div>
                    <div className="glass-panel">
                        <h3>Currents</h3>
                        <MultiLineChart 
                            datasets={[inferenceData.waveforms.Ia, inferenceData.waveforms.Ib, inferenceData.waveforms.Ic]}
                            labels={['Ia', 'Ib', 'Ic']}
                        />
                    </div>
                </div>
            </div>
        );
    };

    const renderPerformance = () => {
        if (!performanceData || !comparisonData) return <div className="loader"></div>;

        return (
            <div className="fade-in">
                <div className="header">
                    <h1>Model Performance Comparison</h1>
                </div>

                <div className="grid-overview" style={{gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))'}}>
                    <div className="glass-panel stat-card status-normal">
                        <span className="stat-label">TCN Det. Accuracy</span>
                        <span className="stat-value highlight">{(performanceData.detection.accuracy * 100).toFixed(2)}%</span>
                    </div>
                    <div className="glass-panel stat-card status-normal">
                        <span className="stat-label">TCN Cls. Accuracy</span>
                        <span className="stat-value highlight">{(performanceData.classification.accuracy * 100).toFixed(2)}%</span>
                    </div>
                    <div className="glass-panel stat-card">
                        <span className="stat-label">Inference Time</span>
                        <span className="stat-value highlight">{performanceData.inference.per_sample_ms.toFixed(3)} ms</span>
                    </div>
                </div>

                <div className="glass-panel" style={{marginTop: '2rem'}}>
                    <h3>Model Comparison (Classification Accuracy)</h3>
                    <div style={{display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap'}}>
                        {comparisonData.map((model, idx) => (
                            <div key={idx} style={{flex: 1, minWidth: '120px', padding: '1rem', background: 'rgba(255,255,255,0.03)', border: '1px solid #333', textAlign: 'center'}}>
                                <div style={{color: '#9e9e9e', fontSize: '0.9rem', marginBottom: '0.5rem', textTransform: 'uppercase'}}>{model.model}</div>
                                <div style={{fontFamily: "'Roboto Mono', monospace", fontSize: '1.5rem', fontWeight: 'bold', color: model.model === 'TCN' ? '#00e5ff' : '#e0e0e0', textShadow: model.model === 'TCN' ? '0 0 10px rgba(0,229,255,0.4)' : 'none'}}>
                                    {(model.cls_accuracy * 100).toFixed(1)}%
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    };

    const renderAdvanced = () => {
        return (
            <div className="fade-in">
                <div className="header">
                    <h1>Advanced Analysis & Evaluation</h1>
                </div>

                <div className="grid-charts" style={{gap: '3rem'}}>
                    <div className="glass-panel">
                        <h3 style={{marginBottom: '1rem'}}><i className="fa-solid fa-eye"></i> Explainability (XAI) - Integrated Gradients</h3>
                        <p style={{color: 'var(--text-secondary)', marginBottom: '1.5rem'}}>
                            Visualizing exactly which parts of the Va, Vb, and Vc waveforms the TCN pays the most attention to when detecting a fault.
                        </p>
                        <img src={`${API_URL}/static/results/xai_attributions.png`} alt="XAI Attributions" style={{width: '100%', borderRadius: '8px'}} />
                    </div>

                    <div className="glass-panel">
                        <h3 style={{marginBottom: '1rem'}}><i className="fa-solid fa-bolt-lightning"></i> High-Impedance Fault (HIF) Testing</h3>
                        <p style={{color: 'var(--text-secondary)', marginBottom: '1.5rem'}}>
                            Model accuracy binned by physical fault resistance to prove robustness against challenging high-impedance scenarios.
                        </p>
                        <img src={`${API_URL}/static/results/hif_performance.png`} alt="HIF Performance" style={{width: '100%', borderRadius: '8px', maxWidth: '800px', display: 'block', margin: '0 auto'}} />
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="dashboard-container">
            {renderSidebar()}
            <main className="main-content">
                {currentView === 'overview' && renderOverview()}
                {currentView === 'waveforms' && renderWaveforms()}
                {currentView === 'performance' && renderPerformance()}
                {currentView === 'advanced' && renderAdvanced()}
            </main>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
