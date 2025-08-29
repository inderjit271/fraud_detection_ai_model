// XposeAI Dashboard JavaScript Application

class XposeAI {
    constructor() {
        this.init();
        this.setupEventListeners();
        this.updateStatus();
        this.storedSyntheticData = null;
    }

    init() {
        // Initialize range sliders
        this.initRangeSliders();
        
        // Initialize file upload
        this.initFileUpload();
        
        // Initialize navigation
        this.initNavigation();
        
        // Show first section by default
        this.showSection('upload');
    }

    initRangeSliders() {
        // Test size slider
        const testSizeRange = document.getElementById('test-size-range');
        const testSizeValue = document.getElementById('test-size-value');
        testSizeRange.addEventListener('input', (e) => {
            testSizeValue.textContent = e.target.value;
        });

        // Fraud ratio slider
        const fraudRatioRange = document.getElementById('fraud-ratio-range');
        const fraudRatioValue = document.getElementById('fraud-ratio-value');
        fraudRatioRange.addEventListener('input', (e) => {
            fraudRatioValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    initFileUpload() {
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');

        // Drag and drop functionality
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });
    }

    initNavigation() {
        document.querySelectorAll('[data-section]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.getAttribute('data-section');
                this.showSection(section);
                
                // Update active state
                document.querySelectorAll('[data-section]').forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
    }

    setupEventListeners() {
        // Analyze button
        document.getElementById('analyze-btn').addEventListener('click', () => {
            this.analyzeData();
        });

        // Model option toggles
        document.querySelectorAll('input[name="model-option"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.id === 'pretrained-option') {
                    document.getElementById('pretrained-section').classList.remove('d-none');
                    document.getElementById('training-section').classList.add('d-none');
                } else {
                    document.getElementById('pretrained-section').classList.add('d-none');
                    document.getElementById('training-section').classList.remove('d-none');
                }
            });
        });

        // Load pretrained model
        document.getElementById('load-pretrained-btn').addEventListener('click', () => {
            this.loadPretrainedModel();
        });

        // Train model
        document.getElementById('train-model-btn').addEventListener('click', () => {
            this.trainModel();
        });

        // Generate synthetic data
        document.getElementById('generate-synthetic-btn').addEventListener('click', () => {
            this.generateSyntheticData();
        });

        // Use synthetic data for training
        document.getElementById('use-synthetic-btn').addEventListener('click', () => {
            this.useSyntheticForTraining();
        });

        // Download synthetic data
        document.getElementById('download-synthetic-btn').addEventListener('click', () => {
            this.downloadSyntheticData();
        });

        // Run predictions
        document.getElementById('run-predictions-btn').addEventListener('click', () => {
            this.runPredictions();
        });
    }

    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.add('d-none');
        });

        // Show selected section
        document.getElementById(`${sectionName}-section`).classList.remove('d-none');
    }

    async handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showToast('Error: Please upload a CSV file', 'danger');
            return;
        }

        if (file.size > 50 * 1024 * 1024) {
            this.showToast('Error: File size must be less than 50MB', 'danger');
            return;
        }

        this.showProgress();
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload_csv', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.hideProgress();
                this.showDataPreview(result);
                this.setupColumnMapping(result.columns, result.detected_mapping);
                this.showToast(result.message, 'success');
                this.updateStatus();
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.hideProgress();
            this.showToast(`Upload failed: ${error.message}`, 'danger');
        }
    }

    showDataPreview(data) {
        const previewSection = document.getElementById('data-preview');
        const table = document.getElementById('preview-table');
        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');

        // Clear existing content
        thead.innerHTML = '';
        tbody.innerHTML = '';

        if (data.preview && data.preview.length > 0) {
            // Create header
            const headerRow = document.createElement('tr');
            Object.keys(data.preview[0]).forEach(key => {
                const th = document.createElement('th');
                th.textContent = key;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);

            // Create rows
            data.preview.forEach(row => {
                const tr = document.createElement('tr');
                Object.values(row).forEach(value => {
                    const td = document.createElement('td');
                    td.textContent = value !== null ? value : 'N/A';
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });

            previewSection.classList.remove('d-none');
        }
    }

    setupColumnMapping(columns, detectedMapping) {
        const mappingControls = document.getElementById('mapping-controls');
        const analyzeBtn = document.getElementById('analyze-btn');
        
        mappingControls.innerHTML = '';

        const requiredColumns = ['amount', 'fraud_label'];
        const optionalColumns = ['timestamp', 'merchant', 'location', 'card_type', 'user_id'];

        let hasAllRequired = true;

        [...requiredColumns, ...optionalColumns].forEach(targetCol => {
            const div = document.createElement('div');
            div.className = 'mb-2';

            const label = document.createElement('label');
            label.className = 'form-label';
            label.textContent = `${this.capitalizeFirst(targetCol.replace('_', ' '))}${requiredColumns.includes(targetCol) ? ' (Required)' : ' (Optional)'}:`;

            const select = document.createElement('select');
            select.className = 'form-select';
            select.id = `map-${targetCol}`;

            // Add empty option
            const emptyOption = document.createElement('option');
            emptyOption.value = '';
            emptyOption.textContent = '-- Select Column --';
            select.appendChild(emptyOption);

            // Add column options
            columns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                if (detectedMapping[targetCol] === col) {
                    option.selected = true;
                }
                select.appendChild(option);
            });

            div.appendChild(label);
            div.appendChild(select);
            mappingControls.appendChild(div);

            // Check if required columns are mapped
            if (requiredColumns.includes(targetCol) && !detectedMapping[targetCol]) {
                hasAllRequired = false;
            }
        });

        // Enable analyze button if mapping is valid
        analyzeBtn.disabled = !hasAllRequired;
        
        // Add change listeners to update button state
        mappingControls.querySelectorAll('select').forEach(select => {
            select.addEventListener('change', () => {
                const allRequiredMapped = requiredColumns.every(col => {
                    const selectElement = document.getElementById(`map-${col}`);
                    return selectElement && selectElement.value !== '';
                });
                analyzeBtn.disabled = !allRequiredMapped;
            });
        });

        document.getElementById('column-mapping').classList.remove('d-none');
    }

    async analyzeData() {
        this.showLoading('Analyzing data patterns...');

        try {
            const response = await fetch('/analyze_data', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (response.ok) {
                this.hideLoading();
                this.showAnalysisResults(result);
                this.showSection('analysis');
                this.showToast('Data analysis completed!', 'success');
                this.updateStatus();
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.hideLoading();
            this.showToast(`Analysis failed: ${error.message}`, 'danger');
        }
    }

    showAnalysisResults(data) {
        // Update metrics
        document.getElementById('total-transactions').textContent = data.metrics.total_transactions.toLocaleString();
        document.getElementById('fraud-transactions').textContent = data.metrics.fraud_transactions.toLocaleString();
        document.getElementById('fraud-rate').textContent = `${data.metrics.fraud_rate}%`;
        document.getElementById('avg-fraud-amount').textContent = `$${data.metrics.avg_fraud_amount.toLocaleString()}`;

        // Show charts
        if (data.charts.fraud_distribution) {
            Plotly.newPlot('fraud-distribution-chart', JSON.parse(data.charts.fraud_distribution));
        }
        
        if (data.charts.amount_distribution) {
            Plotly.newPlot('amount-distribution-chart', JSON.parse(data.charts.amount_distribution));
        }

        // Show fraud transactions table
        if (data.fraud_transactions && data.fraud_transactions.length > 0) {
            this.populateTable('fraud-table', data.fraud_transactions);
        }
    }

    async loadPretrainedModel() {
        this.showLoading('Loading pretrained model...');

        try {
            const response = await fetch('/load_pretrained', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (response.ok) {
                this.hideLoading();
                this.updateModelStatus('Pretrained model loaded');
                this.showToast(result.message, 'success');
                this.updateStatus();
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.hideLoading();
            this.showToast(`Model loading failed: ${error.message}`, 'danger');
        }
    }

    async trainModel() {
        const algorithm = document.getElementById('algorithm-select').value;
        const testSize = document.getElementById('test-size-range').value;
        const randomState = document.getElementById('random-state').value;
        const crossValidation = document.getElementById('cross-validation').checked;

        this.showLoading('Training model... This may take a few minutes.');

        const formData = new FormData();
        formData.append('algorithm', algorithm);
        formData.append('test_size', testSize);
        formData.append('random_state', randomState);
        formData.append('cross_validation', crossValidation);

        try {
            const response = await fetch('/train_model', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.hideLoading();
                this.updateModelStatus('Custom model trained', result.metrics);
                this.showModelMetrics(result.metrics);
                this.showToast(result.message, 'success');
                this.updateStatus();
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.hideLoading();
            this.showToast(`Training failed: ${error.message}`, 'danger');
        }
    }

    async generateSyntheticData() {
        const numSamples = document.getElementById('num-samples').value;
        const fraudRatio = document.getElementById('fraud-ratio-range').value;
        const datasetType = document.getElementById('dataset-type').value;
        const amountMin = document.getElementById('amount-min').value;
        const amountMax = document.getElementById('amount-max').value;
        const includeAnomalies = document.getElementById('include-anomalies').checked;
        const seasonalPatterns = document.getElementById('seasonal-patterns').checked;
        const correlationFeatures = document.getElementById('correlation-features').checked;

        this.showLoading('Generating synthetic data...');

        const formData = new FormData();
        formData.append('num_samples', numSamples);
        formData.append('fraud_ratio', fraudRatio);
        formData.append('dataset_type', datasetType);
        formData.append('amount_min', amountMin);
        formData.append('amount_max', amountMax);
        formData.append('include_anomalies', includeAnomalies);
        formData.append('seasonal_patterns', seasonalPatterns);
        formData.append('correlation_features', correlationFeatures);

        try {
            const response = await fetch('/generate_synthetic', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.hideLoading();
                this.showSyntheticResults(result);
                this.showToast(result.message, 'success');
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.hideLoading();
            this.showToast(`Generation failed: ${error.message}`, 'danger');
        }
    }

    showSyntheticResults(data) {
        // Update summary cards
        document.getElementById('synthetic-total').textContent = data.summary.total_samples.toLocaleString();
        document.getElementById('synthetic-fraud').textContent = data.summary.fraud_samples.toLocaleString();
        document.getElementById('synthetic-ratio').textContent = `${(data.summary.actual_fraud_ratio * 100).toFixed(2)}%`;

        // Show preview table
        if (data.preview) {
            this.populateTable('synthetic-table', data.preview);
        }

        // Store data for download and training
        this.storedSyntheticData = data.download_data;

        // Show results section
        document.getElementById('synthetic-results').classList.remove('d-none');
    }

    async useSyntheticForTraining() {
        if (!this.storedSyntheticData) {
            this.showToast('No synthetic data available', 'warning');
            return;
        }

        this.showLoading('Loading synthetic data for training...');

        try {
            const response = await fetch('/use_synthetic_for_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.storedSyntheticData)
            });

            const result = await response.json();
            
            if (response.ok) {
                this.hideLoading();
                this.showToast(result.message + ' Go to Model Training section.', 'success');
                this.updateStatus();
                
                // Switch to model training section
                setTimeout(() => {
                    this.showSection('model');
                    document.querySelectorAll('[data-section]').forEach(l => l.classList.remove('active'));
                    document.querySelector('[data-section="model"]').classList.add('active');
                }, 1000);
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.hideLoading();
            this.showToast(`Loading failed: ${error.message}`, 'danger');
        }
    }

    downloadSyntheticData() {
        if (!this.storedSyntheticData) {
            this.showToast('No synthetic data available', 'warning');
            return;
        }

        const csvContent = this.convertToCSV(this.storedSyntheticData);
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `synthetic_fraud_data_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        this.showToast('Synthetic data downloaded successfully!', 'success');
    }

    async runPredictions() {
        this.showLoading('Running predictions on current dataset...');

        try {
            const response = await fetch('/predict', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (response.ok) {
                this.hideLoading();
                this.showPredictionResults(result);
                this.showToast(result.message, 'success');
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.hideLoading();
            this.showToast(`Prediction failed: ${error.message}`, 'danger');
        }
    }

    showPredictionResults(data) {
        // Update metrics if available
        if (data.metrics && Object.keys(data.metrics).length > 0) {
            document.getElementById('pred-accuracy').textContent = data.metrics.accuracy || '-';
            document.getElementById('pred-precision').textContent = data.metrics.precision || '-';
            document.getElementById('pred-recall').textContent = data.metrics.recall || '-';
            document.getElementById('pred-f1').textContent = data.metrics.f1 || '-';
        }

        // Show high-risk transactions
        if (data.high_risk_transactions) {
            this.populateTable('predictions-table', data.high_risk_transactions);
        }

        // Show results section
        document.getElementById('prediction-results').classList.remove('d-none');
    }

    updateModelStatus(statusText, metrics = null) {
        document.getElementById('model-status-text').textContent = statusText;
        
        if (metrics) {
            this.showModelMetrics(metrics);
        }
    }

    showModelMetrics(metrics) {
        document.getElementById('metric-accuracy').textContent = metrics.accuracy;
        document.getElementById('metric-precision').textContent = metrics.precision;
        document.getElementById('metric-recall').textContent = metrics.recall;
        document.getElementById('metric-f1').textContent = metrics.f1;
        document.getElementById('model-metrics').classList.remove('d-none');
    }

    populateTable(tableId, data) {
        const table = document.getElementById(tableId);
        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');

        // Clear existing content
        thead.innerHTML = '';
        tbody.innerHTML = '';

        if (data && data.length > 0) {
            // Create header
            const headerRow = document.createElement('tr');
            Object.keys(data[0]).forEach(key => {
                const th = document.createElement('th');
                th.textContent = this.capitalizeFirst(key.replace('_', ' '));
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);

            // Create rows
            data.forEach(row => {
                const tr = document.createElement('tr');
                Object.values(row).forEach(value => {
                    const td = document.createElement('td');
                    if (typeof value === 'number' && value % 1 !== 0) {
                        td.textContent = parseFloat(value).toFixed(2);
                    } else {
                        td.textContent = value !== null ? value : 'N/A';
                    }
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
        }
    }

    convertToCSV(data) {
        if (!data || data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvRows = [headers.join(',')];
        
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header];
                return typeof value === 'string' ? `"${value}"` : value;
            });
            csvRows.push(values.join(','));
        });
        
        return csvRows.join('\n');
    }

    async updateStatus() {
        try {
            const response = await fetch('/status');
            const status = await response.json();
            
            // Update status badges
            document.getElementById('status-data').className = `badge ${status.data_uploaded ? 'bg-success' : 'bg-secondary'}`;
            document.getElementById('status-data').innerHTML = `<i class="bi bi-database"></i> ${status.data_uploaded ? 'Data Loaded' : 'No Data'}`;
            
            document.getElementById('status-model').className = `badge ${status.model_loaded ? 'bg-success' : 'bg-secondary'}`;
            document.getElementById('status-model').innerHTML = `<i class="bi bi-cpu"></i> ${status.model_loaded ? 'Model Ready' : 'No Model'}`;
            
            document.getElementById('status-analysis').className = `badge ${status.has_analysis ? 'bg-success' : 'bg-secondary'}`;
            document.getElementById('status-analysis').innerHTML = `<i class="bi bi-graph-up"></i> ${status.has_analysis ? 'Analysis Done' : 'No Analysis'}`;
            
            // Enable/disable predictions button
            document.getElementById('run-predictions-btn').disabled = !(status.data_uploaded && status.model_loaded);
            
        } catch (error) {
            console.error('Failed to update status:', error);
        }
    }

    showProgress() {
        document.getElementById('upload-progress').classList.remove('d-none');
        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = '100%';
    }

    hideProgress() {
        document.getElementById('upload-progress').classList.add('d-none');
    }

    showLoading(text = 'Processing...') {
        document.getElementById('loading-text').textContent = text;
        const loadingModal = new bootstrap.Modal(document.getElementById('loading-modal'));
        loadingModal.show();
    }

    hideLoading() {
        const loadingModal = bootstrap.Modal.getInstance(document.getElementById('loading-modal'));
        if (loadingModal) {
            loadingModal.hide();
        }
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const toastBody = toast.querySelector('.toast-body');
        const toastHeader = toast.querySelector('.toast-header');
        
        // Update content
        toastBody.textContent = message;
        
        // Update style based on type
        toast.className = `toast toast-${type}`;
        
        // Update icon
        const icon = toastHeader.querySelector('i');
        const iconClasses = {
            'success': 'bi bi-check-circle text-success',
            'danger': 'bi bi-exclamation-triangle text-danger',
            'warning': 'bi bi-exclamation-triangle text-warning',
            'info': 'bi bi-info-circle text-info'
        };
        icon.className = iconClasses[type] || iconClasses['info'];
        
        // Show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new XposeAI();
});