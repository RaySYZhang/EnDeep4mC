document.getElementById('sequenceForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const species = document.getElementById('speciesSelect').value;
    const sequence = document.getElementById('sequenceInput').value;
    const fileInput = document.getElementById('sequenceFile');
    const file = fileInput.files[0];
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<div class="loading">Analyzing sequences...</div>';

    if (!sequence && !file) {
        resultDiv.innerHTML = '<p class="error">Please enter a DNA sequence or upload a file.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('species', species);
    if (sequence) formData.append('sequence', sequence);
    if (file) formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
            return;
        }
        
        // Generate download file content
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const downloadContent = data.results.map((item) => 
            `>${item.seq_id}\n${item.sequence}\nProbability: ${(item.probability * 100).toFixed(2)}%\nPrediction: ${item.is_4mC_site ? '4mC Site' : 'Negative'}\n`
        ).join('\n');
        
        // Create download link
        const blob = new Blob([downloadContent], {type: 'text/plain'});
        const url = URL.createObjectURL(blob);
        
        let resultHtml = `
            <div class="model-info">
                <h2>🧬 Prediction Report</h2>
                <div class="info-item">
                    <span class="info-label">Selected Species:</span>
                    <span class="info-value">${species.replace('4mC_','')}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Ensemble Model:</span>
                    <span class="info-value">CNN + BLSTM + Transformer</span>
                </div>
                ${data.results.length > 10 ? 
                    `<div class="info-item">
                        <span class="info-label">Displaying:</span>
                        <span class="info-value">First 10 of ${data.results.length} results</span>
                        <a href="${url}" download="prediction_${timestamp}.txt" class="download-btn">
                            📥 Download Full Report
                        </a>
                    </div>` : ''}
            </div>
            <div class="results-header">
                <span class="header-id">Sequence ID</span>
                <span class="header-preview">Preview (41bp)</span>
                <span class="header-prob">Probability</span>
                <span class="header-pred">Prediction</span>
            </div>
            <ul class="results-list">`;
        
        // Only display the first 10 results
        const displayResults = data.results.slice(0, 10);
        
        displayResults.forEach((item, index) => {
            const probPercent = (item.probability * 100).toFixed(1);
            const probBar = `<div class="prob-bar" style="width: ${probPercent}%"></div>`;
            const sequenceID = item.seq_id || `Sequence_${index+1}`;
            
            resultHtml += `
                <li class="result-item">
                    <div class="sequence-id">${item.seq_id}</div>
                    <div class="sequence-preview">${item.sequence}</div>
                    <div class="probability">
                        ${probBar}
                        <span class="prob-value">${probPercent}%</span>
                    </div>
                    <div class="prediction ${item.is_4mC_site ? 'positive' : 'negative'}">
                        ${item.is_4mC_site ? '4mC Site ✔️' : 'Negative ❌'}
                    </div>
                </li>`;
        });
        
        resultHtml += `</ul>`;
        resultDiv.innerHTML = resultHtml;
    })
    .catch(error => {
        resultDiv.innerHTML = `<p class="error">Network Error: ${error}</p>`;
    });
});